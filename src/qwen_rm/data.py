from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
import time
from contextlib import contextmanager

from .contracts import SPLITS, causal_frames, targets, validate_sample


@contextmanager
def cache_lock(path, timeout=300.):
    import fcntl
    # Keep the inode stable for waiters. The OS releases flock on process exit,
    # including SIGTERM, so an interrupted decode cannot leave a stale lock.
    with Path(path).open('a+b') as stream:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('frame_cache_lock_wait_timeout')
                time.sleep(.1)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def atomic_cache_path(destination):
    # Cross-node writers may not share flock semantics on the mounted cache.
    # Each writer owns a distinct temporary file, even when publishing the same
    # deterministic RGB frame or manifest. Publish only after the file closes.
    destination = Path(destination)
    fd, name = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class FrameCache:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)

    def paths(self, row):
        if os.environ.get("FRAME_RM_VERIFIED_CACHE_INDEX"):
            from .frozen_cache import verified_paths
            return verified_paths(row)
        video = Path(row["video"]); stat = video.stat()
        if stat.st_size != row["video_bytes"] or stat.st_mtime_ns != row["video_mtime_ns"]:
            raise ValueError("source_video_changed_after_snapshot")
        key = row["media_key"]
        shared = os.environ.get("FRAME_RM_READONLY_HISTORY8_CACHE")
        if shared:
            source = Path(shared) / "history_8" / key
            marker = source / "complete.json"
            if marker.is_file():
                saved = json.loads(marker.read_text())
                if set(row["frames"]).issubset(saved.get("frames", [])):
                    paths = [source / ("frame_%06d.jpg" % f) for f in row["frames"]]
                    if all(p.is_file() for p in paths):
                        return paths
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid_media_cache_key")
        budget = row.get("history_max_frames", 16)
        cache_root = self.root if budget == 16 else self.root / ("history_%d" % budget)
        cache_root.mkdir(parents=True, exist_ok=True)
        folder = cache_root / key; manifest = folder / "complete.json"
        anchors = targets(row["frame_count"])
        wanted = sorted({f for current in anchors for f in causal_frames(range(current + 1), current, "history", budget)})
        lock = cache_root / (key + ".lock")
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved['frames'] != wanted:
                raise ValueError('cached_frames_mismatch')
            return [folder / ('frame_%06d.jpg' % f) for f in row['frames']]
        with cache_lock(lock):
            if not manifest.exists():
                import cv2
                from PIL import Image
                cv2.setNumThreads(1)
                folder.mkdir(exist_ok=True)
                capture = cv2.VideoCapture(str(video)); remaining = set(wanted); index = 0
                if not capture.isOpened():
                    raise ValueError("video_decoder_open_failed")
                try:
                    while remaining:
                        ok, pixels = capture.read()
                        if not ok:
                            raise ValueError("video_missing_exact_target_frame:" + str(min(remaining)))
                        if index in remaining:
                            rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
                            destination = folder / ("frame_%06d.jpg" % index)
                            with atomic_cache_path(destination) as temporary:
                                Image.fromarray(rgb).save(temporary, format="JPEG", quality=95)
                            remaining.remove(index)
                        index += 1
                finally:
                    capture.release()
                with atomic_cache_path(manifest) as temporary:
                    temporary.write_text(json.dumps({"frames": wanted, "video": str(video)}))
        return [folder / ("frame_%06d.jpg" % f) for f in row["frames"]]


class SampleDataset:
    def __init__(self, snapshot, split, mode, max_samples=0):
        if split not in SPLITS or mode not in ("single", "history"):
            raise ValueError("invalid_dataset_selection")
        self.snapshot = Path(snapshot).resolve()
        filename = mode + "_" + split + ".jsonl"
        frozen = json.loads((self.snapshot / "snapshot.json").read_text(encoding="utf-8"))
        if hashlib.sha256((self.snapshot / filename).read_bytes()).hexdigest() != frozen["files_sha256"][filename]:
            raise ValueError("frozen_sample_file_changed")
        self.rows = [validate_sample(json.loads(line)) for line in
                     (self.snapshot / filename).read_text(encoding="utf-8").splitlines() if line]
        if any(row["split"] != split or row["mode"] != mode for row in self.rows):
            raise ValueError("sample_file_split_mismatch")
        self.history_max_frames = frozen.get("history_max_frames", 16)
        if any(row.get("history_max_frames", 16) != self.history_max_frames for row in self.rows):
            raise ValueError("sample_file_history_budget_mismatch")
        if max_samples:
            # Keep terminal and intermediate rows during a smoke test.
            first, terminal = self.rows[:max_samples], [r for r in self.rows if r["targets"]["T"]["mask"]]
            self.rows = list({r["id"]: r for r in first + terminal[:max(1, max_samples // 2)]}.values())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def prompt_messages(row):
    # Neither outcome, annotation text, failure labels, state vectors nor total
    # video duration enter the model's input. The target frame is always last.
    content = [{"type": "text", "text": "Robot task: " + row["instruction"]}]
    for index, frame in enumerate(row["frames"]):
        # Absolute frame/time values at terminal inputs would expose the strong
        # success-vs-timeout length shortcut in this dataset.
        if row["mode"] == "history":
            content.append({"type": "text", "text": "Observed view %d, in chronological order:" % (index + 1)})
        content.append({"type": "image"})
    content.append({"type": "text", "text": "Assess the physical progress and visible execution state at the last observed frame."})
    return [{"role": "user", "content": content}]


class Collator:
    def __init__(self, processor, frame_cache, max_tokens=8192):
        self.processor = processor; self.cache = FrameCache(frame_cache); self.max_tokens = max_tokens

    def __call__(self, rows):
        import torch
        from PIL import Image
        images, texts = [], []
        for row in rows:
            texts.append(self.processor.apply_chat_template(prompt_messages(row), tokenize=False, add_generation_prompt=True))
            for path in self.cache.paths(row):
                with Image.open(path) as pic:
                    images.append(pic.convert("RGB").copy())
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        batch.pop("token_type_ids", None)
        if batch["input_ids"].shape[-1] > self.max_tokens:
            raise ValueError("multimodal_sequence_over_budget_reduce_pixels_or_frames_do_not_truncate")
        supervised = supervised_targets(rows)
        return {"inputs": dict(batch), "targets": supervised,
                "row_ids": [r["id"] for r in rows], "case_ids": [r["case_id"] for r in rows],
                "frames": [r["frame"] for r in rows]}


def supervised_targets(rows):
    import torch
    supervised = {}
    for head in ('A', 'T'):
        supervised[head + '_value'] = torch.tensor([r['targets'][head]['value'] for r in rows], dtype=torch.float32)
        supervised[head + '_mask'] = torch.tensor([r['targets'][head]['mask'] for r in rows], dtype=torch.bool)
    for head in ('failure_actions', 'failure_types', 'failure_results'):
        supervised[head + '_value'] = torch.tensor([r['targets'][head]['values'] for r in rows], dtype=torch.float32)
        supervised[head + '_mask'] = torch.tensor([r['targets'][head]['mask'] for r in rows], dtype=torch.bool)
    supervised['recovery_value'] = torch.tensor([r['targets']['recovery']['value'] for r in rows], dtype=torch.long)
    supervised['recovery_mask'] = torch.tensor([r['targets']['recovery']['mask'] for r in rows], dtype=torch.bool)
    return supervised
