from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from qwen_rm.data import SampleDataset, FrameCache, supervised_targets
from dual_rm.frame_protocol import effective_row
from .contracts import native_answer, REASONS, RUBRIC, requires_reason_word, SCORE_ONLY_SYSTEM


class NativeDataset(SampleDataset):
    def __init__(self, snapshot, split, config, max_samples=0):
        super().__init__(snapshot, split, config['mode'], max_samples)
        self.skipped = Counter()
        reasons = {}
        if requires_reason_word(config):
            path = config.get('native_reason_labels')
            expected = config.get('native_reason_labels_sha256')
            if not path or not expected:
                raise ValueError('native_thinking_SFT_requires_explicit_point_level_reason_labels_use_AT_adapter_if_unavailable')
            raw = Path(path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('native_reason_labels_changed')
            for line in raw.splitlines():
                item = json.loads(line)
                if item['id'] in reasons or item['reason_word'] not in REASONS:
                    raise ValueError('duplicate_or_invalid_native_reason_label')
                reasons[item['id']] = item['reason_word']
        accepted = []
        for source in self.rows:
            row = dict(source)
            if row['id'] in reasons:
                row['native_reason_word'] = reasons[row['id']]
            answer = native_answer(row, config)
            if answer is None:
                key = 'T' if config.get('native_target') == 'terminal_T' else 'A'
                self.skipped['unknown_or_unavailable_' + key if not row['targets'][key]['mask'] else 'unknown_reason_word'] += 1
            else:
                accepted.append(row)
        self.rows = accepted


class NativeCollator:
    def __init__(self, processor, frame_cache, config):
        self.processor, self.config, self.cache = processor, config, FrameCache(frame_cache)

    def prepare(self, rows):
        from PIL import Image
        prepared = []
        for original in rows:
            row = effective_row(original, self.config['input_frame_budget'], self.config.get('frame_sampling', 'uniform_prefix'))
            images = []
            for path in self.cache.paths(row):
                with Image.open(path) as pic:
                    images.append(pic.convert('RGB').copy())
            if self.config['model_key'] == 'densereward':
                images = [images[0]] * (3 - len(images)) + images
            if self.config['backend'] == 'roboreward' and len(images) == 1:
                # Qwen3 video patching needs at least two frames. Both are the
                # same observed image; no future observation is introduced.
                images = [images[0], images[0]]
            prepared.append((row, images))
        return prepared

    def messages(self, row, images):
        if self.config['backend'] == 'roboreward':
            return [{'role': 'user', 'content': [{'type': 'video'},
                {'type': 'text', 'text': RUBRIC.format(task=row['instruction'])}]}]
        system = (Path(self.config['model_name_or_path']) / 'system_prompt.txt').read_text()
        if self.config['model_key'] == 'densereward' and not requires_reason_word(self.config):
            system = SCORE_ONLY_SYSTEM
        return [{'role': 'system', 'content': system},
                {'role': 'user', 'content': [{'type': 'image'} for _ in images] +
                 [{'type': 'text', 'text': row['instruction']}]}]

    def encode(self, texts, prepared):
        import numpy as np
        kwargs = dict(text=texts, padding=True, truncation=False, return_tensors='pt')
        if self.config['backend'] == 'roboreward':
            # Use a fixed relative clock. No total episode length, absolute
            # target frame, outcome or timeout duration enters the video prompt.
            videos = [np.stack([np.asarray(image) for image in images]) for _, images in prepared]
            kwargs.update(videos=videos, do_sample_frames=False,
                video_metadata=[{'fps': 1., 'total_num_frames': len(images),
                                 'frames_indices': list(range(len(images)))} for _, images in prepared])
        else:
            kwargs['images'] = [im for _, images in prepared for im in images]
        encoded = dict(self.processor(**kwargs))
        encoded.pop('token_type_ids', None)
        if encoded['input_ids'].shape[-1] > self.config['max_tokens']:
            raise ValueError('native_multimodal_sequence_over_budget_do_not_truncate')
        return encoded

    def inference(self, rows):
        prepared = self.prepare(rows)
        texts = [self.processor.apply_chat_template(self.messages(row, images), tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for row, images in prepared]
        return self.encode(texts, prepared)

    def __call__(self, rows):
        import torch
        prepared = self.prepare(rows)
        prompts, complete = [], []
        for row, images in prepared:
            messages = self.messages(row, images)
            answer = native_answer(row, self.config)
            if answer is None:
                raise ValueError('native_SFT_row_has_no_known_answer')
            prompts.append(self.processor.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False))
            complete.append(self.processor.apply_chat_template(messages + [{'role': 'assistant', 'content': answer}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False))
        batch, prefix = self.encode(complete, prepared), self.encode(prompts, prepared)
        labels = torch.full_like(batch['input_ids'], -100)
        for i in range(len(rows)):
            full = batch['input_ids'][i][batch['attention_mask'][i].bool()]
            before = prefix['input_ids'][i][prefix['attention_mask'][i].bool()]
            if len(full) <= len(before) or not torch.equal(full[:len(before)], before):
                raise ValueError('assistant_boundary_tokenization_mismatch')
            indices = batch['attention_mask'][i].nonzero().flatten()[len(before):]
            labels[i, indices] = batch['input_ids'][i, indices]
        batch['labels'] = labels
        return {'inputs': batch, 'targets': supervised_targets(rows),
                'row_ids': [r['id'] for r in rows], 'case_ids': [r['case_id'] for r in rows],
                'frames': [r['frame'] for r in rows]}
