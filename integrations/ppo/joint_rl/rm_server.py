"""Serve the frozen fine-tuned Robometer with its native image/token protocol."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def prepare_robometer_inputs(processor, pictures, instruction, config, device):
    """Reproduce RobometerCollator's prompt, progress tokens and final-frame target."""
    import torch
    from dual_rm.robometer import prompt_messages

    row = {"instruction": instruction, "frames": list(range(len(pictures)))}
    text = processor.apply_chat_template(
        prompt_messages(row), tokenize=False, add_generation_prompt=False,
        add_vision_id=True, enable_thinking=False, fps=1,
    )
    batch = dict(processor(text=[text], images=pictures, padding=True, truncation=False, return_tensors="pt"))
    batch.pop("token_type_ids", None)
    if batch["input_ids"].shape[-1] > config["max_tokens"]:
        raise ValueError("multimodal_sequence_over_budget")
    token = processor.tokenizer.convert_tokens_to_ids("<|prog_token|>")
    if not torch.all((batch["input_ids"] == token).sum(-1) == len(pictures)):
        raise ValueError("Robometer_progress_token_frame_mismatch")
    return {"groups": [{"indices": torch.tensor([0]), "frame_count": len(pictures),
                        "batch": {key: value.to(device) for key, value in batch.items()}}], "batch_size": 1}


def load_scorer(checkpoint: Path, expected_hash: str, device: str):
    import torch
    from PIL import Image
    from qwen_rm.checkpoints import model_fingerprint
    from dual_rm.runtime import build_runtime

    if model_fingerprint(checkpoint) != expected_hash:
        raise ValueError("Robometer_checkpoint_hash_changed")
    metadata = json.loads((checkpoint / "reward_model.json").read_text())
    config = metadata["config"]
    if (config["backend"], config["method"], config["mode"]) != ("robometer", "at", "history"):
        raise ValueError("expected_finetuned_history_Robometer")
    if config.get("input_frame_budget", 8) != 8:
        raise ValueError("expected_history8")
    model, processor, _ = build_runtime(
        config, metadata["ontology"], Path(os.environ["JOINT_RL_RUN"]) / "rm/unused_cache", checkpoint / "processor"
    )
    model.load_weights(checkpoint)
    model.to(device).eval().requires_grad_(False)
    # The wrapper restores the native progress/success heads and trained LoRA.
    def score(samples):
        results = []
        with torch.inference_mode():
            for sample in samples:
                if not isinstance(sample["instruction"], str) or not sample["instruction"].strip():
                    raise ValueError("instruction_required")
                if not 1 <= len(sample["frames"]) <= 8:
                    raise ValueError("history_budget_exceeded")
                pictures = []
                for data in sample["frames"]:
                    with Image.open(io.BytesIO(base64.b64decode(data, validate=True))) as picture:
                        pictures.append(picture.convert("RGB").copy())
                inputs = prepare_robometer_inputs(processor, pictures, sample["instruction"], config, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**inputs)
                results.append({"A": float(logits["A"].float().sigmoid()[0]), "T": float(logits["T"].float().sigmoid()[0])})
        return results
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--port", type=int, default=18876)
    args = parser.parse_args()
    scorer = load_scorer(args.checkpoint, args.checkpoint_sha256, "cuda:0")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200 if self.path == "/health" else 404,
                       {"ready": True, "checkpoint_sha256": args.checkpoint_sha256, "heads": ["A", "T"]})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/score" or not 0 < length <= 64 * 1024 * 1024:
                    raise ValueError("invalid_score_request")
                request = json.loads(self.rfile.read(length))
                if request["checkpoint_sha256"] != args.checkpoint_sha256:
                    raise ValueError("request_checkpoint_mismatch")
                if not 1 <= len(request["samples"]) <= 64:
                    raise ValueError("invalid_batch_size")
                self.reply(200, {"scores": scorer(request["samples"]), "checkpoint_sha256": args.checkpoint_sha256})
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.reply(500, {"error": type(error).__name__ + ": " + str(error)})

    print(json.dumps({"event": "Robometer_RM_ready", "port": args.port, "checkpoint": str(args.checkpoint)}), flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
