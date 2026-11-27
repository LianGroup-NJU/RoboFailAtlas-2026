"""Serve the actual joint RM in its own, compatible Transformers environment."""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from numbers import Real
from pathlib import Path


INTERFACE_VERSION = "joint-structured-v1"
HEADS = ("A", "T", "failure_actions", "failure_types", "failure_results", "recovery")
MULTILABEL_HEADS = ("failure_actions", "failure_types", "failure_results")


def _sha256(value, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid_" + name)
    return value


def _ontology_labels(ontology: Mapping) -> dict[str, tuple[str, ...]]:
    """Validate the ordered class names used to label each probability vector."""
    if not isinstance(ontology, Mapping):
        raise ValueError("invalid_ontology")
    _sha256(ontology.get("sha256"), "ontology_sha256")
    result = {}
    for head in (*MULTILABEL_HEADS, "recovery"):
        labels = ontology.get(head)
        if (not isinstance(labels, (list, tuple)) or not labels
                or any(not isinstance(code, str) or not code.strip() for code in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError("invalid_ontology_labels:" + head)
        result[head] = tuple(labels)
    return result


def build_health(checkpoint_sha256: str, ontology: Mapping) -> dict:
    """Describe the exact structured scoring contract without loading a model."""
    _ontology_labels(ontology)
    return {"ready": True, "interface_version": INTERFACE_VERSION,
            "heads": list(HEADS), "ontology_sha256": ontology["sha256"],
            "checkpoint_sha256": _sha256(checkpoint_sha256, "checkpoint_sha256")}


def _python_values(value):
    # Importing this module and testing serialization never imports torch.
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def _finite_number(value, head: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("nonfinite_or_nonnumeric_logit:" + head)
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("nonfinite_or_nonnumeric_logit:" + head) from error
    if not math.isfinite(numeric):
        raise ValueError("nonfinite_or_nonnumeric_logit:" + head)
    return numeric


def _sigmoid(value: float) -> float:
    # Avoid overflow for legitimate large-magnitude, finite logits.
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def serialize_logits(logits: Mapping, ontology: Mapping) -> dict:
    """Serialize one forward result using sigmoid and recovery softmax.

    A/T must have shape [1]. The auxiliary heads must have shape [1, C],
    where C exactly matches their ordered checkpoint ontology. Python lists,
    tuples, NumPy arrays and detached torch tensors are supported.
    """
    labels = _ontology_labels(ontology)
    if not isinstance(logits, Mapping) or set(logits) != set(HEADS):
        raise ValueError("RM_logit_heads_mismatch")
    vectors = {}
    for head in HEADS:
        value = _python_values(logits[head])
        if not isinstance(value, (list, tuple)) or len(value) != 1:
            raise ValueError("RM_logit_batch_shape_mismatch:" + head)
        if head in ("A", "T"):
            vectors[head] = [_finite_number(value[0], head)]
        else:
            row = value[0]
            if not isinstance(row, (list, tuple)) or len(row) != len(labels[head]):
                raise ValueError("RM_logit_class_shape_mismatch:" + head)
            vectors[head] = [_finite_number(item, head) for item in row]
    result = {head: _sigmoid(vectors[head][0]) for head in ("A", "T")}
    for head in MULTILABEL_HEADS:
        result[head + "_probabilities"] = {
            code: _sigmoid(value) for code, value in zip(labels[head], vectors[head], strict=True)
        }
    recovery = vectors["recovery"]
    maximum = max(recovery)
    exponents = [math.exp(value - maximum) for value in recovery]
    denominator = sum(exponents)
    result["recovery_probabilities"] = {
        code: value / denominator for code, value in zip(labels["recovery"], exponents, strict=True)
    }
    probabilities = [result["A"], result["T"],
                     *(p for head in (*MULTILABEL_HEADS, "recovery")
                       for p in result[head + "_probabilities"].values())]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("invalid_serialized_RM_probability")
    return result


def load_scorer(checkpoint: Path, expected_hash: str, device: str):
    import torch
    from PIL import Image
    from qwen_rm.checkpoints import model_fingerprint
    from qwen_rm.data import prompt_messages
    from dual_rm.runtime import build_runtime

    if model_fingerprint(checkpoint) != expected_hash:
        raise ValueError("joint_RM_checkpoint_hash_changed")
    metadata = json.loads((checkpoint / "reward_model.json").read_text())
    build_health(expected_hash, metadata["ontology"])
    config = metadata["config"]
    if config["method"] != "joint" or config["mode"] != "history":
        raise ValueError("expected_joint_history_checkpoint")
    if config.get("input_frame_budget", 8) != 8:
        raise ValueError("expected_history8")
    model, processor, _ = build_runtime(
        config, metadata["ontology"], Path(os.environ["JOINT_RL_RUN"]) / "rm/unused_cache", checkpoint / "processor"
    )
    model.load_weights(checkpoint)
    model.to(device).eval().requires_grad_(False)
    # Keep exactly the prompt, processor and sigmoid heads used by qwen_rm.predict.
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
                row = {"instruction": sample["instruction"], "mode": "history", "frames": list(range(len(pictures)))}
                text = processor.apply_chat_template(prompt_messages(row), tokenize=False, add_generation_prompt=True)
                batch = processor(text=[text], images=pictures, padding=True, return_tensors="pt")
                batch.pop("token_type_ids", None)
                if batch["input_ids"].shape[-1] > config["max_tokens"]:
                    raise ValueError("multimodal_sequence_over_budget")
                inputs = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**inputs)
                result = serialize_logits(logits, metadata["ontology"])
                # Retain the original torch float32 A/T activation exactly, so
                # adding auxiliary outputs does not change baseline score math.
                result["A"] = float(logits["A"].float().sigmoid()[0])
                result["T"] = float(logits["T"].float().sigmoid()[0])
                results.append(result)
        return results
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--port", type=int, default=18876)
    args = parser.parse_args()
    metadata = json.loads((args.checkpoint / "reward_model.json").read_text())
    health = build_health(args.checkpoint_sha256, metadata["ontology"])
    scorer = load_scorer(args.checkpoint, args.checkpoint_sha256, "cuda:0")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200 if self.path == "/health" else 404,
                       health if self.path == "/health" else {"error": "not_found"})

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
                self.reply(200, {"scores": scorer(request["samples"]),
                                 "checkpoint_sha256": args.checkpoint_sha256,
                                 "interface_version": INTERFACE_VERSION,
                                 "ontology_sha256": health["ontology_sha256"]})
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.reply(500, {"error": type(error).__name__ + ": " + str(error)})

    print(json.dumps({"event": "joint_RM_ready", "port": args.port, "checkpoint": str(args.checkpoint)}), flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
