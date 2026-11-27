"""Causal history selection and the fixed A/T reward definition."""
from __future__ import annotations

import base64
import io
import json
import math
import urllib.request


def history_indices(count: int, budget: int = 8) -> list[int]:
    """Select uniform observed frames, including the current frame, without padding."""
    if count < 1 or budget < 2:
        raise ValueError("invalid_history_size")
    if count <= budget:
        return list(range(count))
    return [round(i * (count - 1) / (budget - 1)) for i in range(budget)]


def encode_frame(pixels) -> str:
    from PIL import Image

    image = Image.fromarray(pixels)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95)
    return base64.b64encode(output.getvalue()).decode("ascii")


def at_reward(previous_a: float, a: float, t: float, terminal: bool) -> float:
    """Use signed progress change, plus the terminal success probability once."""
    for value in (previous_a, a, t):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("nonfinite_or_out_of_range_RM_score")
    return a - previous_a + (t if terminal else 0.0)


class RewardClient:
    def __init__(self, url: str, checkpoint_sha256: str, timeout: float = 600):
        self.url = url.rstrip("/")
        self.sha256 = checkpoint_sha256
        self.timeout = timeout

    def score(self, histories: list[list[str]], instructions: list[str]) -> list[dict]:
        rows = [
            {"frames": [history[i] for i in history_indices(len(history))], "instruction": text}
            for history, text in zip(histories, instructions, strict=True)
        ]
        data = json.dumps({"checkpoint_sha256": self.sha256, "samples": rows}).encode()
        request = urllib.request.Request(
            self.url + "/score", data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if result["checkpoint_sha256"] != self.sha256 or len(result["scores"]) != len(rows):
            raise ValueError("RM_response_contract_mismatch")
        for score in result["scores"]:
            at_reward(0.0, score["A"], score["T"], True)
        return result["scores"]

def terminal_t_reward(t: float, terminal: bool) -> float:
    if not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError('invalid_full_T_score')
    return float(t) if terminal else 0.0
