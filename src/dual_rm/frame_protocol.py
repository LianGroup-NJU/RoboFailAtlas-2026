from __future__ import annotations

from qwen_rm.contracts import causal_frames
from qwen_rm.data import Collator


def effective_row(row, budget, sampling='uniform_prefix'):
    """Sample from the original observed prefix, never from a coarser frame list."""
    if type(budget) is not int or not 1 <= budget <= 12:
        raise ValueError('input_frame_budget_must_be_between_one_and_twelve')
    value = dict(row)
    if sampling == 'last3_of_uniform8':
        if budget != 3 or row['mode'] != 'history':
            raise ValueError('last3_sampling_requires_history_budget3')
        value['frames'] = causal_frames(range(row['frame'] + 1), row['frame'], 'history', 8)[-3:]
        return value  # Reuse the already frozen history8 frame cache.
    if sampling != 'uniform_prefix':
        raise ValueError('unknown_frame_sampling_protocol')
    value['history_max_frames'] = budget
    value['frames'] = causal_frames(range(row['frame'] + 1), row['frame'], row['mode'], budget)
    return value


class FrameBudgetCollator(Collator):
    def __init__(self, processor, frame_cache, max_tokens, budget, sampling='uniform_prefix'):
        super().__init__(processor, frame_cache, max_tokens)
        self.budget = budget
        self.sampling = sampling

    def __call__(self, rows):
        actual = [effective_row(row, self.budget, self.sampling) for row in rows]
        batch = super().__call__(actual)
        batch['input_frame_counts'] = [len(row['frames']) for row in actual]
        return batch
