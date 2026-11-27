from __future__ import annotations

import math
import re


REASONS = frozenset(('correct', 'miss', 'collision', 'fall', 'smooth', 'failure'))
SCORE_ONLY_SYSTEM = '''You are a robot manipulation evaluator. The images are in chronological order and the last image is the current observation. Given the robot task, estimate physical execution progress at the last observed frame. Output only one number between 0.000 and 1.000 with exactly three decimal places. Do not output reasoning, tags, JSON, or any other text.'''
RUBRIC = '''Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.
Task: {task}'''


def requires_reason_word(config):
    return config.get('model_key') == 'densereward' and config.get('native_output_format', 'score_only') == 'thinking_score'


def native_answer(row, config):
    kind = config.get('native_target', 'A_only')
    if kind not in ('A_only', 'terminal_T'):
        raise ValueError('native_target_requires_A_only_or_terminal_T')
    target = row['targets']['A' if kind == 'A_only' else 'T']
    if not target['mask']:
        return None
    if kind == 'terminal_T' and row['frame'] != row['frame_count'] - 1:
        raise ValueError('native_T_cannot_supervise_an_intermediate_frame')
    a = float(target['value'])
    if not math.isfinite(a) or not 0 <= a <= 1:
        raise ValueError('invalid_native_progress_target')
    if config['backend'] == 'roboreward':
        # Explicit engineering mapping; not an independently annotated native rubric.
        score = 1 + min(4, math.floor(4 * a + .5))
        return 'ANSWER: ' + str(score)
    scalar = f'{a:.3f}'
    if not requires_reason_word(config):
        return scalar
    reason = row.get('native_reason_word')
    if reason not in REASONS:
        return None
    return '<think>\n' + reason + '\n</think>\n\n' + scalar


def parse_native(raw, config):
    if config['backend'] == 'roboreward':
        match = re.fullmatch(r'ANSWER:\s*([1-5])', raw.strip(), re.I)
        if not match:
            raise ValueError('invalid_RoboReward_native_answer')
        return {'score': (int(match[1]) - 1) / 4, 'discrete_score': int(match[1])}
    if not requires_reason_word(config):
        match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+))', raw.strip())
        reason = None
    else:
        match = re.fullmatch(r'<think>\s*([^<>]{1,500}?)\s*</think>\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))', raw.strip(), re.S)
        reason = match[1].strip() if match else None
    if not match:
        raise ValueError('invalid_DenseReward_native_answer')
    score = float(match[2] if requires_reason_word(config) else match[1])
    if not math.isfinite(score):
        raise ValueError('nonfinite_native_score')
    return {'score': max(0., min(1., score)), 'raw_score': score,
            'reason_word': reason if reason in REASONS else None,
            'reason_valid': reason in REASONS if reason is not None else None}
