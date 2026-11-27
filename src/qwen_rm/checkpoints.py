import hashlib
from pathlib import Path


def model_fingerprint(folder):
    folder = Path(folder); state = hashlib.sha256()
    for relative in ('adapter/adapter_model.safetensors', 'heads.safetensors', 'reward_model.json'):
        state.update(relative.encode() + b'\0')
        with (folder / relative).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                state.update(block)
    return state.hexdigest()


def validate_calibration(calibration, expected):
    if not calibration or calibration.get('split') != 'validation':
        raise ValueError('frozen_validation_calibration_required')
    for key, value in expected.items():
        if calibration.get(key) != value:
            raise ValueError('calibration_checkpoint_or_dataset_mismatch:' + key)
