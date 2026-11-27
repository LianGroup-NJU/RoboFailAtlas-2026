"""Frozen paths and experiment contract for the place_fan policy pilot."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path(os.environ.get('FAILATLAS_SFT_WORK', './work/failure_sft')).resolve()
SOURCE = Path(os.environ.get('FAILATLAS_RLINF_ROOT', '__SET_FAILATLAS_RLINF_ROOT__'))
DATA = Path(os.environ.get('FAILATLAS_EXPERT_DATA', '__SET_FAILATLAS_EXPERT_DATA__'))
BASELINE = Path(os.environ.get('FAILATLAS_ROBOTWIN_ROOT', '__SET_FAILATLAS_ROBOTWIN_ROOT__'))
CHECKPOINT = Path(os.environ.get('FAILATLAS_SFT_CHECKPOINT', '__SET_FAILATLAS_SFT_CHECKPOINT__'))
STATS = Path(os.environ.get('FAILATLAS_NORM_STATS', '__SET_FAILATLAS_NORM_STATS__'))
PYTHON = os.environ.get('FAILATLAS_POLICY_PYTHON', __import__('sys').executable)
SIM_PYTHON = os.environ.get('FAILATLAS_SIM_PYTHON', __import__('sys').executable)
PHASES = ('grasp_lift', 'transport_align', 'release_place')

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def environment(work):
    work = Path(work)
    cache = work/'cache'
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE','TORCHELASTIC_RUN_ID'):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', JAX_PLATFORMS='cpu',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        USE_TF='0', WANDB_MODE='disabled', TZ='Asia/Shanghai',
        PYTHONPATH=':'.join([str(ROOT),str(SOURCE),str(BASELINE),str(BASELINE/'XPolicyLab'),os.environ.get('PYTHONPATH','')]),
        HF_HOME=os.environ.get('HF_HOME',str(cache/'huggingface')),
        OPENPI_DATA_HOME=os.environ.get('OPENPI_DATA_HOME',str(cache/'openpi')),
        PYTORCH_KERNEL_CACHE_PATH=str(cache/'kernels'), CUDA_CACHE_PATH=str(cache/'cuda'))
    env['PATH']=str(Path(PYTHON).parent)+':'+env.get('PATH','')
    return env
