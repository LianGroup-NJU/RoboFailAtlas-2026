"""Use an explicitly prepared private RLinf runtime."""
from pathlib import Path
import os,runpy,sys
sys.path.insert(0,str(Path(os.environ["FAILATLAS_RLINF_RUNTIME"]).resolve()))
sys.path.insert(1,str(Path(__file__).resolve().parent))
from joint_rl.extension import register
from joint_rl.checkpoint_schedule_interval import install
register();install()
runpy.run_path(str(Path(os.environ["FAILATLAS_RLINF_ROOT"])/"examples/embodiment/train_embodied_agent.py"),run_name="__main__")
