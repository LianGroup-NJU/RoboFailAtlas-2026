"""Prepare an independent runtime copy and apply hash-checked overlays."""
from pathlib import Path
import argparse,hashlib,json,shutil
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--backend",choices=["runtime","runtime_legacy"],default="runtime")
    a=p.parse_args();source=Path(a.source).resolve();out=Path(a.output).resolve()
    repo=Path(__file__).resolve().parents[1];spec=json.loads((repo/"integrations/rlinf_overlays/manifest.json").read_text())
    if out.exists() or source==out or source in out.parents:raise ValueError("new_output_outside_source_required")
    if not (source/"rlinf/__init__.py").is_file():raise ValueError("RLinf_source_missing")
    checks=spec["backends"][a.backend]
    for name,row in checks.items():
        path=source/name
        if row["base_sha256"] is not None and (not path.exists() or digest(path)!=row["base_sha256"]):raise ValueError("RLinf_version_mismatch:"+name)
        if digest(repo/"integrations/rlinf_overlays"/a.backend/name)!=row["patched_sha256"]:raise ValueError("overlay_changed:"+name)
    out.mkdir(parents=True);shutil.copytree(source/"rlinf",out/"rlinf",symlinks=True,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for name in checks:shutil.copyfile(repo/"integrations/rlinf_overlays"/a.backend/name,out/name)
    (out/"runtime_provenance.json").write_text(json.dumps({"source":str(source),"backend":a.backend,"checked_files":checks},indent=2)+"\n")
    print(json.dumps({"runtime":str(out),"backend":a.backend,"patched_files":len(checks)}))
if __name__=="__main__":main()
