"""Small CPU-safe command surface; GPU runtimes are loaded only when requested."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from .config import load,write_new
def read(path):return json.loads(Path(path).read_text())
def rows(path):
    from .dataset import jsonl
    return list(jsonl(path))
def main(argv=None):
    p=argparse.ArgumentParser(prog="failatlas")
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("config");a.add_argument("input");a.add_argument("--output")
    a=sub.add_parser("validate");a.add_argument("samples")
    a=sub.add_parser("build-snapshot");a.add_argument("--manifest",required=True);a.add_argument("--ontology",required=True)
    a.add_argument("--output",required=True);a.add_argument("--video-root");a.add_argument("--terminal-only",action="store_true")
    a.add_argument("--exhaustive-failure-labels",action="store_true")
    a=sub.add_parser("evaluate");a.add_argument("--predictions",required=True);a.add_argument("--samples",required=True)
    a.add_argument("--ontology",required=True);a.add_argument("--split",choices=["validation","test"],required=True)
    a.add_argument("--calibration");a.add_argument("--label-threshold",type=float,default=.8);a.add_argument("--output",required=True)
    a=sub.add_parser("aggregate-policy");a.add_argument("--results",required=True);a.add_argument("--output",required=True)
    a.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    a=sub.add_parser("doctor");a.add_argument("--output")
    args=p.parse_args(argv)
    if args.command=="config":
        result=load(args.input)
        if args.output:write_new(args.output,result);result={"configuration_written":args.output}
    elif args.command=="validate":
        from qwen_rm.contracts import validate_sample
        n=0
        for row in rows(args.samples):validate_sample(row);n+=1
        result={"valid_samples":n}
    elif args.command=="build-snapshot":
        from .dataset import build
        result=build(args.manifest,args.ontology,args.output,args.video_root,args.terminal_only,args.exhaustive_failure_labels)
    elif args.command=="evaluate":
        from .evaluation import evaluate
        result=evaluate(rows(args.samples),rows(args.predictions),read(args.ontology),args.split,
                        read(args.calibration) if args.calibration else None,args.label_threshold)
        write_new(args.output,result)
    elif args.command=="aggregate-policy":
        from .evaluation import aggregate_policy
        value=read(args.results);result=aggregate_policy(value["rows"] if isinstance(value,dict) else value,tuple(args.seeds))
        write_new(args.output,result)
    else:
        import importlib.metadata as md
        result={"python":sys.version.split()[0],"packages":{},"CUDA_check":"not_requested"}
        for name in ["torch","transformers","peft","accelerate","numpy","Pillow","ray","omegaconf","hydra-core"]:
            try:result["packages"][name]=md.version(name)
            except md.PackageNotFoundError:result["packages"][name]=None
        if args.output:write_new(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=="__main__":main()
