"""CPU-only replay of saved default-Joint predictions and known-label masks."""
from pathlib import Path
import argparse,gzip,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from failatlas.evaluation import evaluate

def rows(path):
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",type=Path,default=ROOT/"examples/metric_replay")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    ref=json.loads((args.fixture/"reference_metrics.json").read_text())
    calibration=json.loads((args.fixture/"calibration.json").read_text())
    ontology=json.loads((args.fixture/"ontology.json").read_text())
    result=evaluate(rows(args.fixture/"samples.jsonl.gz"),rows(args.fixture/"predictions.jsonl.gz"),ontology,"test",calibration)
    for key in ["samples","A_valid_points","T_terminal_cases"]:
        if result[key]!=ref[key]:raise ValueError("fixture_population_mismatch:"+key)
    measured={key:result[key] for key in ["A_mae","T_auc","balanced_accuracy","success_f1"]}
    for group in ["failure_actions","failure_types","failure_results","recovery"]:
        for metric in ["micro_precision","micro_recall","micro_f1","macro_f1"]:
            measured[group+"."+metric]=result["structured"][group][metric]
    delta={key:abs(measured[key]-expected) for key,expected in ref["metrics"].items()}
    report={"matched":all(value<ref["tolerance"] for value in delta.values()),
            "metrics_compared":len(delta),"maximum_absolute_difference":max(delta.values()),
            "samples":result["samples"],"A_valid_points":result["A_valid_points"],
            "T_terminal_cases":result["T_terminal_cases"],"measured":measured,
            "model_inference_performed":False,"network_requests":0}
    if args.output:
        with args.output.open("x",encoding="utf-8") as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2,allow_nan=False))
    return 0 if report["matched"] else 1
if __name__=="__main__":raise SystemExit(main())
