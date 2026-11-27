"""Metrics use the original masks and validation-only T calibration."""
import math
from qwen_rm.metrics import summarize

HEADS=("failure_actions","failure_types","failure_results")
def endpoint(row):
    if type(row.get("frame")) is int and type(row.get("frame_count")) is int:
        return row["frame"] == row["frame_count"] - 1
    return bool(row["targets"]["T"]["mask"])
def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    return {"micro_precision":p,"micro_recall":r,"micro_f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "tp":tp,"fp":fp,"fn":fn}
def evaluate(samples,predictions,ontology,split,calibration=None,label_threshold=.8):
    if split not in ("validation","test"):raise ValueError("evaluation_split_required")
    if not 0<=label_threshold<=1:raise ValueError("invalid_label_threshold")
    by_id={}
    for pred in predictions:
        if pred["id"] in by_id:raise ValueError("duplicate_prediction_id")
        by_id[pred["id"]]=pred
    ids=[x["id"] for x in samples]
    if len(ids)!=len(set(ids)) or set(ids)!=set(by_id):raise ValueError("prediction_sample_membership_mismatch")
    if any(row["split"]!=split for row in samples):raise ValueError("split_mismatch")
    records=[]
    for row in samples:
        p=by_id[row["id"]]
        if any(not isinstance(p[k],(int,float)) or not math.isfinite(p[k]) or not 0<=p[k]<=1 for k in ("A","T")):
            raise ValueError("invalid_score")
        records.append({"A_prediction":p["A"],"T_prediction":p["T"],
            "A_target":row["targets"]["A"]["value"],"A_mask":row["targets"]["A"]["mask"],
            "T_target":row["targets"]["T"]["value"],"T_mask":row["targets"]["T"]["mask"]})
    result=summarize(records,split,calibration)
    result["label_threshold"]=label_threshold;result["unknown_policy"]="field/category masks; retain other known labels"
    structured={}
    for head in HEADS:
        if not any(head in p or head+"_probabilities" in p for p in predictions):continue
        codes=ontology[head];counts=[[0,0,0,0] for _ in codes]
        for row in samples:
            if not endpoint(row):continue
            pred=by_id[row["id"]];target=row["targets"][head]
            selected=pred.get(head,[]);probs=pred.get(head+"_probabilities")
            if probs is None and not set(selected)<=set(codes):raise ValueError("unknown_predicted_label")
            if len(target["values"])!=len(codes) or len(target["mask"])!=len(codes):raise ValueError("ontology_dimension_mismatch")
            for i,code in enumerate(codes):
                if not target["mask"][i]:continue
                if probs is not None:
                    score=probs[code]
                    if not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:raise ValueError("invalid_label_probability")
                    predicted=score>=label_threshold
                else:predicted=code in selected
                actual=bool(target["values"][i]);c=counts[i];c[3]+=1
                c[0]+=int(predicted and actual);c[1]+=int(predicted and not actual);c[2]+=int(not predicted and actual)
        tp,fp,fn=[sum(c[i] for c in counts) for i in range(3)]
        supported=[prf(*c[:3])["micro_f1"] for c in counts if c[0]+c[2]]
        structured[head]={**prf(tp,fp,fn),"macro_f1":sum(supported)/len(supported) if supported else None,
            "valid_category_decisions":sum(c[3] for c in counts),"positive_supported_classes":len(supported),"categories_with_known_labels":sum(bool(c[3]) for c in counts)}
    if any("recovery" in p for p in predictions):
        labels=ontology["recovery"];tp=fp=fn=correct=known=covered=0;per=[[0,0,0] for _ in labels]
        for row in samples:
            if not endpoint(row):continue
            target=row["targets"]["recovery"]
            if not target["mask"]:continue
            actual=int(target["value"]);pred=by_id[row["id"]].get("recovery")
            if not 0<=actual<len(labels) or pred is not None and pred not in labels:raise ValueError("invalid_recovery_label")
            known+=1;covered+=int(pred is not None)
            if pred==labels[actual]:tp+=1;correct+=1;per[actual][0]+=1
            else:
                fn+=1;per[actual][2]+=1
                if pred is not None:fp+=1;per[labels.index(pred)][1]+=1
        structured["recovery"]={**prf(tp,fp,fn),"accuracy":correct/known if known else None,
            "macro_f1":sum(prf(*x)["micro_f1"] for x in per if x[0]+x[2])/sum(bool(x[0]+x[2]) for x in per) if known else None,
            "known_samples":known,"prediction_coverage":covered/known if known else None,
            "decision_rule":"use supplied class; null is an abstention and counts as a false negative"}
    result["structured_scope"]="trajectory_endpoints"
    result["structured_macro_averaging"]="classes with known positive support; all known decisions count in micro scores"
    result["structured"]=structured
    return result

def aggregate_policy(rows,seeds=(0,1,2)):
    cells={};tasks=sorted({x["task"] for x in rows})
    for row in rows:
        key=(row["task"],int(row["seed"]))
        if key in cells:raise ValueError("duplicate_task_seed")
        if key[1] not in seeds:raise ValueError("unexpected_seed")
        n=row["episodes"];s=row["successes"]
        if type(n) is not int or type(s) is not int or n<=0 or not 0<=s<=n:raise ValueError("invalid_episode_counts")
        cells[key]=100*s/n
    def stats(values):
        mean=math.fsum(values)/len(values)
        return {"mean_percent":mean,"sample_std_percent":math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1)) if len(values)>1 else None}
    task_rows=[]
    for task in tasks:
        values={str(seed):cells.get((task,seed)) for seed in seeds}
        complete=all(v is not None for v in values.values())
        task_rows.append({"task":task,"by_seed":values,"complete":complete,
            **(stats(list(values.values())) if complete else {"mean_percent":None,"sample_std_percent":None})})
    means={str(seed):math.fsum(cells[(task,seed)] for task in tasks)/len(tasks)
           for seed in seeds if tasks and all((task,seed) in cells for task in tasks)}
    return {"seeds":list(seeds),"tasks":task_rows,"five_task_mean_by_seed":means,
        "overall":stats(list(means.values())) if len(means)==len(seeds) and tasks else None,
        "std_protocol":"sample standard deviation, ddof=1; average tasks within each seed first"}
