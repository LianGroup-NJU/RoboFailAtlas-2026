"""Build portable training snapshots from public per-trajectory manifests."""
from pathlib import Path
import copy,hashlib,json
from qwen_rm.contracts import SCHEMA,A_VERSION,SPLITS,causal_frames,targets,validate_sample,auxiliary_targets
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):h.update(block)
    return h.hexdigest()
def jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)
def build(manifest,ontology_path,out,video_root=None,terminal_only=False,exhaustive=False):
    manifest=Path(manifest);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    ontology=json.loads(Path(ontology_path).read_text());root=Path(video_root) if video_root else manifest.parent
    ids=set();counts={s:0 for s in SPLITS};cases={s:0 for s in SPLITS};streams={}
    try:
        for mode in ("single","history"):
            for split in SPLITS:
                name=f"{mode}_{split}.jsonl";streams[name]=(out/name).open("x")
        for entry in jsonl(manifest):
            cid=entry["case_id"];split=entry["split"]
            if cid in ids:raise ValueError("trajectory_id_occurs_more_than_once")
            if split not in SPLITS:raise ValueError("unknown_split")
            ids.add(cid);cases[split]+=1
            video=Path(entry["video"]);video=(video if video.is_absolute() else root/video).resolve()
            stat=video.stat();n=entry["frame_count"]
            if type(n) is not int or n<=0 or not entry["instruction"].strip():raise ValueError("invalid_trajectory")
            outcome=entry.get("environment_success")
            if outcome is not None and outcome not in (0,1):raise ValueError("T_must_be_environment_binary_outcome")
            progress={x["frame"]:x for x in entry.get("progress",[])}
            frames=[n-1] if terminal_only else sorted(set(targets(n))|set(progress))
            if any(type(f) is not int or not 0<=f<n for f in frames):raise ValueError("invalid_scoring_frame")
            media_key=hashlib.sha256(f"{video}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
            for frame in frames:
                a=progress.get(frame,{});value=a.get("value");known=a.get("known",value is not None)
                empty={k:{"values":[0.]*len(ontology[k]),"mask":[False]*len(ontology[k])}
                       for k in ("failure_actions","failure_types","failure_results")}
                empty["recovery"]={"value":0,"mask":False}
                if entry.get("annotation_result") is not None:
                    empty=auxiliary_targets(entry["annotation_result"],frame,n-1,"success" if outcome==1 else "failure" if outcome==0 else "unknown",ontology,exhaustive)
                supervision={"A":{"value":value if known else 0.,"mask":bool(known)},
                    "T":{"value":float(outcome) if outcome is not None else 0.,"mask":frame==n-1 and outcome is not None},**empty}
                for mode in ("single","history"):
                    row={"schema":SCHEMA,"id":f"{cid}:{frame}:{mode}","case_id":cid,"task":entry["task"],
                        "split":split,"instruction":entry["instruction"],"video":str(video),"video_bytes":stat.st_size,
                        "video_mtime_ns":stat.st_mtime_ns,"media_key":media_key,"frame_count":n,"frame":frame,
                        "frames":causal_frames(frames,frame,mode,8),"mode":mode,"history_max_frames":8,
                        "targets":copy.deepcopy(supervision),"provenance":{"A_version":A_VERSION,"source":"provided_clean_trajectory_manifest"}}
                    validate_sample(row);streams[f"{mode}_{split}.jsonl"].write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                counts[split]+=1
    finally:
        for stream in streams.values():stream.close()
    (out/"ontology.json").write_text(json.dumps(ontology,ensure_ascii=False,indent=2)+"\n")
    report={"schema":SCHEMA,"history_max_frames":8,"trajectory_counts":cases,"counts":counts,
        "missing_labels_are_unknown":True,"test_is_never_training_data":True,"input_fields":["instruction","observed_images"],
        "source_manifest_sha256":sha(manifest),"ontology_sha256":ontology["sha256"],
        "files_sha256":{p.name:sha(p) for p in sorted(out.glob("*.jsonl"))}}
    (out/"snapshot.json").write_text(json.dumps(report,indent=2)+"\n")
    return report
