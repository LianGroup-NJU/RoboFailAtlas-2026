from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--split',choices=('validation','test'),required=True);p.add_argument('--frame-cache',required=True)
    p.add_argument('--output',required=True);p.add_argument('--calibration');p.add_argument('--device',default='cuda:0')
    p.add_argument('--a-weight',type=float);p.add_argument('--t-weight',type=float)
    args=p.parse_args()
    import torch
    from .data import SampleDataset
    from dual_rm.runtime import build_runtime
    from accelerate.utils import send_to_device
    from .metrics import summarize
    from .checkpoints import model_fingerprint,validate_calibration
    checkpoint=Path(args.checkpoint).resolve();out=Path(args.output).resolve()
    if out.exists():raise ValueError('prediction_output_exists')
    metadata=json.loads((checkpoint/'reward_model.json').read_text(encoding='utf-8'))
    config=metadata['config'];position=json.loads((checkpoint/'training_position.json').read_text())
    if config['method']=='native_sft':
        raise ValueError('native_SFT_checkpoint_requires_python_-m_native_rm.predict')
    snapshot=Path(args.snapshot).resolve();snapshot_hash=hashlib.sha256((snapshot/'snapshot.json').read_bytes()).hexdigest()
    if snapshot_hash!=position['contract']['snapshot_sha256']:raise ValueError('evaluation_snapshot_changed')
    calibration=json.loads(Path(args.calibration).read_text()) if args.calibration else None
    checkpoint_hash=model_fingerprint(checkpoint)
    if args.split=='test':
        validate_calibration(calibration,{'reference_snapshot_sha256':snapshot_hash,'checkpoint_step':position['step'],
                          'input_mode':config['mode'],'method':config['method'],'checkpoint_sha256':checkpoint_hash})
    combine=args.a_weight is not None or args.t_weight is not None
    if combine and (args.a_weight is None or args.t_weight is None or not all(math.isfinite(v) and v>=0 for v in (args.a_weight,args.t_weight)) or args.a_weight+args.t_weight<=0):
        raise ValueError('explicit_valid_A_and_T_reward_weights_required')
    model,processor,collate=build_runtime(config,metadata['ontology'],args.frame_cache,checkpoint/'processor')
    model.load_weights(checkpoint);model.to(args.device);model.eval()
    dataset=SampleDataset(snapshot,args.split,config['mode']);records=[]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream,torch.inference_mode():
        for row in dataset.rows:
            batch=collate([row]);inputs=send_to_device(batch['inputs'],args.device)
            dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'no':torch.float32}[config['mixed_precision']]
            with torch.autocast(device_type=torch.device(args.device).type,dtype=dtype,enabled=config['mixed_precision']!='no'):
                logits=model(**inputs)
            A=float(logits['A'].float().sigmoid()[0]);T=float(logits['T'].float().sigmoid()[0])
            result={'id':row['id'],'case_id':row['case_id'],'frame':row['frame'],'A':A,'T':T,
                    'method':config['method'],'input_mode':config['mode'],'backend':config['backend']}
            if config['method']=='joint':
                for head in ('failure_actions','failure_types','failure_results'):
                    probs=logits[head].float().sigmoid()[0].cpu().tolist()
                    result[head]=[code for code,prob in zip(metadata['ontology'][head],probs) if prob>=.8]
                    result[head+'_probabilities']=dict(zip(metadata['ontology'][head],probs))
                recovery=logits['recovery'].float().softmax(-1)[0];idx=int(recovery.argmax())
                result['recovery']=metadata['ontology']['recovery'][idx] if float(recovery[idx])>=.5 else None
            if combine:result['reward']=(args.a_weight*A+args.t_weight*T)/(args.a_weight+args.t_weight)
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            records.append({'A_prediction':A,'T_prediction':T,'A_target':row['targets']['A']['value'],
                'T_target':row['targets']['T']['value'],'A_mask':row['targets']['A']['mask'],'T_mask':row['targets']['T']['mask']})
    metrics=summarize(records,args.split,calibration)
    metrics.update(checkpoint_step=position['step'],reference_snapshot_sha256=snapshot_hash,
                   ontology_sha256=metadata['ontology']['sha256'],input_mode=config['mode'],method=config['method'],checkpoint_sha256=checkpoint_hash)
    out.with_suffix(out.suffix+'.metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics))

if __name__=='__main__':main()
