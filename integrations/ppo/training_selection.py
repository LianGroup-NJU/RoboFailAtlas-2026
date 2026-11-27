from pathlib import Path
import json,struct,math,hashlib

def varint(b,i):
    n=0;s=0
    while True:
        c=b[i];i+=1;n|=(c&127)<<s
        if c<128:return n,i
        s+=7

def fields(b):
    i=0
    while i<len(b):
        k,i=varint(b,i);wire=k&7;field=k>>3
        if wire==0:v,i=varint(b,i)
        elif wire==1:v=b[i:i+8];i+=8
        elif wire==2:n,i=varint(b,i);v=b[i:i+n];i+=n
        elif wire==5:v=b[i:i+4];i+=4
        else:raise ValueError('unsupported_protobuf_wire:'+str(wire))
        yield field,wire,v

def scalar_events(path):
    tags={}
    with open(path,'rb') as f:
        while True:
            head=f.read(12)
            if len(head)<12:break
            n=struct.unpack('<Q',head[:8])[0];body=f.read(n);crc=f.read(4)
            if len(body)<n or len(crc)<4:break
            data={k:(wire,v) for k,wire,v in fields(body)}
            step=data.get(2,(0,0))[1]
            wall=struct.unpack('<d',data[1][1])[0] if 1 in data else 0
            if 5 not in data:continue
            for k,w,v in fields(data[5][1]):
                if k!=1:continue
                row={a:(bw,bv) for a,bw,bv in fields(v)}
                if 1 in row and 2 in row:
                    tag=row[1][1].decode();value=struct.unpack('<f',row[2][1])[0]
                    tags.setdefault(tag,[]).append({'step':step,'wall_time':wall,'value':value})
    return tags

def score_candidate(step,series,window=5):
    """Use the last five recorded training rounds strictly before saved update."""
    valid=[]
    for i in sorted(set(series.get('env/success_once',{})) & set(series.get('env/num_trajectories',{}))):
        rate=series['env/success_once'][i];count=series['env/num_trajectories'][i]
        if i<step and math.isfinite(rate) and math.isfinite(count) and 0<=rate<=1 and count>0:
            valid.append({'step':i,'success_rate':rate,'completed_episodes':count})
    rows=valid[-window:]
    if len(rows)<window:return {'step':step,'eligible':False,'reason':'fewer_than_five_recorded_training_rounds','rounds':rows}
    total=sum(r['completed_episodes'] for r in rows)
    successes=sum(r['success_rate']*r['completed_episodes'] for r in rows)
    return {'step':step,'eligible':True,'training_success_rate':successes/total,'success_count_from_scalar_metrics':successes,
        'completed_episodes':total,'rounds':rows,
        'last_logged_rollout_round':rows[-1]['step'],'immediately_preceding_round_logged':rows[-1]['step']==step-1}

def choose_candidate(rows):
    valid=[r for r in rows if r.get('eligible')]
    if not valid:raise ValueError('no_candidate_has_five_valid_training_rounds')
    return min(valid,key=lambda r:(-round(r['training_success_rate'],9),r['step']))
