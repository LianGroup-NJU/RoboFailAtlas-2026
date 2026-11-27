"""Task-specific kinematic proxies for successful-expert action fragments."""
import numpy as np
def merge_windows(windows,n):
 out=[]
 for x in sorted(windows,key=lambda w:w['start']):
  x=dict(x,start=max(0,int(x['start'])),stop=min(n,int(x['stop'])))
  if x['stop']-x['start']<8:raise ValueError('segment_too_short')
  if out and x['start']<=out[-1]['stop']:
   out[-1]['stop']=max(out[-1]['stop'],x['stop']);out[-1]['actions']=sorted(set(out[-1]['actions']+x['actions']))
  else:out.append(x)
 return out
def close_event(grip):
 ids=np.flatnonzero(grip<.3)
 if len(ids)==0:raise ValueError('no_commanded_grasp_close')
 c=int(ids[0]);before=np.flatnonzero(grip[:c]>.8)
 begin=int(before[-1])+1 if len(before)else max(0,c-8)
 return c,max(0,c-12,begin-3)
def rotation_step(rot):
 d=rot[:-1].transpose(0,2,1)@rot[1:]
 return np.r_[0.,np.arccos(np.clip((np.trace(d,axis1=1,axis2=2)-1)/2,-1,1))]
def intervals(mask):
 ids=np.flatnonzero(mask)
 if not len(ids):return []
 breaks=np.flatnonzero(np.diff(ids)>1)+1
 return [(int(a[0]),int(a[-1])+1)for a in np.split(ids,breaks)if len(a)]
def segments(task,action,state,fk):
 n=len(action)
 if n<30 or action.shape!=(n,16)or state.shape!=(n,16):raise ValueError('bad_expert_shape')
 energies=[np.linalg.norm(np.diff(state[:,j:j+7],axis=0),axis=1).sum()for j in [0,8]]
 arm=int(np.argmax(energies));pos,rot=fk(state[:,arm*8:arm*8+7]);grip=action[:,arm*8+7]
 c,gs=close_event(grip);opened=np.flatnonzero((np.arange(n)>c+6)&(grip>.8));op=int(opened[0])if len(opened)else n
 windows=[];events={'active_arm':arm,'close':c,'open':op,'joint_motion':list(map(float,energies))}
 grasp=lambda a,b:{'start':a,'stop':b,'actions':['ATOM.Grasp']}
 if task in ('adjust_bottle','move_can_pot'):
  windows=[grasp(gs,c+6)]
  events['selected_phase']='grasp_closure_with_approach'
 elif task=='lift_pot':
  cs=[];ss=[];zs=[]
  for k in [0,1]:
   cc,start=close_event(action[:,k*8+7]);cs.append(cc);ss.append(start);zs.append(fk(state[:,k*8:k*8+7])[0][:,2])
  c=max(cs);z=np.mean(zs,axis=0);dz=np.r_[0.,np.diff(z)]
  bouts=intervals((np.arange(n)>c+4)&(dz>.001))
  if not bouts:raise ValueError('no_postgrasp_lift')
  a,b=max(bouts,key=lambda ab:z[ab[1]-1]-z[ab[0]])
  if z[b-1]-z[a]<.012:raise ValueError('insufficient_lift_motion')
  windows=[grasp(min(ss),c+6),{'start':a-3,'stop':min(n,b+4),'actions':['ATOM.Lift','ATOM.Hold']}]
  events.update(dual_close=cs,lift_start=a,lift_stop=b)
 elif task=='rotate_qrcode':
  lo=c+12;hi=min(op,n);width=min(24,hi-lo)
  if width<12:raise ValueError('no_rotation_search_window')
  omega=rotation_step(rot);scores=np.convolve(omega[lo:hi],np.ones(width),mode='valid')
  a=lo+int(np.argmax(scores));b=a+width
  if omega[a:b].sum()<.08:raise ValueError('no_resolved_rotation')
  windows=[grasp(gs,c+6),{'start':a,'stop':b,'actions':['ATOM.Rotate']}]
  events.update(rotation_start=a,rotation_stop=b,rotation_radians=float(omega[a:b].sum()))
 elif task=='beat_block_hammer':
  speed=np.r_[0.,np.linalg.norm(np.diff(pos,axis=0),axis=1)];omega=rotation_step(rot)
  moves=np.flatnonzero((np.arange(n)>c+9)&((speed>.001)|(omega>.005)))
  if not len(moves):raise ValueError('no_postgrasp_tool_motion')
  end=int(moves[-1])+1;start=max(c+10,end-24)
  windows=[{'start':start,'stop':min(n,end+2),'actions':['ATOM.Hammer','ATOM.Hold']}]
  events.update(tool_motion_start=start,tool_motion_stop=end)
 elif task in ('place_object_scale','place_empty_cup'):
  if op==n:raise ValueError('no_postgrasp_release')
  windows=[grasp(gs,c+6),{'start':max(c+9,op-24),'stop':min(n,op+6),'actions':['ATOM.Place','ATOM.Release']}]
 elif task=='press_stapler':
  dz=np.r_[0.,np.diff(pos[:,2])]
  bouts=intervals((np.arange(n)>c+8)&(dz<-.001))
  if not bouts:raise ValueError('no_postclose_press_descent')
  a,b=max(bouts,key=lambda ab:pos[ab[0],2]-pos[ab[1]-1,2])
  if pos[a,2]-pos[b-1,2]<.015:raise ValueError('insufficient_press_descent')
  windows=[{'start':a-4,'stop':min(n,b+8),'actions':['ATOM.Press']}]
  events.update(press_start=a,press_stop=b,press_descent=float(pos[a,2]-pos[b-1,2]))
 else:raise ValueError('unplanned_task')
 windows=merge_windows(windows,n)
 if sum(w['stop']-w['start']for w in windows)>.75*n:raise ValueError('target_covers_most_of_episode')
 return windows,events
def matched_random(windows,n,rng):
 lengths=[x['stop']-x['start']for x in windows]
 for _ in range(100):
  occupied=np.zeros(n,dtype=bool);out=[];order=rng.permutation(len(lengths))
  for i in order:
   length=lengths[i];allowed=[s for s in range(n-length+1)if not occupied[s:s+length].any()]
   if not allowed:break
   s=int(rng.choice(allowed));occupied[s:s+length]=True;out.append((i,{'start':s,'stop':s+length,'actions':['random_control']}))
  if len(out)==len(lengths):return [x for _,x in sorted(out)]
 # Feasible nonoverlapping fallback: randomize gap allocation, retaining paired clip lengths.
 spare=n-sum(lengths);gaps=rng.multinomial(spare,np.ones(len(lengths)+1)/(len(lengths)+1));cursor=int(gaps[0]);out=[]
 for i,length in enumerate(lengths):
  out.append({'start':cursor,'stop':cursor+length,'actions':['random_control']});cursor+=length+int(gaps[i+1])
 return out
