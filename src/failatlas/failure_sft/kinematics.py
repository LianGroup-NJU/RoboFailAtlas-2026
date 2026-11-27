"""URDF FK for offline successful-expert phase localization."""
import xml.etree.ElementTree as ET
import numpy as np
def rpy_matrix(v):
 x,y,z=v;cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
 return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1.]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1.,0,0],[0,cx,-sx],[0,sx,cx]])
class PandaFK:
 def __init__(self,path,end='panda_hand'):
  root=ET.parse(path).getroot();joints={j.find('child').get('link'):j for j in root.findall('joint')}
  chain=[];link=end
  while link in joints:
   j=joints[link];o=j.find('origin');axis=j.find('axis')
   xyz=np.fromstring(o.get('xyz','0 0 0')if o is not None else '0 0 0',sep=' ')
   rpy=np.fromstring(o.get('rpy','0 0 0')if o is not None else '0 0 0',sep=' ')
   a=np.fromstring(axis.get('xyz','0 0 1')if axis is not None else '0 0 1',sep=' ')
   mat=np.eye(4);mat[:3,:3]=rpy_matrix(rpy);mat[:3,3]=xyz
   chain.append((j.get('name'),j.get('type'),mat,a/np.linalg.norm(a) if np.linalg.norm(a)>0 else np.array([0.,0.,1.])));link=j.find('parent').get('link')
  self.chain=list(reversed(chain));self.names=[name for name,typ,_,axis in self.chain if typ in ('revolute','continuous')]
  assert self.names==['panda_joint'+str(i)for i in range(1,8)],self.names
 def __call__(self,q):
  q=np.asarray(q);n=len(q);t=np.broadcast_to(np.eye(4),(n,4,4)).copy();col=0
  for name,typ,origin,axis in self.chain:
   t=t@origin
   if typ in ('revolute','continuous'):
    x,y,z=axis;k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    theta=q[:,col];col+=1;rot=np.eye(3)[None]+np.sin(theta)[:,None,None]*k+(1-np.cos(theta))[:,None,None]*(k@k)
    trans=np.broadcast_to(np.eye(4),(n,4,4)).copy();trans[:,:3,:3]=rot;t=t@trans
  return t[:,:3,3],t[:,:3,:3]
