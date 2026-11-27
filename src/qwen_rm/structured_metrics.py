"""Masked multilabel validation counts; no missing label is treated as negative."""
class StructuredCounts:
 def __init__(self,ontology):
  self.ontology=ontology
  self.counts={h:[[0,0,0,0,0] for _ in ontology[h]] for h in ['failure_actions','failure_types','failure_results']}
  self.recovery=[[0]*3 for _ in range(3)]
 def add(self,values,i):
  for head,rows in self.counts.items():
   for j,c in enumerate(rows):
    if not values[head+'_mask'][i][j]:continue
    y=values[head+'_value'][i][j]>.5;p=values[head+'_pred'][i][j]>=.5
    c[0]+=1;c[1]+=int(y);c[2]+=int(p and y);c[3]+=int(p and not y);c[4]+=int(not p and y)
  if values['recovery_mask'][i]:self.recovery[int(values['recovery_value'][i])][int(values['recovery_pred'][i])]+=1
 def summary(self):
  result={}
  for head,rows in self.counts.items():
   report=[];TP=FP=FN=0;fs=[]
   for label,(known,pos,tp,fp,fn) in zip(self.ontology[head],rows):
    f=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    if pos:fs.append(f)
    TP+=tp;FP+=fp;FN+=fn
    report.append({'label':label,'known_scoring_points':known,'positive_scoring_points':pos,'tp':tp,'fp':fp,'fn':fn,
      'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/pos if pos else None,'f1':f if known else None})
   result[head]={'threshold':.5,'aggregation':'known_mask_scoring_points','micro_f1':2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else None,
    'macro_f1_positive_supported_classes':sum(fs)/len(fs) if fs else None,'positive_supported_classes':len(fs),'classes':report}
  result['recovery']={'known_scoring_points':sum(map(sum,self.recovery)),'confusion_true_by_pred':self.recovery}
  return result
