"""Keep complete real flow targets and exclude only out-of-segment loss."""
import ast,copy,inspect,textwrap,types
import numpy as np

class ClipDataset:
    def __init__(self,dataset,indices,valid_steps,horizon=50):
        self.dataset=dataset;self.indices=indices;self.valid_steps=valid_steps;self.horizon=horizon
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        sample=dict(self.dataset[int(self.indices[index])])
        actions=np.asarray(sample['actions'],dtype=np.float32).copy()
        if actions.shape!=(self.horizon,32) or not np.isfinite(actions).all():raise ValueError('invalid_native_action_chunk')
        valid=int(self.valid_steps[index])
        if not 1<=valid<=self.horizon:raise ValueError('invalid_clip_mask')
        mask=np.arange(self.horizon)<valid
        sample['actions']=actions;sample['clip_loss_mask']=mask
        return sample

def masked_flow_loss(loss,mask):
    import torch
    mask=torch.as_tensor(mask,device=loss.device,dtype=torch.bool)
    if loss.ndim!=3 or mask.shape!=loss.shape[:2] or not mask.any(dim=1).all():
        raise ValueError('invalid_per_sample_clip_loss_mask')
    safe=torch.where(mask[:,:,None],loss,torch.zeros_like(loss))
    return (safe.sum(dim=(1,2))/(mask.sum(dim=1)*loss.shape[2])).mean()

def install_masked_sft(model):
    original_class=type(model);original=original_class.sft_forward
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    counts={'super':0,'reduction':0}
    class Rewrite(ast.NodeTransformer):
        def visit_Call(self,node):
            node=self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='super' and not node.args:
                node.args=[ast.Name(id='_clip_original_class',ctx=ast.Load()),ast.Name(id='self',ctx=ast.Load())]
                counts['super']+=1
            return node
        def visit_Return(self,node):
            if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='loss'
                and node.value.func.attr=='mean'):
                counts['reduction']+=1
                return ast.parse("return _clip_masked_flow_loss(loss, data['clip_loss_mask'])").body[0]
            return self.generic_visit(node)
    tree=Rewrite().visit(tree);ast.fix_missing_locations(tree)
    if counts!={'super':1,'reduction':1}:raise ValueError('native_SFT_structure_changed')
    namespace=dict(original.__globals__,_clip_original_class=original_class,_clip_masked_flow_loss=masked_flow_loss)
    exec(compile(tree,'<isolated_clip_masked_native_SFT>','exec'),namespace)
    model.sft_forward=types.MethodType(namespace['sft_forward'],model)

def batches(loader):
    import jax,torch
    from openpi.models.model import Observation
    while True:
        for row in loader:
            batch=jax.tree.map(torch.as_tensor,row)
            yield {'observation':Observation.from_dict(batch),'actions':batch['actions'],
                   'clip_loss_mask':batch['clip_loss_mask']}

