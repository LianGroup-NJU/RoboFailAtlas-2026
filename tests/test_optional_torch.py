import importlib.util,unittest
HAS_TORCH=importlib.util.find_spec("torch") is not None
@unittest.skipUnless(HAS_TORCH,"optional Torch CPU checks")
class TorchTests(unittest.TestCase):
    def test_outside_clip_target_actions_are_preserved(self):
        import numpy as np
        from failatlas.failure_sft.clip_training import ClipDataset
        actions=np.arange(50*32,dtype=np.float32).reshape(50,32)
        data=ClipDataset([{"actions":actions}],[0],[7])
        item=data[0]
        np.testing.assert_array_equal(item["actions"],actions)
        self.assertEqual(int(item["clip_loss_mask"].sum()),7)
        self.assertIsNot(item["actions"],actions)
    def test_outside_clip_loss_and_gradient_are_zero(self):
        import torch
        from failatlas.failure_sft.clip_training import masked_flow_loss
        loss=torch.tensor([[[2.,2.],[100.,100.]]],requires_grad=True)
        value=masked_flow_loss(loss,[[True,False]])
        self.assertEqual(float(value.detach()),2.)
        value.backward();self.assertEqual(float(loss.grad[0,1].sum()),0.)
    def test_unknown_label_logit_has_no_loss_gradient(self):
        import torch
        from qwen_rm.model import loss_fn
        z={"A":torch.zeros(1,requires_grad=True),"T":torch.zeros(1,requires_grad=True),
           **{h:torch.zeros(1,2,requires_grad=True) for h in ["failure_actions","failure_types","failure_results"]},
           "recovery":torch.zeros(1,3,requires_grad=True)}
        t={"A_value":torch.zeros(1),"A_mask":torch.tensor([False]),"T_value":torch.ones(1),"T_mask":torch.tensor([True]),
           "recovery_value":torch.zeros(1,dtype=torch.long),"recovery_mask":torch.tensor([False])}
        for h in ["failure_actions","failure_types","failure_results"]:
            t[h+"_value"]=torch.tensor([[1.,0.]]);t[h+"_mask"]=torch.tensor([[True,False]])
        loss,_=loss_fn(z,t,{"method":"joint","loss_weights":{k:1. for k in z}})
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertEqual(float(z["A"].grad.abs().sum()),0.)
        for h in ["failure_actions","failure_types","failure_results"]:self.assertEqual(float(z[h].grad[0,1]),0.)
if __name__=="__main__":unittest.main()
