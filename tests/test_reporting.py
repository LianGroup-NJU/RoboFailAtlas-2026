import copy,unittest
from failatlas.evaluation import evaluate,aggregate_policy
from failatlas.config import expand
from qwen_rm.contracts import causal_frames,targets
O={"failure_actions":["x","y"],"failure_types":["t"],"failure_results":["o"],"recovery":["a","b","c"]}
def sample(i,outcome,recovery):
    return {"id":str(i),"split":"validation","targets":{
      "A":{"value":.5,"mask":True},"T":{"value":outcome,"mask":True},
      "failure_actions":{"values":[1,0],"mask":[True,False]},
      "failure_types":{"values":[0],"mask":[False]},
      "failure_results":{"values":[0],"mask":[False]},
      "recovery":{"value":recovery,"mask":True}}}
def pred(i,t,recovery):
    return {"id":str(i),"A":.5,"T":t,"failure_actions_probabilities":{"x":.9,"y":.99},"recovery":recovery}
class ReportingTests(unittest.TestCase):
    def test_unknown_category_does_not_create_false_positive(self):
        rows=[sample(0,0,0),sample(1,1,1)]
        result=evaluate(rows,[pred(0,.1,"a"),pred(1,.9,"b")],O,"validation")
        self.assertEqual(result["structured"]["failure_actions"]["fp"],0)
        self.assertEqual(result["structured"]["failure_actions"]["valid_category_decisions"],2)
        self.assertEqual(result["T_auc"],1.)
        self.assertEqual(result["balanced_accuracy"],1.)
    def test_recovery_abstention_changes_precision_and_recall(self):
        result=evaluate([sample(0,0,0),sample(1,1,1)],[pred(0,.1,"a"),pred(1,.9,None)],O,"validation")["structured"]["recovery"]
        self.assertEqual(result["micro_precision"],1.)
        self.assertEqual(result["micro_recall"],.5)
        self.assertEqual(result["prediction_coverage"],.5)
    def test_complete_single_label_micro_equals_accuracy(self):
        result=evaluate([sample(0,0,0),sample(1,1,1)],[pred(0,.1,"a"),pred(1,.9,"c")],O,"validation")["structured"]["recovery"]
        for k in ("micro_precision","micro_recall","micro_f1"):self.assertEqual(result[k],result["accuracy"])
    def test_test_threshold_requires_validation(self):
        rows=[sample(0,0,0),sample(1,1,1)]
        for row in rows:row["split"]="test"
        with self.assertRaisesRegex(ValueError,"calibration"):evaluate(rows,[pred(0,.1,"a"),pred(1,.9,"b")],O,"test")
        result=evaluate(rows,[pred(0,.1,"a"),pred(1,.9,"b")],O,"test",{"T_threshold":.95})
        self.assertEqual(result["success_f1"],0.)
    def test_missing_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError,"membership"):evaluate([sample(0,0,0)],[],O,"validation")
    def test_duplicate_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError,"duplicate"):evaluate([sample(0,0,0)],[pred(0,.1,"a")]*2,O,"validation")
    def test_overall_std_preserves_cross_task_correlation(self):
        rows=[{"task":task,"seed":seed,"successes":n,"episodes":100}
              for task,vals in [("one",[10,90]),("two",[90,10])] for seed,n in enumerate(vals)]
        result=aggregate_policy(rows,(0,1))
        self.assertEqual(result["overall"]["sample_std_percent"],0.)
        self.assertGreater(result["tasks"][0]["sample_std_percent"],50.)
    def test_missing_seed_is_not_imputed(self):
        result=aggregate_policy([{"task":"one","seed":0,"successes":10,"episodes":100}])
        self.assertIsNone(result["overall"]);self.assertIsNone(result["tasks"][0]["mean_percent"])
    def test_config_is_explicit(self):
        self.assertEqual(expand({"x":"${MODEL}"},{"MODEL":"/tmp/model"}),{"x":"/tmp/model"})
        with self.assertRaisesRegex(ValueError,"unset"):expand("${MODEL}",{})
    def test_prefix_observations_never_include_future(self):
        self.assertEqual(causal_frames([0,2,4],2,"history",8),[0,2])
        self.assertEqual(len(targets(10000)),16)
class PaperProtocolTests(unittest.TestCase):
    def test_probability_between_half_and_paper_threshold_is_negative(self):
        prediction=pred(0,.1,"a")
        prediction["failure_actions_probabilities"]["x"]=.7
        result=evaluate([sample(0,0,0)],[prediction],O,"validation")
        self.assertEqual(result["label_threshold"],.8)
        self.assertEqual(result["structured"]["failure_actions"]["fn"],1)

    def test_macro_uses_positive_support_but_micro_keeps_negative_class_fp(self):
        row=sample(0,0,0)
        row["targets"]["failure_actions"]["mask"]=[True,True]
        prediction=pred(0,.1,"a")
        result=evaluate([row],[prediction],O,"validation")["structured"]["failure_actions"]
        self.assertEqual(result["macro_f1"],1.)
        self.assertEqual(result["micro_f1"],2/3)
        self.assertEqual(result["positive_supported_classes"],1)

    def test_structured_heads_are_evaluated_at_endpoints_only(self):
        early=sample(0,0,0);early.update(frame=0,frame_count=10)
        early["targets"]["T"]["mask"]=False
        last=sample(1,1,1);last.update(frame=9,frame_count=10)
        first_pred=pred(0,.1,"c")
        first_pred["failure_actions_probabilities"]["x"]=.1
        result=evaluate([early,last],[first_pred,pred(1,.9,"b")],O,"validation")
        self.assertEqual(result["structured"]["failure_actions"]["micro_f1"],1.)
        self.assertEqual(result["structured"]["recovery"]["known_samples"],1)

if __name__=="__main__":unittest.main()
