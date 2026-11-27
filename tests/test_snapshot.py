from pathlib import Path
import json,tempfile,unittest
from failatlas.dataset import build
from qwen_rm.data import SampleDataset
class SnapshotTests(unittest.TestCase):
    def test_new_snapshot_preserves_masks_and_causality(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/"video.mp4").write_bytes(b"fixture is not decoded in this contract test")
            ontology={"failure_actions":["x"],"failure_types":["t"],"failure_results":["o"],"recovery":["a","b","c"],"sha256":"fixture"}
            (p/"ontology.json").write_text(json.dumps(ontology))
            manifest=p/"trajectories.jsonl"
            entries=[{"case_id":str(i),"task":"demo","split":split,"instruction":"Move an object","video":"video.mp4",
                "frame_count":4,"environment_success":i%2,"progress":[{"frame":0,"value":0.,"known":True}]}
                for i,split in enumerate(["train","validation","test"])]
            manifest.write_text("".join(json.dumps(x)+"\n" for x in entries))
            report=build(manifest,p/"ontology.json",p/"snapshot")
            self.assertEqual(report["trajectory_counts"],{"train":1,"validation":1,"test":1})
            data=SampleDataset(p/"snapshot","train","history")
            self.assertEqual(len(data),4)
            self.assertTrue(all(max(row["frames"])<=row["frame"] for row in data.rows))
            self.assertEqual(sum(row["targets"]["T"]["mask"] for row in data.rows),1)
            self.assertEqual(sum(row["targets"]["A"]["mask"] for row in data.rows),1)
            self.assertFalse(any(data.rows[-1]["targets"]["failure_actions"]["mask"]))
            with (p/"snapshot/history_train.jsonl").open("a") as f:f.write(" ")
            with self.assertRaisesRegex(ValueError,"frozen_sample"):SampleDataset(p/"snapshot","train","history")
    def test_trajectory_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/"v").write_bytes(b"x")
            (p/"o.json").write_text(json.dumps({"failure_actions":[],"failure_types":[],"failure_results":[],"recovery":["a"],"sha256":"fixture"}))
            row={"case_id":"same","task":"demo","split":"train","instruction":"Task","video":"v","frame_count":1}
            (p/"m.jsonl").write_text(json.dumps(row)+"\n"+json.dumps({**row,"split":"test"})+"\n")
            with self.assertRaisesRegex(ValueError,"more_than_once"):build(p/"m.jsonl",p/"o.json",p/"out")
if __name__=="__main__":unittest.main()
