from pathlib import Path
import hashlib,json,unittest
class OntologyTests(unittest.TestCase):
    def load(self):
        return json.loads((Path(__file__).resolve().parents[1]/"taxonomy/ontology.json").read_text())
    def test_dimensions_match_model_heads(self):
        data=self.load()
        self.assertEqual([len(data[k]) for k in ["failure_actions","failure_types","failure_results","recovery"]],[70,36,15,3])
    def test_frozen_ontology_identity_is_preserved(self):
        data=self.load();recorded=data.pop("sha256")
        encoded=json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(),recorded)
