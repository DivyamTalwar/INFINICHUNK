from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProvenanceTests(unittest.TestCase):
    def test_required_records_exist(self):
        for filename in ("LICENSE", "Notice.txt", "UPSTREAM.md"):
            self.assertTrue((ROOT / filename).is_file(), filename)

    def test_direct_upstream_is_named(self):
        upstream = (ROOT / "UPSTREAM.md").read_text()
        self.assertIn("McGill-NLP/the-markovian-thinker", upstream)
        self.assertIn("volcengine/verl", upstream)
        self.assertIn("7b870e93db7cc9beb105fd77950ff9890182b063", upstream)

    def test_inherited_authors_are_preserved(self):
        for filename in (
            "verl/experimental/agent_loop/infinichunk_agent_loop.py",
            "verl/experimental/agent_loop/infinichunk_trimmers.py",
            "verl/trainer/infinichunk_ppo/ray_trainer.py",
        ):
            source = (ROOT / filename).read_text()
            self.assertIn("Amirhossein Kazemnejad", source, filename)
            self.assertIn("Milad Aghajohari", source, filename)
            self.assertIn("Kamran Chitsaz", source, filename)


if __name__ == "__main__":
    unittest.main()
