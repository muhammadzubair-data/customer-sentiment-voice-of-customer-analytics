import csv, json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class PublicationIntegrity(unittest.TestCase):
    def test_scale(self):
        with open(ROOT/'outputs/project_metrics.json') as f: d=json.load(f)
        self.assertEqual(d['feedback_interactions'],120000)
        self.assertEqual(d['customers'],8000)
    def test_sentiment_metrics(self):
        d=json.load(open(ROOT/'outputs/sentiment_model_comparison.json'))
        self.assertAlmostEqual(d['vader_baseline']['macro_f1'],0.48832421626083117,places=9)
        self.assertAlmostEqual(d['tfidf_logreg']['macro_f1'],1.0,places=9)
    def test_topic_metrics(self):
        d=json.load(open(ROOT/'outputs/topic_discovery_evaluation.json'))['method_comparison']
        self.assertGreater(d['nmf']['purity_vs_ground_truth_theme'],d['lda']['purity_vs_ground_truth_theme'])
    def test_emerging_issue(self):
        d=json.load(open(ROOT/'outputs/emerging_issue_alerts.json'))
        self.assertEqual(len(d),1); self.assertEqual(d[0]['theme'],'Payment Issues (Double Charge)'); self.assertAlmostEqual(d[0]['recent_7d_share_pct'],7.45,places=2)
    def test_operational_code_avoids_true_themes(self):
        for f in ['scripts/06_emerging_issue_detection.py','scripts/07_priority_scoring_explainability.py']:
            s=(ROOT/f).read_text(); active='\n'.join(x for x in s.splitlines() if not x.lstrip().startswith('#'))
            self.assertNotIn('["true_themes"]',active); self.assertIn('inferred_themes',active)
    def test_priority_outputs(self):
        with open(ROOT/'outputs/project_metrics.json') as f: d=json.load(f)
        self.assertEqual(d['priority_distribution'],{'Critical':516,'High':49075,'Medium':58664,'Low':11745})
if __name__=='__main__': unittest.main()
