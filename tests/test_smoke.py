import sys;from pathlib import Path;sys.path.insert(0,str(Path(__file__).parent.parent))
from src.data import make_synthetic;from src.model import fit_and_evaluate;from src.core import score_stability
def test_data():d=make_synthetic(500);assert d["n_samples"]==500
def test_stability():assert len(score_stability([0.1,0.5,0.9]))==5
def test_fit():d=make_synthetic(500);m,met=fit_and_evaluate(d);assert met["auc"]>0.5
if __name__=="__main__":test_data();test_stability();test_fit();print("OK")
