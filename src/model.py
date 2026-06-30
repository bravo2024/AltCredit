from __future__ import annotations
import numpy as np;from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from src.core import feature_interaction_strength,score_stability
def fit_and_evaluate(data,seed=42):
    X,y=data["X"].values.astype(float),data["y"].values
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,stratify=y,random_state=seed)
    gbm=GradientBoostingClassifier(n_estimators=150,learning_rate=0.05,max_depth=5,random_state=seed)
    gbm.fit(Xtr,ytr);proba=gbm.predict_proba(Xte)[:,1]
    from sklearn.metrics import roc_auc_score
    auc=float(roc_auc_score(yte,proba));intx=feature_interaction_strength(gbm,Xte)
    return{"model":gbm},{"auc":auc,"top_interactions":intx,"score_dist":score_stability(proba)}
