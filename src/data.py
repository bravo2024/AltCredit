from __future__ import annotations
import numpy as np; import pandas as pd
FEATURES=["app_usage_daily","bill_pay_reliability","recharge_freq","social_connections","digital_footprint","income_estimate","savings_balance","merchant_diversity"]
def make_synthetic(n=5000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({f:rng.beta(3,2,n).round(3) for f in FEATURES})
    score=sum(df[f]*w for f,w in zip(FEATURES,[0.2,0.25,0.1,0.05,0.1,0.15,0.1,0.05]))
    prob=1/(1+np.exp(-(3*score-2)));y=(prob>np.percentile(prob,75)).astype(float)
    return{"X":df,"y":y,"features":FEATURES,"categorical_features":[],"numerical_features":FEATURES,"n_samples":n,"positive_rate":y.mean()}
