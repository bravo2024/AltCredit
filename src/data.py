from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["app_usage_days","social_connectivity_score","bill_payment_consistency","transaction_regularity","behavioral_score","device_freshness","location_stability_months","employment_sector","income_volatility","credit_utilization"]
CATEGORICAL_FEATURES = ["employment_sector"]
NUMERICAL_FEATURES = ["app_usage_days","social_connectivity_score","bill_payment_consistency","transaction_regularity","behavioral_score","device_freshness","location_stability_months","income_volatility","credit_utilization"]
TARGET_NAME = "default_risk"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "app_usage_days": rng.uniform(30,1095,size=n).astype(int),
        "social_connectivity_score": rng.uniform(0,100,size=n).round(1),
        "bill_payment_consistency": rng.beta(6,2,size=n).round(3),
        "transaction_regularity": rng.uniform(0,100,size=n).round(1),
        "behavioral_score": rng.normal(60,15,size=n).clip(0,100).round(1),
        "device_freshness": rng.uniform(0,100,size=n).round(1),
        "location_stability_months": rng.exponential(scale=24,size=n).clip(1,120).astype(int),
        "employment_sector": rng.choice(["formal","informal","gig","self_employed","unemployed"],size=n,p=[0.35,0.25,0.20,0.15,0.05]),
        "income_volatility": rng.uniform(0,1,size=n).round(3),
        "credit_utilization": rng.uniform(0,1,size=n).round(3),
    })
    usage=np.clip(df["app_usage_days"]/1095,0,1); social=df["social_connectivity_score"]/100
    bill=df["bill_payment_consistency"]; trans=df["transaction_regularity"]/100; beh=df["behavioral_score"]/100
    device=np.clip((100-df["device_freshness"])/100,0,1); loc=np.clip(df["location_stability_months"]/120,0,1)
    sector_map={"formal":0,"informal":0.3,"gig":0.5,"self_employed":0.7,"unemployed":1}
    sector=df["employment_sector"].map(sector_map).values; vol=df["income_volatility"]; cu=df["credit_utilization"]
    log_odds = -2.0 + 0.3*usage - 0.2*social - 0.5*bill + 0.1*trans - 0.3*beh + 0.2*device - 0.1*loc + 0.3*sector + 0.4*vol + 0.3*cu + rng.normal(0,0.5,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,82)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(default_risk=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
