# Alternative credit metrics: feature interaction strength, score stability
import numpy as np

def feature_interaction_strength(model, X, n_repeats=5):
    X = np.asarray(X, dtype=float)
    base_pred = model.predict_proba(X)[:, 1]
    rng = np.random.default_rng(42)
    strengths = []
    n_features = X.shape[1]
    for j in range(n_features):
        for i in range(j + 1, n_features):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, i] = Xp[perm, i]
            diff = np.abs(base_pred - model.predict_proba(Xp)[:, 1]).mean()
            strengths.append({"pair": (j, i), "drop": float(diff)})
    return sorted(strengths, key=lambda x: x["drop"], reverse=True)[:5]

def score_stability(probas, bins=5):
    arr = np.asarray(probas, dtype=float)
    return [float((arr > i / bins).mean()) for i in range(bins)]