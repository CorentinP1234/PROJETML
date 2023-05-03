from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from scipy.stats import spearmanr
import numpy as np

def spearmanr_scorer(y, y_pred):
    try:
        correlation, _ = spearmanr(y, y_pred)
        if np.isfinite(correlation):
            return correlation
        else:
            print("Correlation Non Finie ")
            return 0.0 
    except Exception as e:
        print(f"Error calculating Spearman correlation: {e}")
        return 0.0 

def r2_scorer(y, y_pred):
    return r2_score(y, y_pred)

def rmse_scorer(y, y_pred):
    return mean_squared_error(y, y_pred, squared=False)  # RMSE


r2_scoring = make_scorer(r2_scorer, greater_is_better=True)
rmse_scoring = make_scorer(rmse_scorer, greater_is_better=False)
spearman_scoring = make_scorer(spearmanr_scorer, greater_is_better=True)