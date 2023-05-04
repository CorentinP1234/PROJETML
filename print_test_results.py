from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

def calculate_adj_r2(r2, n, p):
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return adj_r2



def print_test_results(grid_search, X_test, y_test):
    best_steps = grid_search.best_estimator_.steps
    print("Best estimator: ")
    for step in best_steps:
        print(f'  {step}')
        
    y_pred = grid_search.predict(X_test)
        
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse = round(rmse, 3)
    print(f"rmse: {rmse}")

    r2 = r2_score(y_test, y_pred)
    r2 = round(r2, 3)
    print(f"r2: {r2}")
    
    adj_r2 = calculate_adj_r2(r2, n=len(y_test), p=X_test.shape[1])
    adj_r2 = round(adj_r2, 3)
    print(f"Adjusted R-squared: {adj_r2}")

    spearman = spearmanr(y_test, y_pred)[0]
    spearman = round(spearman, 3)
    print(f"spearman: {spearman}")
    
    print('Done')
    return {'rmse': rmse, 'r2': r2, 'spearman': spearman}

