from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

def calculate_adj_r2(r2, n, p):
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return adj_r2

def get_test_results(grid_search, X_test, y_test, y_pred):
    results = {}
    
    # Store best transformers and params
    best_steps = grid_search.best_estimator_.steps
    results['steps'] = {}
    for step in best_steps:
        results['steps'][step[0]] = step[1]
        
    # Calculte scores
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse = round(rmse, 3)
    results['rmse'] = rmse

    r2 = r2_score(y_test, y_pred)
    r2 = round(r2, 3)
    results['r2'] = r2
    
    adj_r2 = calculate_adj_r2(r2, n=len(y_test), p=X_test.shape[1])
    adj_r2 = round(adj_r2, 3)
    results['r2 ajusté'] = adj_r2

    spearman = spearmanr(y_test, y_pred)[0]
    spearman = round(spearman, 3)
    results['spearman'] = spearman
    
    return results

def print_test_results(results, refit, title):
    print()
    print(title)
    print(" Best estimator: ")
    for key, value in results['steps'].items():
        print(f'  {key} :', value)
    print(" Metrics:")
    print(f"  rmse: {results['rmse']}")
    print(f"  r2: {results['r2']}")
    print(f"  r2 ajusté: {results['r2 ajusté']}")
    print(f"  spearman: {results['spearman']}")
    
def get_comparison_table(test_results, title):
    
    markdown_table = f"""| {title}      | FR                                       | DE                                       |
|-------------|--------------------------------------------|--------------------------------------------|
| Imputer     | {test_results['fr']['steps']['imputer']}   | {test_results['de']['steps']['imputer']}   |
| Scaler      | {test_results['fr']['steps']['scaler']}    | {test_results['de']['steps']['scaler']}    |
| Selection   | {test_results['fr']['steps']['selection']} | {test_results['de']['steps']['selection']} |
| Model       | {test_results['fr']['steps']['model']}     | {test_results['de']['steps']['model']}     |
| RMSE        | {test_results['fr']['rmse']}               | {test_results['de']['rmse']}               |
| R2          | {test_results['fr']['r2']}                 | {test_results['de']['r2']}                 |
| R2 ajusté   | {test_results['fr']['r2 ajusté']}          | {test_results['de']['r2 ajusté']}          |
| Spearman    | {test_results['fr']['spearman']}           | {test_results['de']['spearman']}           |"""
    
    return markdown_table