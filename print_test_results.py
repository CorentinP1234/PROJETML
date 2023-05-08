import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    # Store y_pred and y_test
    results['y_pred'] = y_pred
    results['y_test'] = y_test
    
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
    fr_model = str(test_results['fr']['steps']['model']).replace('\n', ' ').replace('\t', ' ').replace('  ', ' ')
    de_model = str(test_results['de']['steps']['model']).replace('\n', ' ').replace('\t', ' ').replace('  ', ' ')

    markdown_table = f"""| {title}      | FR                                       | DE                                       |
|-------------|--------------------------------------------|--------------------------------------------|
| Imputer     | {test_results['fr']['steps']['imputer']}   | {test_results['de']['steps']['imputer']}   |
| Scaler      | {test_results['fr']['steps']['scaler']}    | {test_results['de']['steps']['scaler']}    |
| Model       | {fr_model}                                 | {de_model}                                 |
| RMSE        | {test_results['fr']['rmse']}               | {test_results['de']['rmse']}               |
| R2          | {test_results['fr']['r2']}                 | {test_results['de']['r2']}                 |
| R2 ajusté   | {test_results['fr']['r2 ajusté']}          | {test_results['de']['r2 ajusté']}          |
| Spearman    | {test_results['fr']['spearman']}           | {test_results['de']['spearman']}           |"""
    
    return markdown_table



import matplotlib.pyplot as plt

def plot_scatter_predictions(y_test, y_pred, title='', xlabel='True Values', ylabel='Predictions'):
    plt.figure()
    plt.scatter(y_test, y_pred)
    
    # x = y (diagonal) black dotted line
    plt.axline((0,0), slope=1, color='k', linestyle='--', label='y=x', linewidth=0.8)
    plt.legend()
    
    plt.axis('equal')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Center on (0,0)
    yabs_max = abs(max(y_pred, key=abs))
    xabs_max = abs(max(y_test, key=abs))
    plt.ylim([-yabs_max, yabs_max])
    plt.xlim([-xabs_max,xabs_max])

    
def plot_top_coefficients_or_importances(grid_search, X, num_var=10, coef_type='coef_', title=''):
    
    # Extrait les coefficients du meilleur estimateur du GridSearchCV pour les modèles linéaires
    if coef_type == 'coef_':
        coef = grid_search.best_estimator_.named_steps['model'].coef_
        coef_df = pd.DataFrame.from_dict({'variable': X.columns, f'{coef_type}': coef.flatten()})
    
     # Extrait les importances des variables du meilleur estimateur du GridSearchCV pour les modèles basés sur les arbres
    elif coef_type == 'feature_importances_':
        importances = grid_search.best_estimator_.named_steps['model'].feature_importances_
        coef_df = pd.DataFrame.from_dict({'variable': X.columns, f'{coef_type}': importances})

    # Trie le DataFrame par la valeur absolue des coefficients ou des importances
    coef_df = coef_df.reindex(coef_df[coef_type].abs().sort_values(ascending=False).index)

    # Sélectionne les coefficients ou importances non nuls et les noms des variables correspondantes pour les N_VAR premières variables
    coef_df = coef_df[coef_df[coef_type] != 0].head(num_var).sort_values(by=coef_type, ascending=False)

    # Trace les coefficients ou importances des variables sous forme de graphique à barres
    if coef_type == 'coef_':
        sns.barplot(x=f'{coef_type}', y='variable', data=coef_df, color='tab:blue')
        plt.title(f'Les {num_var} coefficients les plus importants ({title})')
        plt.xlabel('Coefficient')
    elif coef_type == 'feature_importances_':
        sns.barplot(x=f'{coef_type}', y='variable', data=coef_df, color='tab:blue')
        plt.title(f'Les {num_var} importances de variables les plus importantes ({title})')
        plt.xlabel('Importance')

    plt.ylabel('Variable')
    plt.tight_layout()
    plt.show()
    
    return coef_df
   
