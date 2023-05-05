from sklearn.feature_selection import RFECV

def get_features(X, y, model, min_features=1):
    rfecv = RFECV(model, scoring='neg_mean_squared_error', n_jobs=-1, min_features_to_select=min_features)
    rfecv.fit_transform(X, y)
    X = X[X.columns[rfecv.support_]]
    return X