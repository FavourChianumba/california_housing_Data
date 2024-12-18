
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, RandomizedSearchCV # type: ignore
from sklearn.linear_model import Ridge # type: ignore
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, HuberRegressor, LassoCV, RidgeCV, SGDRegressor
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def get_model_pipelines(preprocessing_pipeline):
    """
    Returns a dictionary of initialized model pipelines.
    """
    return {
    'Linear Regression': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', LinearRegression())
    ]),
    'Random Forest': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    'Gradient Boosting': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]),
    'XGBoost': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]),
    'K-Nearest Neighbors': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', KNeighborsRegressor(n_neighbors=5))
    ]),
    'Support Vector Regression': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ]),
    'ElasticNet': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42))
    ]),
    'LightGBM': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]),
    'Huber Regressor': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', HuberRegressor(epsilon=1.35))
    ]),
    'LassoCV': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', LassoCV(cv=5, random_state=42))
    ]),
    'RidgeCV': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', RidgeCV(cv=5))
    ]),
    'MLP Regressor': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', random_state=42))
    ]),
    'SGDRegressor': Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('model', SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))
    ])
}

def evaluate_model(y_true, y_pred):
    """
    Prints and returns evaluation metrics for a model.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R² Score: {r2}")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

