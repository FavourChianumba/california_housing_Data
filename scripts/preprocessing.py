import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


# Haversine formula to calculate distance between two points (latitude, longitude)
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface given their latitudes and longitudes.
    
    Parameters:
    lat1, lon1 (float): Latitude and longitude of the first point (in radians).
    lat2, lon2 (float): Latitude and longitude of the second point (in radians).
    
    Returns:
    float: The distance between the points in miles.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])  # Convert degrees to radians
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    miles = 3956 * c  # Earth's radius in miles
    return miles

# Function to calculate minimum distance to the coastline using Haversine formula
def min_distance_to_coast(lat, lon, coastline_points):
    """
    Calculate the minimum distance from a point (lat, lon) to the coastline.
    
    Parameters:
    lat (float): Latitude of the point.
    lon (float): Longitude of the point.
    coastline_points (list): List of tuples representing latitude and longitude of coastline points.
    
    Returns:
    float: The minimum distance to the coastline.
    """
    distances = [haversine(lat, lon, coast_lat, coast_lon) for coast_lat, coast_lon in coastline_points]
    return min(distances)

# Feature engineering class that creates new features based on existing ones
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, coastline_points):
        """
        Initialize the FeatureEngineering class with coastline points for distance calculation.
        
        Parameters:
        coastline_points (list): List of tuples containing latitude and longitude of coastline points.
        """
        self.coastline_points = coastline_points

    def fit(self, X, y=None):
        """
        Fit method required for scikit-learn transformers.
        
        Parameters:
        X (DataFrame): The data to be transformed.
        y (optional): Target variable, not needed here.
        
        Returns:
        self: Returns the instance of the transformer.
        """
        return self  # No fitting required for feature engineering

    def transform(self, X):
        """
        Apply feature transformations to the input data.
        
        Parameters:
        X (DataFrame): The input data.
        
        Returns:
        DataFrame: The transformed data with new features.
        """
        X = X.copy()  # Avoid modifying the original dataframe

        # Log transformations to handle skewed numerical features
        X['total_rooms'] = np.log(X['total_rooms'] + 1)
        X['total_bedrooms'] = np.log(X['total_bedrooms'] + 1)
        X['population'] = np.log(X['population'] + 1)
        X['households'] = np.log(X['households'] + 1)

        # New feature creation: Ratios and interactions between features
        X['bedroom_ratio'] = X['total_bedrooms'] / X['total_rooms']
        X['rooms_per_person'] = X['total_rooms'] / X['population']
        X['income_per_household'] = X['median_income'] / X['households']
        X['bedrooms_per_household'] = X['total_bedrooms'] / X['households']
        X['lat_long_interaction'] = X['latitude'] * X['longitude']

        # Calculate distance to coast using Haversine formula
        X['dist_to_coast'] = X.apply(
            lambda row: min_distance_to_coast(row['latitude'], row['longitude'], self.coastline_points), axis=1
        )

        # Additional feature interactions that combine multiple existing features
        X['population_per_household'] = X['population'] / X['households']
        X['rooms_per_household'] = X['total_rooms'] / X['households']
        X['high_income_area'] = (X['median_income'] > X['median_income'].median()).astype(int)  # Binary feature for high-income areas
        X['residential_density_per_capita'] = (X['total_rooms'] / X['population']) * X['median_income']
        X['population_density_per_room'] = X['population'] / X['total_rooms']
        X['population_per_bedroom'] = X['population'] / X['total_bedrooms']
        X['income_to_room_ratio'] = X['median_income'] / X['total_rooms']
        X['income_per_age_of_housing'] = X['median_income'] / X['housing_median_age']
        X['household_room_interaction'] = X['households'] * X['housing_median_age']
        X['households_per_income_category'] = X['households'] / X['high_income_area']
        X['median_age_interaction'] = X['median_income'] * X['housing_median_age']

        return X  # Return transformed dataset with new features

# Outlier handling class to handle missing or infinite values by imputing them
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        """
        Initialize the OutlierHandler with a specified imputation strategy.
        
        Parameters:
        strategy (str): The imputation strategy ('mean' or other).
        """
        self.strategy = strategy

    def fit(self, X, y=None):
        """
        Fit method to calculate the imputation values.
        
        Parameters:
        X (DataFrame): The input data.
        y (optional): Target variable, not used here.
        
        Returns:
        self: Returns the instance of the transformer.
        """
        if self.strategy == 'mean':
            self.fill_values_ = X.replace([np.inf, -np.inf], np.nan).mean()  # Replace infinite values with NaN and compute mean
        return self

    def transform(self, X):
        """
        Transform the data by replacing infinite values with imputed values.
        
        Parameters:
        X (DataFrame): The input data.
        
        Returns:
        DataFrame: The data with missing values filled.
        """
        X = X.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
        X = X.fillna(self.fill_values_)  # Fill NaN with computed values (e.g., mean)
        return X


# Feature selection class
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold='median'):
        self.threshold = threshold
        self.selector = None

    def fit(self, X, y):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        self.selector = SelectFromModel(model, threshold=self.threshold, prefit=True)
        return self

    def transform(self, X):
        return pd.DataFrame(self.selector.transform(X), index=X.index, columns=X.columns[self.selector.get_support()])


# Function to create preprocessing pipeline for numerical and categorical features
def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Create a pipeline for preprocessing numerical and categorical features.
    
    Parameters:
    numerical_features (list): List of numerical feature names.
    categorical_features (list): List of categorical feature names.
    
    Returns:
    preprocessing_pipeline: The combined pipeline for feature preprocessing.
    """
    # Numerical pipeline: Outlier handling, imputation, and scaling
    numerical_pipeline = Pipeline([
        ('outlier_handler', OutlierHandler(strategy='mean')),  # Handle outliers by imputing with the mean
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
        ('scaler', StandardScaler())  # Standardize numerical features (zero mean, unit variance)
    ])

    # Categorical pipeline: One-hot encoding for categorical features
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))  # One-hot encoding with handling for unknown categories
    ])

    # Combine numerical and categorical pipelines into a single column transformer
    preprocessing_pipeline = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessing_pipeline  # Return the combined preprocessing pipeline

# Main preprocessing function that handles the entire pipeline
def preprocess_data(train_data, coastline_points):
    """
    Preprocess the input data by performing feature engineering, splitting features and target, 
    and applying preprocessing pipelines.
    
    Parameters:
    train_data (DataFrame): The input data containing both features and target.
    coastline_points (list): List of tuples with coastline latitude and longitude for distance calculation.
    
    Returns:
    x_train_fe (DataFrame): Transformed feature data with feature engineering applied.
    y_train (Series): Target variable (median_house_value).
    preprocessing_pipeline (ColumnTransformer): The preprocessing pipeline for numerical and categorical features.
    """
    # Separate features (X) and target (y)
    x_train = train_data.drop('median_house_value', axis=1)
    y_train = train_data['median_house_value'].copy()

    # Apply feature engineering to create new features
    feature_engineer = FeatureEngineering(coastline_points=coastline_points)
    x_train_fe = feature_engineer.fit_transform(x_train)  # Apply the feature engineering transformation

    # Identify numerical and categorical features for the pipeline
    numerical_features = x_train_fe.select_dtypes(include=[np.number]).columns.tolist()  # Numerical features
    categorical_features = x_train_fe.select_dtypes(exclude=[np.number]).columns.tolist()  # Categorical features

    # Create preprocessing pipeline based on feature types
    preprocessing_pipeline = create_preprocessing_pipeline(numerical_features, categorical_features)

    # Apply the preprocessing pipeline
    # Apply the preprocessing pipeline
    x_train_preprocessed = preprocessing_pipeline.fit_transform(x_train_fe)

    # ** Why preprocess again here? **
    # At this point, `x_train_preprocessed` is the standardized, encoded version of `x_train_fe`.
    # Feature selection methods like Random Forest expect numerical arrays and work with cleaned data.
    # Preprocessing ensures the data is in the correct format (e.g., no missing values, standardized scales, etc.)
    
    # Apply feature selection
    feature_selector = FeatureSelector(threshold='median')
    x_train_selected = feature_selector.fit_transform(
        pd.DataFrame(x_train_preprocessed, columns=numerical_features + categorical_features),
        y_train)
    
    return x_train_selected, y_train, preprocessing_pipeline
