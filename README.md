# **Housing Price Prediction Using Advanced Machine Learning Models**

This repository presents a comprehensive machine learning project for predicting housing prices. The project explores a variety of approaches, from individual regression models to advanced ensemble methods, including weighted stacking and neural network meta-models, to achieve the best prediction accuracy. This project showcases my ability to preprocess data, apply machine learning algorithms, and evaluate model performance.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Data Description](#data-description)
3. [Project Workflow](#project-workflow)
4. [Models and Methodologies](#models-and-methodologies)
5. [Results and Performance](#results-and-performance)
6. [How to Use This Repository](#how-to-use-this-repository)
7. [Future Work](#future-work)
8. [Acknowledgments](#acknowledgments)

---

## **Overview**

Housing price prediction is a critical application of machine learning, widely used in real estate, banking, and investment industries. The goal of this project is to accurately predict housing prices using historical data on housing features. By employing a variety of machine learning techniques and experimenting with advanced ensemble methods, we aim to achieve robust predictions.

---

## **Data Description**

We used a publicly available dataset called [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices). This dataset includes the following features:

1. **longitude**: A measure of how far west a house is; higher values indicate locations farther west.
2. **latitude**: A measure of how far north a house is; higher values indicate locations farther north.
3. **housingMedianAge**: Median age of houses within a block; lower values suggest newer buildings.
4. **totalRooms**: Total number of rooms within a block.
5. **totalBedrooms**: Total number of bedrooms within a block.
6. **population**: Total number of people residing within a block.
7. **households**: Total number of households (groups of people residing in home units) within a block.
8. **medianIncome**: Median income for households in a block, measured in tens of thousands of US dollars.
9. **medianHouseValue**: Median house value for households in a block, measured in US dollars.
10. **oceanProximity**: A categorical feature representing the house's location relative to the ocean (e.g., "NEAR BAY," "INLAND").

---

### **Preprocessing Steps**

#### 1. Stratified Test Splitting
To ensure a balanced division of data into training and test sets, we applied **stratified sampling** based on the `medianIncome` feature. This was crucial because `medianIncome` strongly correlates with housing prices, and an unrepresentative split could lead to skewed evaluation metrics. Here's how we performed stratified splitting:

- **Income Categories**: 
   - We divided `medianIncome` into discrete bins using a custom transformation to create a temporary feature, `income_cat`. 
   - This feature categorized `medianIncome` into five levels, with higher incomes grouped into the last category.
- **Stratification**: 
   - Using this `income_cat` feature, we applied stratified sampling to split the data while maintaining similar proportions of each income category in both the training and test sets.
- **Benefits of Stratification**: 
   - This approach ensures the test set reflects the overall dataset distribution, reducing the risk of biased performance metrics. It is especially helpful when working with medium-sized datasets or sparse feature spaces.

Finally, the `income_cat` feature was dropped after splitting to avoid data leakage into the training or test sets.

---

#### 2. Handling Categorical Features
The `oceanProximity` feature is a categorical variable containing labels like "NEAR BAY," "ISLAND," and "INLAND." These non-numeric values cannot be directly used by machine learning models like XGBoost or Ridge Regression. To handle this:

- We applied **one-hot encoding** to convert `oceanProximity` into binary features. For example, the category "NEAR BAY" was transformed into a binary column (`NEAR BAY`) where `1` indicates presence and `0` indicates absence.
- To avoid multicollinearity, we used the `drop_first=True` option, which removes one binary column as a reference. This ensures the encoded features remain independent and prevents redundancy.

This preprocessing step ensures compatibility with machine learning models and allows the models to identify patterns related to specific house locations.

---

### **Dataset Summary**
- The dataset was divided into **training (80%)** and **test (20%)** subsets using stratified sampling to preserve the distribution of the `medianIncome` feature.
- After preprocessing, the training and test sets were cleaned and encoded, making them ready for use in various machine learning models.

---

## **Project Workflow**

## Workflow

graph TD
    A[Data Collection] --> B[Exploratory Data Analysis]
    B --> C[Feature Engineering]
    C --> D[Model Development]
    D --> E[Ensemble Methods]
    E --> F[Performance Evaluation]
    
    subgraph "Feature Engineering"
    C1[Log Transformations] --> C
    C2[Interaction Features] --> C
    C3[Geographical Features] --> C
    C4[Custom Transformers] --> C
    end
    
    subgraph "Model Development"
    D1[Linear Models] --> D
    D2[Tree-Based Models] --> D
    D3[Advanced Regression] --> D
    D4[Neural Networks] --> D
    end
    
    subgraph "Ensemble Methods"
    E1[Weighted Averaging] --> E
    E2[Stacking] --> E
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#f9f,stroke:#333,stroke-width:2px



1. **Exploratory Data Analysis (EDA):**
   - Investigated feature distributions and correlations.
   - Visualized patterns between features and target values.
   - Identified and handled missing values.

2. **Feature Engineering and Preprocessing**
   ### Advanced Feature Transformation Techniques
   - **Comprehensive Feature Creation**
     - Log transformations to handle skewed numerical features
     - Generated interaction features capturing complex relationships
     - Created derived features including:
       * Bedroom-to-room ratio
       * Rooms per person
       * Income per household
       * Residential density metrics
       * Geographical interaction features

   ### Preprocessing Pipeline
      graph LR
         A[Raw Data] --> B[Custom Feature Engineering]
         B --> C[Missing Value Imputation]
         C --> D[Outlier Handling]
         D --> E[Feature Scaling]
         E --> F[Feature Selection]
         
         style A fill:#f9f,stroke:#333,stroke-width:2px
         style B fill:#bbf,stroke:#333,stroke-width:2px
         style C fill:#bbf,stroke:#333,stroke-width:2px
         style D fill:#bbf,stroke:#333,stroke-width:2px
         style E fill:#bbf,stroke:#333,stroke-width:2px
         style F fill:#f9f,stroke:#333,stroke-width:2px

   - **Robust Data Preparation**
     - Custom `FeatureEngineering` transformer for advanced feature generation
     - Haversine distance calculation for geographical features
     - Outlier handling using mean imputation
     - Numerical feature scaling with StandardScaler
     - Categorical feature encoding using OneHotEncoder
     - Feature selection using Random Forest feature importance

   ### Key Preprocessing Innovations
   - Dynamic feature type identification
   - Handling of missing and infinite values
   - Preservation of feature interpretability
   - Scalable preprocessing approach compatible with multiple models

3. **Modeling and Optimization**
   ### Comprehensive Modeling Strategy
   - **Extensive Model Exploration**
     - Evaluated 13 different regression algorithms
     - Implemented scikit-learn Pipeline for consistent preprocessing
     - Explored both linear and non-linear modeling approaches

   ### Advanced Modeling Techniques
   - **Ensemble Methods**
     - Implemented weighted stacking ensemble
     - Experimented with different meta-models
     - Optimized ensemble weights through iterative refinement

   ### Hyperparameter and Model Optimization
   - Systematic model comparison
   - Cross-validation for robust performance estimation
   - Explored model-specific hyperparameters
   - Implemented feature selection to reduce model complexity

4. **Model Evaluation**
   ### Rigorous Performance Assessment
   - **Comprehensive Evaluation Metrics**
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Mean Absolute Error (MAE)
     - R² Score

   ### Evaluation Approach
   - Systematic comparison across multiple algorithms
   - Detailed performance tracking
   - Identification of model strengths and limitations
   - Transparent reporting of model performance

   ### Insights from Evaluation
   - Identified challenges in predictive modeling
   - Highlighted the complexity of housing price prediction
   - Provided clear path for future model improvements

---

## **Models and Methodologies**

### **1. Machine Learning Algorithms**
The project explores a diverse range of regression algorithms to predict housing prices, leveraging scikit-learn and gradient boosting libraries:

#### **Linear Models**
- **Linear Regression**: A baseline linear model capturing simple linear relationships
- **ElasticNet**: Combines L1 and L2 regularization to prevent overfitting
- **Lasso (LassoCV)**: Performs feature selection through L1 regularization
- **Ridge (RidgeCV)**: Applies L2 regularization to prevent model complexity
- **Huber Regressor**: Robust to outliers by using a combination of squared and absolute loss

#### **Tree-Based Models**
- **Random Forest**: Ensemble method creating multiple decision trees
- **Gradient Boosting**: Sequential tree building to minimize prediction errors
- **XGBoost**: High-performance gradient boosting with advanced regularization
- **LightGBM**: Gradient boosting framework optimized for efficiency and speed

#### **Advanced Regression Techniques**
- **Support Vector Regression (SVR)**: Applies kernel tricks to handle non-linear relationships
- **K-Nearest Neighbors**: Predicts based on proximity to similar data points
- **Stochastic Gradient Descent (SGD) Regressor**: Online learning algorithm for large datasets

#### **Neural Network**
- **Multi-Layer Perceptron (MLP) Regressor**: Flexible neural network with configurable architecture

### **2. Model Evaluation Strategy**
Each model was rigorously evaluated using multiple performance metrics:
- **Mean Squared Error (MSE)**: Measures average squared prediction error
- **Root Mean Squared Error (RMSE)**: Standard deviation of prediction residuals
- **Mean Absolute Error (MAE)**: Average magnitude of prediction errors
- **R² Score**: Proportion of variance explained by the model

### **3. Advanced Modeling Techniques**

#### **Preprocessing Pipeline**
- Custom preprocessing steps including:
  - Feature engineering
  - Outlier handling
  - Numerical feature scaling
  - Categorical feature encoding
  - Feature selection using Random Forest importance

#### **Model Selection Approach**
- Comprehensive model comparison across multiple algorithms
- Standardized scikit-learn Pipeline for consistent preprocessing
- Cross-validation to ensure robust performance estimation

### **4. Experimental Ensemble Methods**
While not explicitly shown in the current implementation, the project framework supports advanced ensemble techniques:

#### **Potential Ensemble Strategies**
- **Weighted Averaging**: Combining predictions with optimized weights
- **Stacking Ensemble**: 
  - Using Ridge Regression as a meta-model
  - Potential for neural network meta-model to capture non-linear interactions
- **Boosting Techniques**: Leveraging gradient boosting frameworks

### **5. Future Improvements**
- Hyperparameter tuning using GridSearchCV
- Implementing more sophisticated ensemble methods
- Exploring advanced feature engineering techniques

---

## **Results and Performance**

### **Model Comparison**

| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
| Linear Regression | 11,567,650,000,000.00 | 3,401,125.00 | 2,880,969.00 | -855.911 |
| Random Forest | 12,872,580,000.00 | 113,457.40 | 88,170.69 | 0.046 |
| Gradient Boosting | 12,609,820,000.00 | 112,293.50 | 83,052.06 | 0.066 |
| XGBoost | 12,183,500,000.00 | 110,378.90 | 83,520.26 | 0.097 |
| K-Nearest Neighbors | 12,766,760,000.00 | 112,990.10 | 91,155.05 | 0.054 |
| Support Vector Regression | 14,311,470,000.00 | 119,630.60 | 89,535.14 | -0.060 |
| ElasticNet | 25,233,750,000.00 | 158,851.40 | 124,221.60 | -0.869 |
| LightGBM | 11,574,950,000.00 | 107,587.00 | 80,157.93 | 0.143 |
| Huber Regressor | 7,607,470,000,000.00 | 2,758,164.00 | 2,729,654.00 | -562.548 |
| LassoCV | 5,235,640,000,000.00 | 2,288,152.00 | 2,244,016.00 | -386.847 |
| RidgeCV | 10,340,570,000,000.00 | 3,215,676.00 | 2,817,743.00 | -765.011 |
| MLP Regressor | 4,443,168,000,000.00 | 2,107,882.00 | 2,077,751.00 | -328.142 |
| SGDRegressor | 3,993,377,000,000.00 | 1,998,344.00 | 1,981,435.00 | -294.822 |

### **Model Selection**

#### **Chosen Model: LightGBM**
LightGBM emerged as the top-performing model based on comprehensive evaluation metrics:

- **Mean Squared Error (MSE)**: 11,574,950,000.00
- **Root Mean Squared Error (RMSE)**: 107,587.00
- **Mean Absolute Error (MAE)**: 80,157.93
- **R² Score**: 0.143

#### **Selection Rationale**
The LightGBM model was selected due to its superior performance across key metrics:
- Lowest Mean Squared Error (MSE)
- Lowest Root Mean Squared Error (RMSE)
- Lowest Mean Absolute Error (MAE)
- Highest R² Score

While the R² value of 0.143 indicates that the model explains only a modest proportion of the variance in the target variable, it consistently outperformed alternative models such as Gradient Boosting and XGBoost.

### **Performance Insights**
- Most linear models (Linear Regression, RidgeCV) showed extremely poor performance with negative R² scores
- Ensemble and tree-based methods (Random Forest, Gradient Boosting, XGBoost, LightGBM) demonstrated significantly better predictive capabilities
- The modest R² score suggests complex underlying patterns in the housing price data that require further feature engineering or advanced modeling techniques

### Further Base Models Performance

Our analysis began with training two powerful gradient boosting models:

#### LightGBM
- MSE: 7,872,852,810.00
- RMSE: 88,729.10
- MAE: 66,068.63
- R² Score: 0.4168

#### XGBoost
- MSE: 7,636,181,829.56
- RMSE: 87,385.25
- MAE: 65,628.05
- R² Score: 0.4343

### Ensemble Methods

We explored various ensemble techniques to improve model performance:

#### 1. Weighted Averaging
We tested different weight combinations between LightGBM and XGBoost:

| Weight Distribution (LGB/XGB) | R² Score | RMSE | MAE |
|------------------------------|----------|------|-----|
| 0.1/0.9 | 0.4338 | 87,425.15 | 65,597.59 |
| 0.01/0.99 | 0.4343 | 87,388.28 | 65,624.25 |

#### 2. Stacking Ensembles
We implemented two stacking approaches using different meta-learners:

##### Ridge Regression Meta-learner
- Best weights: LightGBM (0.1) / XGBoost (0.9)
- MSE: 7,505,194,082.97
- RMSE: 86,632.52
- MAE: 64,812.11
- R² Score: 0.4440 (Best performing model)

##### Neural Network Meta-learner
- Best weights: LightGBM (0.1) / XGBoost (0.9)
- MSE: 7,554,026,894.22
- RMSE: 86,913.91
- MAE: 65,157.73
- R² Score: 0.4404

### Model Performance Visualization


graph TB
    subgraph "Model Training Pipeline"
        A[Individual Models] --> B[Weighted Averaging]
        A --> C[Stacking Ensembles]
        
        subgraph "Base Models"
            D[LightGBM<br/>R² = 0.4168] --> A
            E[XGBoost<br/>R² = 0.4343] --> A
        end
        
        subgraph "Ensemble Methods"
            B --> F[0.1/0.9 Split<br/>R² = 0.4338]
            B --> G[0.01/0.99 Split<br/>R² = 0.4343]
            C --> H[Ridge Meta-learner<br/>R² = 0.4440]
            C --> I[Neural Meta-learner<br/>R² = 0.4404]
        end
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style D fill:#dfd,stroke:#333,stroke-width:2px
    style E fill:#dfd,stroke:#333,stroke-width:2px
    style F fill:#ddf,stroke:#333,stroke-width:2px
    style G fill:#ddf,stroke:#333,stroke-width:2px
    style H fill:#ddf,stroke:#333,stroke-width:2px
    style I fill:#ddf,stroke:#333,stroke-width:2px

### Key Findings

Base Models: XGBoost outperformed LightGBM in individual performance
Weighted Averaging: A strong bias towards XGBoost (0.99 weight) produced optimal results
Stacking Ensembles: Ridge regression meta-learner achieved the best overall performance
Weight Distribution: Consistently found 0.1/0.9 (LGB/XGB) to be the optimal weight ratio across different ensemble methods
Model Selection: Tree-based models significantly outperformed linear models and other traditional approaches
Best Performance: Achieved R² score of 0.4440 with Ridge-based stacking ensemble



### **Potential Improvements**
- Advanced feature engineering
- Hyperparameter tuning
- Exploring more sophisticated ensemble methods
- Collecting additional relevant features

---

## **How to Use This Repository**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/housing-price-prediction.git
   cd housing-price-prediction
