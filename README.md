# DA5401 Assignment 8 - Ensemble Learning for Bike Sharing Demand Prediction

## File Structure
```
├── assn8.ipynb   # Main analysis notebook
├── README.md    # This file
└── data/
    └── hour.csv # Bike sharing hourly demand data
```

## Notebook Structure

### Imports & Configuration
- Import required libraries (scikit-learn, pandas, numpy, matplotlib, seaborn, optuna)
- Set up plotting configurations and RMSE evaluation function
- Configure warnings and random seeds for reproducibility

### Data Loading & Preprocessing
- Load bike sharing hourly demand dataset from UCI repository
- Feature engineering with one-hot encoding for categorical variables
- Train-test split with target variable distribution analysis
- Data exploration and visualization

### Baseline Models Implementation
- **Linear Regression** - Simple linear baseline model
- **Decision Tree Regressor** - Non-linear baseline with depth=6
- Performance comparison and baseline establishment

### Bagging Regressor Implementation
- **Theory**: Bootstrap aggregation for variance reduction
- **Optuna Optimization**: Hyperparameter tuning (max_depth, n_estimators)
- **Performance Analysis**: RMSE evaluation and variance reduction evidence
- **Bias-Variance Discussion**: Theoretical and empirical analysis

### Gradient Boosting Regressor Implementation  
- **Theory**: Sequential learning for bias reduction
- **Optuna Optimization**: Hyperparameter tuning (n_estimators, learning_rate, max_depth)
- **Performance Analysis**: RMSE evaluation and bias reduction evidence
- **Superior Performance**: Comparison with Bagging and baseline models

### K-Nearest Neighbors Regressor Implementation
- **Theory**: Instance-based learning for ensemble diversity
- **Optuna Optimization**: Hyperparameter tuning (n_neighbors, weights, metric)
- **Strategic Role**: Contribution to stacking ensemble diversity
- **Local Pattern Recognition**: Non-parametric learning analysis

### Stacking Regressor Implementation
- **Theory**: Meta-learning for optimal model combination
- **Architecture**: Level-0 learners (KNN, Bagging, Boosting) + Ridge meta-learner
- **Optuna Optimization**: Ridge alpha parameter tuning for meta-learner
- **Performance Analysis**: Combined bias-variance optimization

### Comprehensive Model Comparison
- **Final Results Table**: RMSE comparison across all models
- **Performance Visualization**: Horizontal bar chart with annotations
- **Best Model Selection**: Statistical significance and improvement analysis

### Bias-Variance Trade-off Demonstration
- **Efficient Analysis**: Cross-validation stability and prediction consistency
- **Variance Reduction Evidence**: Bagging vs Decision Tree comparison
- **Bias Reduction Evidence**: Boosting superiority demonstration  
- **Visual Proof**: Four-panel comprehensive visualization
- **Theoretical Validation**: Empirical evidence for ensemble learning principles


## Input Data
The analysis uses the **Bike Sharing Dataset** from the UCI Machine Learning Repository, a regression benchmark for demand forecasting in bike-sharing systems. This dataset contains hourly bike rental data with weather conditions, temporal information, and user patterns. The dataset includes:

- **17,379 hourly records** spanning 2 years (2011-2012)
- **16 features** including weather conditions, temporal factors, and system information
- **Target variable**: Total bike rentals per hour (`cnt`)
- **Categorical features**: Season, weather situation, month, hour, weekday, holiday, working day
- **Numerical features**: Temperature, humidity, wind speed, etc.

### Data Preprocessing
- One-hot encoding applied to categorical variables (season, weather, temporal factors)
- Feature expansion from 16 to 59 features after encoding
- Train-test split: 80%-20% with random state for reproducibility
- No missing values or data cleaning required


## How to Use
1. Open `assn8.ipynb` for complete ensemble learning analysis with results
2. Ensure the bike sharing dataset (`hour.csv`) is placed in the `data/` directory as shown above
3. Run cells sequentially for comprehensive ensemble method comparison
4. All hyperparameter optimization is automated using Optuna
5. Expected runtime: 10-15 minutes for complete analysis

## Key Results Summary
- **Best Performing Model**: Stacking Regressor (RMSE: 47.87)
- **Significant Improvement**: 52.3% better than best baseline model
- **Variance Reduction**: Bagging reduces RMSE by 36.7% vs Decision Tree
- **Bias Reduction**: Gradient Boosting reduces RMSE by 51.8% vs Linear Regression  
- **Meta-Learning**: Stacking optimally combines diverse learners for best performance

## Requirements
- Python 3.x  
- pandas, numpy, scikit-learn, matplotlib, seaborn, optuna

Install dependencies with:  
```bash
pip install pandas numpy scikit-learn matplotlib seaborn optuna
```

## Author

*Jerry Jose (BE22B022)*  
*IITM Data Analytics Lab, Semester 7*