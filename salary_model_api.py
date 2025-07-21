# Complete Production-Ready Salary Prediction Model Training Pipeline
# Enhanced with Visualization, Improved Regularization, and Web App Integration

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import skew
import joblib
import json
import logging
from datetime import datetime
import os

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedSalaryModelTrainer:
    def __init__(self, data_path=None, output_dir='models'):
        """
        Initialize the Advanced Salary Model Trainer
        
        Args:
            data_path: Path to the CSV file containing salary data
            output_dir: Directory to save models and outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = None
        self.feature_names = None
        self.target_name = 'Salary'
        self.model_metadata = {}
        self.preprocessing_info = {}
        self.visualization_data = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
    def load_and_prepare_data(self, data_path=None):
        """Load and prepare the dataset for training"""
        try:
            if data_path:
                self.data_path = data_path
            elif not self.data_path:
                # Create sample data if no path provided
                logger.info("No data path provided. Creating sample dataset...")
                return self._create_sample_data()
                
            logger.info(f"Loading data from {self.data_path}")
            
            # Load data with error handling
            try:
                data = pd.read_csv(self.data_path)
            except FileNotFoundError:
                # Fallback for different possible file names
                possible_paths = [
                    "Salary_Data.csv",
                    "salary_data.csv", 
                    "data/Salary_Data.csv",
                    "input/Salary_Data.csv"
                ]
                data = None
                for path in possible_paths:
                    try:
                        data = pd.read_csv(path)
                        self.data_path = path
                        logger.info(f"Data loaded from fallback path: {path}")
                        break
                    except FileNotFoundError:
                        continue
                
                if data is None:
                    logger.info("Could not find salary data file. Creating sample dataset...")
                    return self._create_sample_data()
            
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return self._process_loaded_data(data)
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.info("Creating sample dataset as fallback...")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample salary data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample data
        ages = np.random.normal(35, 8, n_samples).clip(22, 65).astype(int)
        experience = np.maximum(0, ages - 22 - np.random.normal(2, 1, n_samples)).clip(0, 40)
        
        genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
        education_levels = np.random.choice([
            "Bachelor's", "Master's", "PhD", "High School"
        ], n_samples, p=[0.5, 0.3, 0.1, 0.1])
        
        job_titles = np.random.choice([
            'Software Engineer', 'Data Scientist', 'Manager', 'Analyst',
            'Developer', 'Consultant', 'Director', 'Senior Engineer'
        ], n_samples, p=[0.25, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.1])
        
        # Create realistic salary based on features
        base_salary = 50000
        salary = base_salary + (experience * 2500) + (ages - 25) * 800
        
        # Education level impact
        edu_multiplier = {'High School': 0.8, "Bachelor's": 1.0, "Master's": 1.3, 'PhD': 1.6}
        salary = salary * np.array([edu_multiplier[edu] for edu in education_levels])
        
        # Job title impact
        job_multiplier = {
            'Software Engineer': 1.4, 'Data Scientist': 1.5, 'Manager': 1.6,
            'Analyst': 1.0, 'Developer': 1.3, 'Consultant': 1.2, 'Director': 2.0,
            'Senior Engineer': 1.7
        }
        salary = salary * np.array([job_multiplier[job] for job in job_titles])
        
        # Add some noise
        salary = salary + np.random.normal(0, 8000, n_samples)
        salary = salary.clip(40000, 300000)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Age': ages,
            'Gender': genders,
            'Education Level': education_levels,
            'Job Title': job_titles,
            'Years of Experience': experience,
            'Salary': salary
        })
        
        logger.info(f"Sample data created. Shape: {data.shape}")
        return self._process_loaded_data(data)
    
    def _process_loaded_data(self, data):
        """Process the loaded data"""
        # Basic data inspection
        logger.info("Data Info:")
        logger.info(f"Columns: {list(data.columns)}")
        logger.info(f"Missing values:\n{data.isnull().sum()}")
        
        # Data cleaning
        original_rows = len(data)
        
        # Drop duplicates
        data.drop_duplicates(inplace=True)
        logger.info(f"Removed {original_rows - len(data)} duplicate rows")
        
        # Drop rows with missing target variable
        data.dropna(subset=[self.target_name], inplace=True)
        logger.info(f"After target cleaning: {data.shape}")
        
        # Handle outliers in salary (remove extreme outliers)
        Q1 = data[self.target_name].quantile(0.05)
        Q3 = data[self.target_name].quantile(0.95)
        data = data[(data[self.target_name] >= Q1) & (data[self.target_name] <= Q3)]
        logger.info(f"After outlier removal: {data.shape}")
        
        # Store preprocessing info
        self.preprocessing_info = {
            'salary_range': [data[self.target_name].min(), data[self.target_name].max()],
            'salary_mean': data[self.target_name].mean(),
            'salary_std': data[self.target_name].std(),
            'salary_median': data[self.target_name].median(),
            'skewness': skew(data[self.target_name]),
            'total_samples': len(data),
            'feature_stats': {}
        }
        
        # Store feature statistics
        for col in data.columns:
            if col != self.target_name:
                if data[col].dtype in ['int64', 'float64']:
                    self.preprocessing_info['feature_stats'][col] = {
                        'type': 'numerical',
                        'mean': data[col].mean(),
                        'std': data[col].std(),
                        'min': data[col].min(),
                        'max': data[col].max()
                    }
                else:
                    self.preprocessing_info['feature_stats'][col] = {
                        'type': 'categorical',
                        'unique_values': list(data[col].unique()),
                        'value_counts': data[col].value_counts().to_dict()
                    }
        
        logger.info(f"Salary statistics - Mean: {self.preprocessing_info['salary_mean']:.2f}, "
                   f"Std: {self.preprocessing_info['salary_std']:.2f}, "
                   f"Skewness: {self.preprocessing_info['skewness']:.3f}")
        
        return data
    
    def create_eda_visualizations(self, data):
        """Create comprehensive EDA visualizations"""
        try:
            logger.info("Creating EDA visualizations...")
            
            # Set up the plotting style
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Salary distribution
            plt.subplot(3, 3, 1)
            plt.hist(data[self.target_name], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Salary Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Salary')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # 2. Salary vs Age
            plt.subplot(3, 3, 2)
            plt.scatter(data['Age'], data[self.target_name], alpha=0.6, color='coral')
            plt.title('Salary vs Age', fontsize=14, fontweight='bold')
            plt.xlabel('Age')
            plt.ylabel('Salary')
            plt.grid(True, alpha=0.3)
            
            # 3. Salary vs Experience
            plt.subplot(3, 3, 3)
            plt.scatter(data['Years of Experience'], data[self.target_name], alpha=0.6, color='lightgreen')
            plt.title('Salary vs Years of Experience', fontsize=14, fontweight='bold')
            plt.xlabel('Years of Experience')
            plt.ylabel('Salary')
            plt.grid(True, alpha=0.3)
            
            # 4. Salary by Gender
            plt.subplot(3, 3, 4)
            gender_salary = data.groupby('Gender')[self.target_name].mean()
            bars = plt.bar(gender_salary.index, gender_salary.values, color=['lightblue', 'pink'])
            plt.title('Average Salary by Gender', fontsize=14, fontweight='bold')
            plt.xlabel('Gender')
            plt.ylabel('Average Salary')
            plt.grid(True, alpha=0.3)
            for bar, value in zip(bars, gender_salary.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                        f'${value:,.0f}', ha='center', va='bottom')
            
            # 5. Salary by Education Level
            plt.subplot(3, 3, 5)
            edu_salary = data.groupby('Education Level')[self.target_name].mean().sort_values(ascending=False)
            bars = plt.bar(edu_salary.index, edu_salary.values, color='lightcoral')
            plt.title('Average Salary by Education Level', fontsize=14, fontweight='bold')
            plt.xlabel('Education Level')
            plt.ylabel('Average Salary')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            for bar, value in zip(bars, edu_salary.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                        f'${value:,.0f}', ha='center', va='bottom', rotation=90)
            
            # 6. Salary by Job Title
            plt.subplot(3, 3, 6)
            job_salary = data.groupby('Job Title')[self.target_name].mean().sort_values(ascending=False)
            bars = plt.bar(range(len(job_salary)), job_salary.values, color='gold')
            plt.title('Average Salary by Job Title', fontsize=14, fontweight='bold')
            plt.xlabel('Job Title')
            plt.ylabel('Average Salary')
            plt.xticks(range(len(job_salary)), job_salary.index, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # 7. Correlation heatmap
            plt.subplot(3, 3, 7)
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            correlation_matrix = data[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            
            # 8. Box plot for salary by education
            plt.subplot(3, 3, 8)
            data.boxplot(column=self.target_name, by='Education Level', ax=plt.gca())
            plt.title('Salary Distribution by Education Level', fontsize=14, fontweight='bold')
            plt.suptitle('')  # Remove automatic title
            plt.xticks(rotation=45)
            
            # 9. Age vs Experience scatter with color by salary
            plt.subplot(3, 3, 9)
            scatter = plt.scatter(data['Age'], data['Years of Experience'], 
                                c=data[self.target_name], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Salary')
            plt.title('Age vs Experience (colored by Salary)', fontsize=14, fontweight='bold')
            plt.xlabel('Age')
            plt.ylabel('Years of Experience')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/visualizations/eda_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Store visualization data for web app
            self.visualization_data['eda'] = {
                'salary_stats': {
                    'mean': data[self.target_name].mean(),
                    'median': data[self.target_name].median(),
                    'std': data[self.target_name].std(),
                    'min': data[self.target_name].min(),
                    'max': data[self.target_name].max()
                },
                'gender_salary': gender_salary.to_dict(),
                'education_salary': edu_salary.to_dict(),
                'job_salary': job_salary.to_dict(),
                'correlation_matrix': correlation_matrix.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error creating EDA visualizations: {str(e)}")
    
    def prepare_features(self, data):
        """Prepare features and target variables"""
        try:
            # Use all available features
            available_features = [col for col in data.columns if col != self.target_name]
            
            # Define feature types based on available columns
            numerical_features = []
            categorical_features = []
            
            for feature in available_features:
                if data[feature].dtype in ['int64', 'float64']:
                    numerical_features.append(feature)
                else:
                    categorical_features.append(feature)
            
            self.feature_names = numerical_features + categorical_features
            
            logger.info(f"Numerical features: {numerical_features}")
            logger.info(f"Categorical features: {categorical_features}")
            
            # Prepare X and y
            X = data[self.feature_names].copy()
            y = data[self.target_name].copy()
            
            return X, y, numerical_features, categorical_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def create_preprocessing_pipeline(self, numerical_features, categorical_features):
        """Create improved preprocessing pipeline with robust scaling"""
        try:
            # Numerical pipeline - using RobustScaler for better outlier handling
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())  # More robust to outliers
            ])
            
            # Categorical pipeline
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))  # Drop first to avoid multicollinearity
            ])
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer([
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ])
            
            return preprocessor
            
        except Exception as e:
            logger.error(f"Error creating preprocessing pipeline: {str(e)}")
            raise
    
    def train_models_with_regularization(self, X, y, numerical_features, categorical_features):
        """Train multiple models with improved regularization to reduce overfitting"""
        try:
            logger.info("Starting model training with regularization...")
            
            # Create preprocessing pipeline
            preprocessor = self.create_preprocessing_pipeline(numerical_features, categorical_features)
            
            # Define models with better regularization
            models = {
                'RandomForest': RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1,
                    min_samples_split=10,  # Increased for regularization
                    min_samples_leaf=5     # Increased for regularization
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    random_state=42,
                    validation_fraction=0.1,  # Early stopping
                    n_iter_no_change=10
                ),
                'ElasticNet': ElasticNet(random_state=42, max_iter=2000),
                'Ridge': Ridge(random_state=42)
            }
            
            # Improved hyperparameter grids with regularization focus
            param_grids = {
                'RandomForest': {
                    'regressor__n_estimators': [100, 200],
                    'regressor__max_depth': [10, 15, None],
                    'regressor__min_samples_split': [10, 20],
                    'regressor__min_samples_leaf': [5, 10],
                    'regressor__max_features': ['sqrt', 0.8]
                },
                'GradientBoosting': {
                    'regressor__n_estimators': [100, 150],
                    'regressor__learning_rate': [0.05, 0.1],
                    'regressor__max_depth': [3, 5],
                    'regressor__subsample': [0.8, 0.9],
                    'regressor__min_samples_split': [10, 20]
                },
                'ElasticNet': {
                    'regressor__alpha': [0.1, 1.0, 10.0],
                    'regressor__l1_ratio': [0.1, 0.5, 0.7, 0.9]
                },
                'Ridge': {
                    'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
                }
            }
            
            # Split data with stratified sampling for better generalization
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Further split training data for validation
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            best_model = None
            best_score = float('-inf')
            best_model_name = None
            model_results = {}
            
            # Train and evaluate each model
            for model_name, model in models.items():
                logger.info(f"Training {model_name}...")
                
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])
                
                # Grid search with more robust cross-validation
                grid_search = GridSearchCV(
                    pipeline, 
                    param_grids[model_name], 
                    cv=5, 
                    scoring='r2', 
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit model
                grid_search.fit(X_train_sub, y_train_sub)
                
                # Get best model
                current_model = grid_search.best_estimator_
                
                # Make predictions on all sets
                y_train_pred = current_model.predict(X_train_sub)
                y_val_pred = current_model.predict(X_val)
                y_test_pred = current_model.predict(X_test)
                
                # Calculate comprehensive metrics
                train_r2 = r2_score(y_train_sub, y_train_pred)
                val_r2 = r2_score(y_val, y_val_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                train_rmse = mean_squared_error(y_train_sub, y_train_pred, squared=False)
                val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
                test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
                
                train_mae = mean_absolute_error(y_train_sub, y_train_pred)
                val_mae = mean_absolute_error(y_val, y_val_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(current_model, X_train, y_train, cv=5, scoring='r2')
                
                # Calculate overfitting metrics
                train_val_gap = abs(train_r2 - val_r2)
                train_test_gap = abs(train_r2 - test_r2)
                generalization_score = min(val_r2, test_r2)
                
                # Store results
                model_results[model_name] = {
                    'best_params': grid_search.best_params_,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'val_mae': val_mae,
                    'test_mae': test_mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'train_val_gap': train_val_gap,
                    'train_test_gap': train_test_gap,
                    'generalization_score': generalization_score
                }
                
                logger.info(f"{model_name} - Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}, "
                           f"CV R²: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
                # Select best model based on generalization performance and low overfitting
                model_score = (generalization_score * 0.8) - (train_val_gap * 0.2)
                if model_score > best_score:
                    best_score = model_score
                    best_model = current_model
                    best_model_name = model_name
            
            logger.info(f"Best model: {best_model_name}")
            
            # Final evaluation on best model
            y_train_pred_final = best_model.predict(X_train)
            y_test_pred_final = best_model.predict(X_test)
            
            # Store model metadata
            self.model_metadata = {
                'model_name': best_model_name,
                'training_date': datetime.now().isoformat(),
                'feature_names': self.feature_names,
                'numerical_features': numerical_features,
                'categorical_features': categorical_features,
                'model_results': model_results,
                'final_metrics': {
                    'train_r2': r2_score(y_train, y_train_pred_final),
                    'test_r2': r2_score(y_test, y_test_pred_final),
                    'train_rmse': mean_squared_error(y_train, y_train_pred_final, squared=False),
                    'test_rmse': mean_squared_error(y_test, y_test_pred_final, squared=False),
                    'train_mae': mean_absolute_error(y_train, y_train_pred_final),
                    'test_mae': mean_absolute_error(y_test, y_test_pred_final),
                    'generalization_gap': abs(r2_score(y_train, y_train_pred_final) - r2_score(y_test, y_test_pred_final))
                },
                'preprocessing_info': self.preprocessing_info
            }
            
            self.model = best_model
            
            # Create model performance visualizations
            self._create_model_performance_plots(X_test, y_test, y_test_pred_final, model_results)
            
            # Print final results
            print("\n" + "="*60)
            print("FINAL MODEL PERFORMANCE (IMPROVED)")
            print("="*60)
            print(f"Best Model: {best_model_name}")
            print(f"Best Parameters: {model_results[best_model_name]['best_params']}")
            print(f"Train R²: {self.model_metadata['final_metrics']['train_r2']:.4f}")
            print(f"Test R²: {self.model_metadata['final_metrics']['test_r2']:.4f}")
            print(f"Generalization Gap: {self.model_metadata['final_metrics']['generalization_gap']:.4f}")
            print(f"Train RMSE: {self.model_metadata['final_metrics']['train_rmse']:.2f}")
            print(f"Test RMSE: {self.model_metadata['final_metrics']['test_rmse']:.2f}")
            print(f"Train MAE: {self.model_metadata['final_metrics']['train_mae']:.2f}")
            print(f"Test MAE: {self.model_metadata['final_metrics']['test_mae']:.2f}")
            print("="*60)
            
            return X_train, X_test, y_train, y_test, y_test_pred_final
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def _create_model_performance_plots(self, X_test, y_test, y_pred, model_results):
        """Create comprehensive model performance visualizations"""
        try:
            # Set up the figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Actual vs Predicted scatter plot
            axes[0,0].scatter(y_test, y_pred, alpha=0.6, color='blue')
            axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0,0].set_xlabel('Actual Salary')
            axes[0,0].set_ylabel('Predicted Salary')
            axes[0,0].set_title('Actual vs Predicted Salary')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add R² score to plot
            r2 = r2_score(y_test, y_pred)
            axes[0,0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0,0].transAxes,
                          bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
            # 2. Residuals plot
            residuals = y_test - y_pred
            axes[0,1].scatter(y_pred, residuals, alpha=0.6, color='green')
            axes[0,1].axhline(y=0, color='r', linestyle='--')
            axes[0,1].set_xlabel('Predicted Salary')
            axes[0,1].set_ylabel('Residuals')
            axes[0,1].set_title('Residual Plot')
            axes[0,1].grid(True, alpha=0.3)
            
            # 3. Model comparison
            models = list(model_results.keys())
            test_r2_scores = [model_results[model]['test_r2'] for model in models]
            val_r2_scores = [model_results[model]['val_r2'] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[0,2].bar(x - width/2, test_r2_scores, width, label='Test R²', alpha=0.8)
            axes[0,2].bar(x + width/2, val_r2_scores, width, label='Validation R²', alpha=0.8)
            axes[0,2].set_xlabel('Models')
            axes[0,2].set_ylabel('R² Score')
            axes[0,2].set_title('Model Performance Comparison')
            axes[0,2].set_xticks(x)
            axes[0,2].set_xticklabels(models, rotation=45)
            axes[0,2].legend()
            axes[0,2].grid(True, alpha=0.3)
            
            # 4. Residuals histogram
            axes[1,0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1,0].set_xlabel('Residuals')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Distribution of Residuals')
            axes[1,0].grid(True, alpha=0.3)
            
            # Add normal curve
            mu, sigma = residuals.mean(), residuals.std()
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            y_norm = ((1 / (np.sqrt(2 * np.pi) * sigma)) * 
                     np.exp(-0.5 * (x_norm - mu) ** 2 / sigma ** 2)) * len(residuals) * (residuals.max() - residuals.min()) / 30
            axes[1,0].plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal Curve')
            axes[1,0].legend()
            
            # 5. Feature importance (if available)
            if hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
                feature_importance = self.get_feature_importance()
                if feature_importance and len(feature_importance) > 0:
                    # Get top 10 features
                    top_features = list(feature_importance.items())[:10]
                    features, importance = zip(*top_features)
                    
                    axes[1,1].barh(range(len(features)), importance, color='purple', alpha=0.7)
                    axes[1,1].set_yticks(range(len(features)))
                    axes[1,1].set_yticklabels(features)
                    axes[1,1].set_xlabel('Importance')
                    axes[1,1].set_title('Top 10 Feature Importance')
                    axes[1,1].grid(True, alpha=0.3)
                else:
                    axes[1,1].text(0.5, 0.5, 'Feature importance\nnot available', 
                                  ha='center', va='center', transform=axes[1,1].transAxes)
                    axes[1,1].set_title('Feature Importance')
            else:
                axes[1,1].text(0.5, 0.5, 'Feature importance\nnot available', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Feature Importance')
            
            # 6. Prediction error distribution
            prediction_errors = np.abs(residuals)
            axes[1,2].hist(prediction_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
            axes[1,2].set_xlabel('Absolute Prediction Error')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].set_title('Prediction Error Distribution')
            axes[1,2].grid(True, alpha=0.3)
            
            # Add statistics
            mae = np.mean(prediction_errors)
            axes[1,2].axvline(mae, color='black', linestyle='--', linewidth=2, label=f'MAE: ${mae:,.0f}')
            axes[1,2].legend()
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/visualizations/model_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating performance plots: {str(e)}")
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        try:
            if not hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
                return None
                
            # Get feature names after preprocessing
            preprocessor = self.model.named_steps['preprocessor']
            
            # Get feature names from preprocessor
            feature_names_out = []
            
            # Numerical features
            if 'num' in preprocessor.named_transformers_:
                num_features = self.model_metadata['numerical_features']
                feature_names_out.extend(num_features)
            
            # Categorical features (after one-hot encoding)
            if 'cat' in preprocessor.named_transformers_:
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                if hasattr(cat_encoder, 'get_feature_names_out'):
                    cat_feature_names = cat_encoder.get_feature_names_out(self.model_metadata['categorical_features'])
                    feature_names_out.extend(cat_feature_names)
            
            # Get importance scores
            importance_scores = self.model.named_steps['regressor'].feature_importances_
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names_out, importance_scores))
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            logger.info("Top 10 Feature Importances:")
            for i, (feature, importance) in enumerate(list(sorted_importance.items())[:10]):
                logger.info(f"{i+1}. {feature}: {importance:.4f}")
            
            return sorted_importance
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None
    
    def save_model(self):
        """Save the trained model and metadata"""
        try:
            # Save model
            model_path = os.path.join(self.output_dir, 'salary_prediction_model.joblib')
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save metadata
            metadata_path = os.path.join(self.output_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=2, default=str)
            logger.info(f"Metadata saved to {metadata_path}")
            
            # Get and save feature importance
            feature_importance = self.get_feature_importance()
            if feature_importance:
                importance_path = os.path.join(self.output_dir, 'feature_importance.json')
                with open(importance_path, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                logger.info(f"Feature importance saved to {importance_path}")
            
            # Save visualization data
            viz_path = os.path.join(self.output_dir, 'visualization_data.json')
            with open(viz_path, 'w') as f:
                json.dump(self.visualization_data, f, indent=2, default=str)
            logger.info(f"Visualization data saved to {viz_path}")
            
            return model_path, metadata_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def create_learning_curves(self, X, y):
        """Create learning curves to analyze model performance vs training size"""
        try:
            from sklearn.model_selection import learning_curve
            
            logger.info("Creating learning curves...")
            
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.model, X, y, cv=5, n_jobs=-1, 
                train_sizes=train_sizes, scoring='r2'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            plt.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('R² Score')
            plt.title('Learning Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{self.output_dir}/visualizations/learning_curves.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating learning curves: {str(e)}")
    
    def train_complete_pipeline(self, data_path=None):
        """Complete training pipeline with all improvements"""
        try:
            # Load and prepare data
            data = self.load_and_prepare_data(data_path)
            
            # Create EDA visualizations
            self.create_eda_visualizations(data)
            
            # Prepare features
            X, y, numerical_features, categorical_features = self.prepare_features(data)
            
            # Train models with regularization
            X_train, X_test, y_train, y_test, y_pred = self.train_models_with_regularization(
                X, y, numerical_features, categorical_features
            )
            
            # Create learning curves
            self.create_learning_curves(X, y)
            
            # Save model
            model_path, metadata_path = self.save_model()
            
            logger.info("Training completed successfully!")
            
            return {
                'model_path': model_path,
                'metadata_path': metadata_path,
                'model_metadata': self.model_metadata
            }
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise

# Enhanced Web App Integration Functions
class SalaryPredictionAPI:
    def __init__(self, model_path='models/salary_prediction_model.joblib', 
                 metadata_path='models/model_metadata.json'):
        """Initialize the prediction API"""
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        """Load trained model for web app usage"""
        try:
            self.model = joblib.load(self.model_path)
            
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            logger.info("Model loaded successfully for API usage")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def validate_input(self, data):
        """Enhanced input validation"""
        try:
            required_features = self.metadata['feature_names']
            
            # Check if all required features are present
            missing_features = [f for f in required_features if f not in data.keys()]
            if missing_features:
                return False, f"Missing features: {missing_features}"
            
            # Validate data types and ranges
            numerical_features = self.metadata['numerical_features']
            categorical_features = self.metadata['categorical_features']
            
            # Validate numerical features
            for feature in numerical_features:
                try:
                    value = float(data[feature])
                    
                    if feature == 'Age':
                        if value < 18 or value > 70:
                            return False, f"Age must be between 18 and 70, got {value}"
                    elif feature == 'Years of Experience':
                        if value < 0 or value > 50:
                            return False, f"Years of Experience must be between 0 and 50, got {value}"
                        # Check logical consistency
                        if 'Age' in data:
                            age = float(data['Age'])
                            if value > (age - 16):  # Assuming minimum working age is 16
                                return False, f"Years of Experience ({value}) cannot exceed Age minus 16 ({age - 16})"
                    
                except (ValueError, TypeError):
                    return False, f"{feature} must be a valid number, got {data[feature]}"
            
            # Validate categorical features
            for feature in categorical_features:
                feature_stats = self.metadata['preprocessing_info']['feature_stats'].get(feature, {})
                if 'unique_values' in feature_stats:
                    valid_values = feature_stats['unique_values']
                    if data[feature] not in valid_values:
                        return False, f"{feature} must be one of {valid_values}, got {data[feature]}"
            
            return True, "Valid input"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def predict_salary(self, data):
        """Make salary prediction with confidence interval"""
        try:
            # Validate input
            is_valid, message = self.validate_input(data)
            if not is_valid:
                return {
                    'success': False,
                    'error': message,
                    'prediction': None
                }
            
            # Prepare input data
            input_df = pd.DataFrame([data])
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            
            # Calculate prediction interval (approximate using residuals if available)
            prediction_std = self.metadata['preprocessing_info'].get('salary_std', 15000)
            margin_of_error = prediction_std * 0.2  # Approximate 20% margin
            
            lower_bound = max(0, prediction - margin_of_error)
            upper_bound = prediction + margin_of_error
            
            # Round to nearest dollar
            prediction = round(prediction, 2)
            lower_bound = round(lower_bound, 2)
            upper_bound = round(upper_bound, 2)
            
            # Determine confidence level based on model performance
            model_r2 = self.metadata['final_metrics']['test_r2']
            confidence = "High" if model_r2 > 0.9 else "Medium" if model_r2 > 0.8 else "Low"
            
            return {
                'success': True,
                'prediction': prediction,
                'prediction_range': {
                    'lower': lower_bound,
                    'upper': upper_bound
                },
                'confidence': confidence,
                'model_accuracy': f"{model_r2:.1%}",
                'explanation': self._generate_prediction_explanation(data, prediction)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'success': False,
                'error': f"Prediction error: {str(e)}",
                'prediction': None
            }
    
    def _generate_prediction_explanation(self, data, prediction):
        """Generate explanation for the prediction"""
        try:
            explanations = []
            
            # Experience factor
            if 'Years of Experience' in data:
                exp = data['Years of Experience']
                if exp < 2:
                    explanations.append("Low experience level contributes to lower salary range")
                elif exp > 10:
                    explanations.append("High experience level contributes to higher salary range")
            
            # Education factor
            if 'Education Level' in data:
                edu = data['Education Level']
                if edu in ['PhD', "Master's"]:
                    explanations.append(f"{edu} degree typically commands higher compensation")
                elif edu == 'High School':
                    explanations.append("High school education may limit salary potential")
            
            # Age factor
            if 'Age' in data:
                age = data['Age']
                if age > 45:
                    explanations.append("Senior age typically correlates with higher salaries")
                elif age < 25:
                    explanations.append("Younger age typically means entry-level salaries")
            
            # Job title factor
            if 'Job Title' in data:
                job = data['Job Title']
                high_paying_roles = ['Director', 'Senior Engineer', 'Data Scientist', 'Manager']
                if any(role in job for role in high_paying_roles):
                    explanations.append(f"{job} is typically a well-compensated role")
            
            return explanations if explanations else ["Prediction based on comprehensive analysis of all provided factors"]
            
        except Exception as e:
            return ["Prediction based on trained model analysis"]
    
    def get_feature_impact(self, data):
        """Get the impact of each feature on the prediction"""
        try:
            if not hasattr(self.model.named_steps['regressor'], 'feature_importances_'):
                return None
            
            # Get feature importance
            feature_importance_path = self.model_path.replace('salary_prediction_model.joblib', 'feature_importance.json')
            
            try:
                with open(feature_importance_path, 'r') as f:
                    feature_importance = json.load(f)
            except FileNotFoundError:
                return None
            
            # Map user input to feature importance
            impact_analysis = {}
            for feature, value in data.items():
                if feature in self.metadata['numerical_features']:
                    if feature in feature_importance:
                        impact_analysis[feature] = {
                            'value': value,
                            'importance': feature_importance[feature],
                            'type': 'numerical'
                        }
                else:
                    # For categorical features, find the encoded version
                    for imp_feature, importance in feature_importance.items():
                        if feature.replace(' ', '') in imp_feature.replace('_', '').replace(' ', '') and str(value).replace(' ', '') in imp_feature.replace('_', '').replace(' ', ''):
                            impact_analysis[feature] = {
                                'value': value,
                                'importance': importance,
                                'type': 'categorical',
                                'encoded_feature': imp_feature
                            }
                            break
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Error getting feature impact: {str(e)}")
            return None
    
    def get_salary_statistics(self):
        """Get salary statistics for comparison"""
        try:
            stats = self.metadata['preprocessing_info']
            return {
                'mean_salary': stats['salary_mean'],
                'median_salary': stats.get('salary_median', stats['salary_mean']),
                'min_salary': stats['salary_range'][0],
                'max_salary': stats['salary_range'][1],
                'std_salary': stats['salary_std']
            }
        except Exception as e:
            logger.error(f"Error getting salary statistics: {str(e)}")
            return None
    
    def health_check(self):
        """Check if the API is ready"""
        try:
            if self.model is None or self.metadata is None:
                return {
                    'status': 'unhealthy',
                    'message': 'Model not loaded'
                }
            
            # Test with sample data
            sample_data = {
                'Age': 30,
                'Gender': 'Male',
                'Education Level': "Bachelor's",
                'Job Title': 'Software Engineer',
                'Years of Experience': 5
            }
            
            # Adjust sample data based on available features
            available_sample = {}
            for feature in self.metadata['feature_names']:
                if feature in sample_data:
                    available_sample[feature] = sample_data[feature]
                elif feature in self.metadata['preprocessing_info']['feature_stats']:
                    stats = self.metadata['preprocessing_info']['feature_stats'][feature]
                    if stats['type'] == 'numerical':
                        available_sample[feature] = stats['mean']
                    else:
                        available_sample[feature] = stats['unique_values'][0]
            
            result = self.predict_salary(available_sample)
            
            if result['success']:
                return {
                    'status': 'healthy',
                    'message': 'API is ready',
                    'model_info': {
                        'name': self.metadata['model_name'],
                        'accuracy': self.metadata['final_metrics']['test_r2'],
                        'features': len(self.metadata['feature_names'])
                    }
                }
            else:
                return {
                    'status': 'unhealthy',
                    'message': f'Model test failed: {result["error"]}'
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Health check failed: {str(e)}'
            }

# Main execution and demonstration
if __name__ == "__main__":
    # Create and train the model
    print("="*60)
    print("ADVANCED SALARY PREDICTION MODEL TRAINING")
    print("="*60)
    
    trainer = AdvancedSalaryModelTrainer()
    
    try:
        # Train the complete pipeline (will use sample data if no file provided)
        results = trainer.train_complete_pipeline()
        
        print(f"\n✅ Model training completed successfully!")
        print(f"📁 Model saved at: {results['model_path']}")
        print(f"📊 Metadata saved at: {results['metadata_path']}")
        print(f"🎯 Model Accuracy: {results['model_metadata']['final_metrics']['test_r2']:.1%}")
        print(f"📉 Generalization Gap: {results['model_metadata']['final_metrics']['generalization_gap']:.4f}")
        
        # Test the API
        print("\n" + "="*60)
        print("TESTING WEB APP API INTEGRATION")
        print("="*60)
        
        api = SalaryPredictionAPI()
        
        # Health check
        health = api.health_check()
        print(f"🏥 API Health: {health['status'].upper()}")
        print(f"📋 Message: {health['message']}")
        
        if health['status'] == 'healthy':
            # Test predictions with various scenarios
            test_cases = [
                {
                    'name': 'Senior Software Engineer',
                    'data': {
                        'Age': 35,
                        'Gender': 'Male',
                        'Education Level': "Master's",
                        'Job Title': 'Software Engineer',
                        'Years of Experience': 10
                    }
                },
                {
                    'name': 'Junior Data Scientist',
                    'data': {
                        'Age': 26,
                        'Gender': 'Female',
                        'Education Level': "Bachelor's",
                        'Job Title': 'Data Scientist',
                        'Years of Experience': 2
                    }
                },
                {
                    'name': 'Experienced Manager',
                    'data': {
                        'Age': 45,
                        'Gender': 'Male',
                        'Education Level': "Master's",
                        'Job Title': 'Manager',
                        'Years of Experience': 20
                    }
                }
            ]
            
            print("\n🧪 TESTING PREDICTION SCENARIOS:")
            print("-" * 60)
            
            for i, case in enumerate(test_cases, 1):
                print(f"\n{i}. {case['name']}:")
                result = api.predict_salary(case['data'])
                
                if result['success']:
                    print(f"   💰 Predicted Salary: ${result['prediction']:,.2f}")
                    print(f"   📊 Range: ${result['prediction_range']['lower']:,.0f} - ${result['prediction_range']['upper']:,.0f}")
                    print(f"   🎯 Confidence: {result['confidence']}")
                    print(f"   📈 Model Accuracy: {result['model_accuracy']}")
                    
                    # Show explanations
                    if result['explanation']:
                        print("   💡 Key Factors:")
                        for explanation in result['explanation']:
                            print(f"      • {explanation}")
                    
                    # Show feature impact if available
                    impact = api.get_feature_impact(case['data'])
                    if impact:
                        print("   📊 Feature Impact (top 3):")
                        sorted_impact = sorted(impact.items(), key=lambda x: x[1]['importance'], reverse=True)[:3]
                        for feature, info in sorted_impact:
                            print(f"      • {feature}: {info['importance']:.3f}")
                else:
                    print(f"   ❌ Prediction failed: {result['error']}")
            
            # Show salary statistics
            stats = api.get_salary_statistics()
            if stats:
                print(f"\n📊 DATASET SALARY STATISTICS:")
                print("-" * 40)
                print(f"Average Salary: ${stats['mean_salary']:,.2f}")
                print(f"Median Salary: ${stats['median_salary']:,.2f}")
                print(f"Salary Range: ${stats['min_salary']:,.0f} - ${stats['max_salary']:,.0f}")
                print(f"Standard Deviation: ${stats['std_salary']:,.0f}")
        
        print(f"\n🎉 SETUP COMPLETE! Your salary prediction model is ready for web deployment!")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("💡 The system will still work with sample data for demonstration purposes.")