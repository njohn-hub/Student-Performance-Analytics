
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------


TARGET = "score"

NUMERIC_FEATURES = [
    "age",
    "study_hours",
    "attendance",
    "homework_completion",
    "class_participation",
]

CATEGORICAL_FEATURES = [
    "gender",
    "class_level",
    "subject",
    "term",
    "internet_access",
    "extracurricular_activity",
    "parent_education",
]


RANDOM_STATE= 42
TEST_SIZE=0.20


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

def prepare_features(data):
    """Prepare model features and target variable"""
    
    if TARGET not in data.columns:
        raise ValueError("Dataset must contain a score column")
    
    data = data.copy()
    
    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce"
    )
    
    data = data.dropna(subset=[TARGET])
    
    if data.empty:
        raise ValueError("No valid target scores are available")
    
    numeric_columns = [
        column for column in NUMERIC_FEATURES
        if column in data.columns
    ]
    
    categorical_columns = [
        column for column in CATEGORICAL_FEATURES
        if column in data.columns
    ]
    
    feature_columns = numeric_columns + categorical_columns
    
    if not feature_columns:
        raise ValueError("No supported model features were found")
    
    x = data[feature_columns].copy()
    y = data[TARGET].copy()
    
    for column in numeric_columns:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce"
        )
    
    for column in categorical_columns:
        x[column] = x[column].apply(
            lambda value: (
                str(value) if pd.notna(value) else np.nan
            )
        )
    
    return data, x, y, numeric_columns, categorical_columns



# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------



def split_data(data, x, y):
    """
    
    Split training and training data.
    
    Group by student_id when available so records from 
    the same student do not appear in both sets.
    
    """
    
    if "student_id" in data.columns:
        groups = data["student_id"]
        
        if groups.nunique() < 2:
            raise ValueError(
                "At least two unique student IDs are required"
                "for a group-based split"
            )
            
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
        train_indices, test_indices = next(
            splitter.split(x, y, groups=groups)
        )
        
    else:
        train_indices, test_indices = train_test_split(
            np.arange(len(x)),
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
    
    x_train = x.iloc[train_indices]
    x_test = x.iloc[test_indices]
    
    
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    return (
        x_train,
        x_test,
        y_train,
        y_test,
        test_indices
    )
    

# --------------------------------------------------
# BUILD PREPROCESSING PIPELINE
# --------------------------------------------------


def build_preprocessor(numeric_columns, categorical_columns):
    """Create preprocessing steps for model features"""
    
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )
    
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            )
        ]
    ) 
    
    
    transformers = []
    
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )
        
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )
        
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    

# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------


def evaluate_model(model, x_test, y_test):
    """Calculate regression evaluation metrics"""
    
    predictions = model.predict(x_test)
    
    metrics = {
        "MAE": mean_absolute_error(
            y_test,
            predictions,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            ),
        ),
        "R2": r2_score(
            y_test,
            predictions
        ),
    }
    
    return metrics, predictions



# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

def get_feature_importance(model):
    """Extract random forest feature importance"""
    
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    
    feature_names = preprocessor.get_feature_names_out()
    
    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": regressor.feature_importances_,
        }
    )
    
    return importance.sort_values(
        "Importance",
        ascending=False,
    )



# --------------------------------------------------
# RUN MACHINE LEARNING
# --------------------------------------------------

def run_machine_learning(data, reports_dir):
    """Train, evaluate and save student score prediction models."""
    
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = reports_dir.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nMACHINE LEARNING")
    print("=" * 50)
    
    # Prepare dataset 
    prepared_data, x, y, numeric_columns, categorical_columns = (
        prepare_features(data)
    )  
    
    print(f"Total records: {len(x)}")
    print(f"Numerical features: {numeric_columns}")
    print(f"Categorical features: {categorical_columns}")
    
    # split dataset
    
    (
        x_train,
        X_test,
        y_train,
        y_test,
        test_indices
    ) = split_data(prepared_data, x, y)
    
    print(f"Training records: {len(x_train)}")
    print(f"Testing records: {len(X_test)}")
    
    # preprocessing 
    
    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns
    )
    
    # baseline model 
    baseline_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                DummyRegressor(strategy="mean"),
            ),
        ]
    )
    
    baseline_model.fit(x_train, y_train)
    
    baseline_metrics, _ = evaluate_model(
        baseline_model,
        X_test,
        y_test
    )
    
    # Random forest model 
    
    random_forest = Pipeline(
        steps = [
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns,
                    categorical_columns,
                ),
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    
    
    random_forest.fit(x_train, y_train)
    
    forest_metrics, predictions = evaluate_model(
        random_forest,
        X_test,
        y_test
    )
    
    # compare models 
    
    metrics = pd.DataFrame(
        [
            {
                "Model": "Baseline",
                **baseline_metrics,
            },
            {
                "Model": "Random Forest",
                **forest_metrics,
            },
        ]
    )
    
    metrics = metrics.round(4)
    
    metrics.to_csv(
        reports_dir / "machine_learning_metrics.csv",
        index=False,
    )
    
    # save feature importance 
    
    feature_importance = get_feature_importance(
        random_forest
    )
    
    feature_importance.to_csv(
        reports_dir / "feature_importance.csv",
        index=False,
    )
    
    
    # save predictions 
    test_records = prepared_data.iloc[test_indices]
    
    prediction_data = pd.DataFrame(
        {
            "Actual_Score": y_test.to_numpy(),
            "Predicted_Score": predictions,
        },
    )
    
    prediction_data["Prediction_Error"] = (
        prediction_data["Actual_Score"]
        - prediction_data["Predicted_Score"]
    )
    
    for column in [
        "student_id",
        "name",
        "class_level",
        "subject",
        "term",
    ]:
        if column in test_records.columns:
            prediction_data.insert(
                0,
                column,
                test_records[column].to_numpy(),
            )
            
    prediction_data.to_csv(
        reports_dir / "student_score_predictions.csv",
        index=False
    )
    
    
    # save trained model and feature metadata 
    model_artifact = {
        "model": random_forest,
        "features": x.columns.tolist(),
        "target": TARGET,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
    }
    
    model_path = models_dir / "student_score_model.joblib"
    
    joblib.dump(
        model_artifact,
        model_path,
    )
    
    # display results 
    print("\nMODEL PERFORMANCE")
    print(metrics.to_string(index=False))
    
    print("\nTOP 10 IMPORTANT FEATURES")
    print(feature_importance.head(10).to_string(index=False))
    
    print(f"\nTrained model saved: {model_path}")
    
    print("\nMachine learning completed")
    
    
    return{
        "metrics": metrics,
        "predictions": prediction_data,
        "feature_importance": feature_importance,
        "model_path": model_path,
        "model": random_forest
    }




