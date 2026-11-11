
# Student Performance Analytics

An end-to-end data science and machine learning project for analyzing student academic performance, identifying learning patterns, and predicting student scores through an interactive Streamlit dashboard.

## Project Overview

Student Performance Analytics is a Python-based data science application that transforms student academic records into meaningful insights.

The application combines data cleaning, exploratory data analysis, statistical analysis, data visualization, advanced analytics, and machine learning.

It also provides an interactive dashboard where users can explore academic performance and estimate student scores using a trained Random Forest regression model.

## Key Features

### 1. Data Processing
- Load and inspect student datasets.
- Normalize column names and standardize categorical values.
- Handle missing numerical values.
- Detect duplicate records.
- Identify potential outliers.
- Validate dataset quality.
- Export cleaned datasets and quality reports.

### 2. Exploratory Data Analysis
- Dataset dimensions and structure.
- Numerical and categorical summaries.
- Missing-value analysis.
- Distribution analysis.
- Relationships between student characteristics and academic scores.

### 3. Statistical Analysis
- Descriptive statistics.
- Academic grade distribution.
- Pass/fail analysis.
- Correlation analysis.
- Performance comparisons by academic grade.

### 4. Data Visualization
- Score distribution.
- Academic performance comparisons.
- Pass/fail distribution.
- Study hours versus scores.
- Attendance versus scores.
- Correlation heatmaps.

### 5. Advanced Analytics
- Subject performance analysis.
- Class-level performance analysis.
- Term performance analysis.
- Correlations with academic scores.
- Automated performance insights.
- Student academic support screening.

### 6. Machine Learning
- Regression-based score prediction.
- Baseline model comparison.
- Random Forest regression.
- Model evaluation using MAE, RMSE, and R².
- Feature importance analysis.
- Model persistence using Joblib.

### 7. Interactive Dashboard
- Interactive dataset filters.
- Performance overview and KPIs.
- Academic performance visualizations.
- Learning pattern analysis.
- Student record search.
- Filtered dataset downloads.
- Individual student score predictions.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| Streamlit | Interactive dashboard |
| Joblib | Model serialization |
| Pytest | Automated testing |
| Poetry | Dependency and environment management |

## Project Structure

```text
student-performance-analytics/
│
├── dashboard/
│   ├── __init__.py
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── students.csv
│   └── processed/
│       └── students_cleaned.csv
│
├── models/
│   └── student_score_model.joblib
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── visualizations/
│   ├── data_quality_report.csv
│   ├── outlier_report.csv
│   ├── validation_report.csv
│   ├── correlation_matrix.csv
│   ├── subject_performance.csv
│   ├── class_performance.csv
│   ├── term_performance.csv
│   ├── students_for_support.csv
│   ├── machine_learning_metrics.csv
│   ├── feature_importance.csv
│   └── student_score_predictions.csv
│
├── src/
│   ├── __init__.py
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── exploration.py
│   │   ├── cleaning.py
│   │   ├── statistics.py
│   │   ├── advanced.py
│   │   ├── machine_learning.py
│   │   └── prediction.py
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
│
├── tests/
│   ├── test_cleaning.py
│   ├── test_statistics.py
│   └── test_prediction.py
│
├── main.py
├── pyproject.toml
├── poetry.lock
├── README.md
└── .gitignore
```

## Dataset

The project currently uses a synthetic dataset containing 300 student records and 16 original columns.

The dataset includes:

- Student identification and demographic information.
- Class level and academic subject.
- Academic term and grade.
- Study hours.
- Attendance.
- Homework completion.
- Classroom participation.
- Internet access.
- Extracurricular activities.
- Parent education.
- Academic scores.

The dataset is intended for demonstrating data science and machine learning workflows. It does not represent a real school population.

## Installation

### Prerequisites

- Python compatible with the project dependencies.
- Poetry.
- Git.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd student-performance-analytics
```

### 2. Install dependencies

```bash
poetry install
```

### 3. Activate the project environment

```bash
poetry env activate
```

Alternatively, execute project commands using `poetry run`.

## Running the Project

### 1. Execute the complete analytics pipeline

```bash
poetry run python main.py
```

This runs the data processing, statistical analysis, visualization, advanced analytics, and machine learning workflows.

Generated reports are saved in the `reports/` directory.

The trained model is saved in:

```text
models/student_score_model.joblib
```

### 2. Launch the Streamlit dashboard

```bash
poetry run streamlit run dashboard/app.py
```

Open the local URL displayed in your terminal.

### 3. Run automated tests

```bash
poetry run pytest -v
```

## Machine Learning Results

The project compares a baseline regression model with a Random Forest regression model.

Current evaluation results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 9.8547 | 11.7353 | -0.0286 |
| Random Forest | 7.9177 | 9.7387 | 0.2916 |

### Evaluation Metrics

- **MAE:** Mean Absolute Error. Measures average absolute prediction error.
- **RMSE:** Root Mean Squared Error. Penalizes larger prediction errors.
- **R²:** Coefficient of determination. Measures how much variation in the target is explained by the model.

The Random Forest currently performs better than the baseline on the held-out test set.

These results are preliminary and should not be interpreted as evidence of real-world educational prediction accuracy.

### Important Features

The current Random Forest model identifies the following as its most important features:

1. Study hours.
2. Homework completion.
3. Attendance.
4. Class participation.
5. Term.

Feature importance indicates how the model uses features; it does not establish causation.

## Automated Testing

The project includes tests for:

- Data cleaning.
- Data validation.
- Grade standardization.
- Statistical analysis.
- Student input preparation.
- Performance category classification.
- Prediction functionality.
- Trained model loading.

Latest test result:

```text
21 passed
```

Run the tests using:

```bash
poetry run pytest -v
```

## Limitations and Responsible Use

- The current dataset is synthetic.
- The model has only preliminary predictive performance.
- Model predictions are estimates, not guaranteed academic outcomes.
- Correlation and feature importance do not establish causation.
- Student support screening is a preliminary analytical aid, not a diagnosis.
- Predictions and screening results should never replace professional educational judgment.

## Future Improvements

Potential future development includes:

- Hyperparameter tuning.
- Cross-validation and more robust model evaluation.
- Additional regression algorithms.
- Model performance monitoring.
- Improved dashboard reporting.
- More extensive automated tests.
- Deployment to a cloud hosting platform.
- Evaluation on representative real-world datasets with appropriate privacy safeguards.

## Author

**John Ndungu**

Full Stack Developer | Python Developer | Data Science Enthusiast

GitHub: njohn-hub


## License

MIT License