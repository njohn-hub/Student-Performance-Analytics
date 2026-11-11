
import numpy as np
import pandas as pd

from src.analytics.cleaning import (
    clean_numeric_columns,
    handle_missing_values,
    normalize_column_names,
    remove_duplicates,
    standardize_categories,
    validate_dataset,
)


def sample_data():

    return pd.DataFrame({
        "name": ["John", "Mary", "Peter", "Jane"],
        "grade": [" a ", "B", "grade c", None],
        "score": [95, 82, 120, 70],
        "study_hours": [8, 5, -2, np.nan]
    })


def test_normalize_column_names():

    data = pd.DataFrame({
        "Student Name": ["John"],
        "Study Hours": [8]
    })

    result = normalize_column_names(data)

    assert "student_name" in result.columns
    assert "study_hours" in result.columns


def test_invalid_scores_become_missing():

    data = sample_data()

    result = clean_numeric_columns(data)

    assert pd.isna(result.loc[2, "score"])


def test_negative_study_hours_become_missing():

    data = sample_data()

    result = clean_numeric_columns(data)

    assert pd.isna(result.loc[2, "study_hours"])


def test_standardize_grades():

    data = sample_data()

    result = standardize_categories(data)

    assert result.loc[0, "grade"] == "A"
    assert result.loc[2, "grade"] == "C"


def test_remove_duplicates():

    data = pd.DataFrame({
        "name": ["John", "John"],
        "grade": ["A", "A"],
        "score": [95, 95],
        "study_hours": [8, 8]
    })

    result = remove_duplicates(data)

    assert len(result) == 1


def test_missing_grade_handling():

    data = sample_data()

    result = handle_missing_values(data)

    assert result["grade"].iloc[3] == "Unknown"


def test_validation_returns_dataframe():

    data = pd.DataFrame({
        "name": ["John"],
        "grade": ["A"],
        "score": [95],
        "study_hours": [8]
    })

    result = validate_dataset(data)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
