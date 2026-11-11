import pandas as pd
import pytest

from src.analytics.prediction import (
    get_performance_category,
    load_model,
    predict_student_score,
    prepare_student_data,
)


def test_prepare_student_data():
    """Test that student input is converted to a DataFrame."""

    student_data = prepare_student_data(
        age=16,
        gender="Female",
        class_level="Grade 10",
        subject="Mathematics",
        term="Term 1",
        study_hours=5.0,
        attendance=90.0,
        homework_completion=85.0,
        class_participation=4,
        internet_access="Yes",
        extracurricular_activity="Sports",
        parent_education="Secondary",
    )

    assert isinstance(student_data, pd.DataFrame)
    assert len(student_data) == 1

    expected_columns = [
        "age",
        "gender",
        "class_level",
        "subject",
        "term",
        "study_hours",
        "attendance",
        "homework_completion",
        "class_participation",
        "internet_access",
        "extracurricular_activity",
        "parent_education",
    ]

    assert list(student_data.columns) == expected_columns


@pytest.mark.parametrize(
    ("score", "expected_category"),
    [
        (85, "Excellent"),
        (75, "Good"),
        (65, "Satisfactory"),
        (55, "Pass"),
        (40, "Needs Academic Support"),
    ],
)
def test_get_performance_category(score, expected_category):
    """Test performance category classification."""

    result = get_performance_category(score)

    assert result == expected_category


def test_predict_student_score():
    """Test score prediction with a mock model."""

    class MockModel:
        def predict(self, data):
            return [72.5]

    model = MockModel()

    student_data = pd.DataFrame(
        {
            "age": [16],
            "study_hours": [5.0],
        }
    )

    predicted_score = predict_student_score(
        model,
        student_data,
    )

    assert predicted_score == 72.5
    assert isinstance(predicted_score, float)
    


def test_load_model():
    """Test that the trained model can be loaded."""

    model = load_model()

    assert hasattr(model, "predict")