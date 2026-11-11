
import pandas as pd
import pytest

from src.analytics.statistics import (
    descriptive_statistics,
    grade_distribution,
    pass_fail_analysis,
    correlation_analysis,
    performance_by_grade,
    run_statistical_analysis,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "name": ["Ava", "Liam", "Sophia", "Noah"],
        "grade": ["A", "A", "B", "C"],
        "study_hours": [4.5, 6.0, 3.5, 7.0],
        "attendance": [95, 98, 89, 99],
        "score": [82, 91, 76, 94],
    })


def test_descriptive_statistics(sample_data):
    result = descriptive_statistics(sample_data)

    assert "score" in result.index
    assert result.loc["score", "mean"] == pytest.approx(85.75)
    assert result.loc["score", "median"] == pytest.approx(86.5)


def test_grade_distribution(sample_data):
    result = grade_distribution(sample_data)

    assert result["Students"].sum() == 4
    assert result.loc[result["Grade"] == "A", "Students"].iloc[0] == 2


def test_pass_fail_analysis(sample_data):
    result = pass_fail_analysis(sample_data, pass_mark=80)

    assert result.loc[result["Result"] == "Pass", "Students"].iloc[0] == 3
    assert result.loc[result["Result"] == "Fail", "Students"].iloc[0] == 1


def test_correlation_analysis(sample_data):
    result = correlation_analysis(sample_data)

    assert "score" in result.columns
    assert "study_hours" in result.columns
    assert result.shape == (3, 3)


def test_performance_by_grade(sample_data):
    result = performance_by_grade(sample_data)

    assert len(result) == 3
    assert "Average_Score" in result.columns


def test_run_statistical_analysis(sample_data):
    result = run_statistical_analysis(sample_data)

    assert "descriptive" in result
    assert "grades" in result
    assert "pass_fail" in result
    assert "correlation" in result
    assert "performance_by_grade" in result
