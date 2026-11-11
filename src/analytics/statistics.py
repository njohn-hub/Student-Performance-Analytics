import pandas as pd


def descriptive_statistics(data):
    """
    Calculate descriptive statistics for student performance
    
    Returns mean, median, standard deviation, minimum
    maximum and other useful summary statistics
    
    """
    
    columns = [
        column
        for column in ["score", "study_hours", "attendance"]
        if column in data.columns
    ]
    
    if not columns:
        raise ValueError("No supported numerical columns found.")
    
    numerical_data = data[columns].apply(
        pd.to_numeric,
        errors="coerce"
    )
    
    statistics = numerical_data.describe().T
    
    statistics['median'] = numerical_data.median()
    statistics['mode'] = numerical_data.mode().iloc[0] if not numerical_data.mode().empty else float("nan")
    
    return statistics

def grade_distribution(data):
    """
    Calculate the number and percentage of students
    in each examination grade category.
    """

    if "grade" not in data.columns:
        raise ValueError("Dataset must contain a grade column.")

    distribution = (
        data["grade"]
        .fillna("UNKNOWN")
        .value_counts()
        .rename_axis("Grade")
        .reset_index(name="Students")
    )

    total_students = distribution["Students"].sum()

    distribution["Percentage"] = (
        distribution["Students"] / total_students * 100
    ).round(2)

    return distribution

def pass_fail_analysis(data, pass_mark=50):
    """
    Calculate student pass and fail counts.

    A score equal to the pass mark is considered a pass.
    """

    if "score" not in data.columns:
        raise ValueError("Dataset must contain a score column.")

    scores = pd.to_numeric(
        data["score"],
        errors="coerce"
    ).dropna()

    if scores.empty:
        raise ValueError("No valid examination scores available.")

    results = pd.Series(
        ["Pass" if score >= pass_mark else "Fail" for score in scores],
        name="Result"
    )

    summary = (
        results.value_counts()
        .reindex(["Pass", "Fail"], fill_value=0)
        .rename_axis("Result")
        .reset_index(name="Students")
    )

    summary["Percentage"] = (
        summary["Students"] / len(scores) * 100
    ).round(2)

    return summary

def correlation_analysis(data):
    """
    Calculate Pearson correlations between
    examination scores, study hours and attendance.
    """

    columns = [
        column
        for column in ["score", "study_hours", "attendance"]
        if column in data.columns
    ]

    if len(columns) < 2:
        raise ValueError(
            "At least two supported numerical columns are required."
        )

    numerical_data = data[columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    return numerical_data.corr(method="pearson")

def performance_by_grade(data):
    """
    Compare examination performance across
    grade categories.
    """

    required_columns = {"grade", "score"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Dataset must contain grade and score columns."
        )

    working_data = data[["grade", "score"]].copy()

    working_data["score"] = pd.to_numeric(
        working_data["score"],
        errors="coerce"
    )

    working_data = working_data.dropna(
        subset=["grade", "score"]
    )

    if working_data.empty:
        raise ValueError(
            "No valid grade and score records available."
        )

    result = (
        working_data
        .groupby("grade", as_index=False)
        .agg(
            Students=("score", "count"),
            Average_Score=("score", "mean"),
            Highest_Score=("score", "max"),
            Lowest_Score=("score", "min"),
        )
    )

    result["Average_Score"] = result[
        "Average_Score"
    ].round(2)

    return result


def run_statistical_analysis(data, pass_mark=50):
    """
    Execute all statistical analysis functions
    """
    
    return {
        "descriptive": descriptive_statistics(data),
        "grades": grade_distribution(data),
        "pass_fail": pass_fail_analysis(data, pass_mark),
        "correlation": correlation_analysis(data),
        "performance_by_grade": performance_by_grade(data)
        
    } 
        
    


