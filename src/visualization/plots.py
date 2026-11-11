from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_visualization():
    """Configure consistent visualization settings"""
    
    sns.set_theme(style="whitegrid", context="notebook")
    
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    

def ensure_output_directory(output_dir):
    """Create the output directory if it does not exist."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path

def plot_score_distribution(data, output_dir):
    """Create and save a distribution plot for student scores"""
    
    setup_visualization()
    output_path = ensure_output_directory(output_dir)
    
    
    scores = pd.to_numeric(
        data["score"],
        errors="coerce"
    ).dropna()
    
    if scores.empty:
        
        raise ValueError("No valid scores available for visualization")
    
    plt.figure()
    
    sns.histplot(
        scores,
        bins=10,
        kde=True
    )
    
    plt.title("Distribution of student scores")
    plt.xlabel("Score")
    plt.ylabel("Number of students")
    
    file_path = output_path / "score_distribution.png"
    
    plt.savefig(file_path)
    plt.close()
    
    return file_path



def plot_performance_by_grade(data, output_dir):
    """Create and save average student performance by grade"""
    
    setup_visualization()
    output_path = ensure_output_directory(output_dir)
    
    required_columns = {"grade", "score"}
    
    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Dataset must contain grade and score columns"
        )
    
    working_data = data[["grade", "score"]].copy()
    
    working_data["score"] = pd.to_numeric(
        working_data["score"],
        errors="coerce"
    )
    
    working_data = working_data.dropna()
    
    average_scores = (
        working_data
        .groupby("grade", as_index=False)["score"]
        .mean()
    )
    
    plt.figure()
    
    sns.barplot(
        data = average_scores,
        x="grade",
        y="score"
    )
    
    plt.title("Average score by grade")
    plt.xlabel("Grade")
    plt.ylabel("Average score")
    
    file_path = output_path / "performance_by_grade.png"
    
    plt.savefig(file_path)
    plt.close()
    
    return file_path


def plot_pass_fail(data, output_dir, pass_mark=50):
    """
    Create a bar chart showing the number of students
    who passed and failed.
    """
    
    output_dir = ensure_output_directory(output_dir)


    # Convert scores to numeric values
    scores = pd.to_numeric(data["score"], errors="coerce").dropna()

    # Classify each student
    results = scores.apply(
        lambda score: "Pass" if score >= pass_mark else "Fail"
    )

    # Create a DataFrame with explicit column names
    result_counts = (
        results.value_counts()
        .rename_axis("Result")
        .reset_index(name="Students")
    )

    # Create the plot
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=result_counts,
        x="Result",
        y="Students"
    )

    plt.title("Pass and Fail Distribution")
    plt.xlabel("Result")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    output_path = output_dir / "pass_fail_distribution.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def plot_study_hours_vs_score(data, output_dir):
    """create and save a study hours versus score scatter plot"""
    
    setup_visualization()
    output_path = ensure_output_directory(output_dir)
    
    required_columns = {"study_hours", "score"}
    
    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Dataset must contain study_hours and score columns"
        )
        
    working_data = data[
        ["study_hours", "score"]
    ].copy()
    
    working_data = working_data.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()
    
    if working_data.empty:
        raise ValueError(
            "No valid study hours and score data available."
        )
        
    plt.figure()
    
    sns.regplot(
        data=working_data,
        x="study_hours",
        y="score"
    )
    
    plt.title("study hours vs student score")
    plt.xlabel("Study hours")
    plt.ylabel("Score")
    
    file_path = output_path / "study_hours_vs_score.png"
    
    plt.savefig(file_path)
    plt.close()
    
    return file_path


def plot_attendance_vs_score(data, output_dir):
    """
    Create a scatter plot showing the relationship between
    student attendance and examination scores.
    """

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check that the required columns exist
    required_columns = ["attendance", "score"]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    # Create a working copy containing only the required columns
    working_data = data[["attendance", "score"]].copy()

    # Convert values to numeric data
    working_data = working_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove rows with missing values
    working_data = working_data.dropna()

    # Check that data is available for visualization
    if working_data.empty:
        raise ValueError(
            "No valid attendance and score data available for visualization."
        )

    # Create the scatter plot
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=working_data,
        x="attendance",
        y="score"
    )

    plt.title("Attendance vs Student Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Score")

    plt.tight_layout()

    # Save the visualization
    output_path = output_dir / "attendance_vs_score.png"

    plt.savefig(output_path, dpi=300)
    plt.close()



def plot_correlation_heatmap(data, output_dir):
    """Create and save a correlation heatmap."""

    setup_visualization()
    output_path = ensure_output_directory(output_dir)

    columns = [
        column
        for column in ["score", "study_hours", "attendance"]
        if column in data.columns
    ]

    if len(columns) < 2:
        raise ValueError(
            "At least two numerical columns are required"
        )

    numerical_data = data[columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    correlation = numerical_data.corr()

    plt.figure()

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        square=True
    )

    plt.title("Student performance correlation heatmap")

    file_path = output_path / "correlation_heatmap.png"

    plt.savefig(file_path)
    plt.close()

    return file_path



def generate_all_visualizations(data, output_dir, pass_mark=50):
    """
    Generate all project visualizations.

    Returns:
        dict: Paths to all generated chart files.
    """
    
    # setup_visualization()
    # ensure_output_directory(output_dir)

    return {
        "score_distribution": plot_score_distribution(
            data, output_dir
        ),
        "performance_by_grade": plot_performance_by_grade(
            data, output_dir
        ),
        "pass_fail": plot_pass_fail(
            data, output_dir, pass_mark
        ),
        "study_hours_vs_score": plot_study_hours_vs_score(
            data, output_dir
        ),
        "attendance_vs_score": plot_attendance_vs_score(
            data, output_dir
        ),
        "correlation_heatmap": plot_correlation_heatmap(
            data, output_dir
        ),
    }



