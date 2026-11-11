
from pathlib import Path

from src.analytics.advanced import run_advanced_analytics
from src.analytics.cleaning import clean_dataset
from src.analytics.exploration import load_dataset
from src.analytics.machine_learning import run_machine_learning
from src.analytics.statistics import run_statistical_analysis
from src.visualization.plots import generate_all_visualizations

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RAW_DATA = BASE_DIR / "data" / "raw" / "students.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

REPORTS_DIR = BASE_DIR / "reports"

VISUALIZATION_DIR = REPORTS_DIR / "visualizations"

FIGURES_DIR = REPORTS_DIR / "figures"


# --------------------------------------------------
# CREATE PROJECT DIRECTORIES
# --------------------------------------------------

def create_directories():
    """Create required output directories."""

    directories = [
        PROCESSED_DIR,
        REPORTS_DIR,
        VISUALIZATION_DIR,
        FIGURES_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_student_data():
    """Load the original student dataset."""

    print("\nLOADING DATASET")
    print("=" * 50)

    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA}"
        )

    data = load_dataset(RAW_DATA)

    if data.empty:
        raise ValueError(
            "The raw dataset is empty."
        )

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {data.shape}")
    print(f"Number of records: {len(data)}")
    print(f"Number of columns: {len(data.columns)}")

    return data


# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

def clean_student_data(data):
    """Clean student records and generate quality reports."""

    print("\nCLEANING DATA")
    print("=" * 50)

    cleaned_data, reports = clean_dataset(data)

    if cleaned_data.empty:
        raise ValueError(
            "The cleaned dataset is empty."
        )

    output_path = PROCESSED_DIR / "students_cleaned.csv"

    cleaned_data.to_csv(
        output_path,
        index=False,
    )

    print(f"\nCleaned dataset saved: {output_path}")
    print(f"Cleaned dataset shape: {cleaned_data.shape}")

    # Export data quality report
    if "quality" in reports:
        reports["quality"].to_csv(
            REPORTS_DIR / "data_quality_report.csv"
        )

    # Export outlier report
    if "outliers" in reports:
        reports["outliers"].to_csv(
            REPORTS_DIR / "outlier_report.csv"
        )

    # Export validation report
    if "validation" in reports:
        reports["validation"].to_csv(
            REPORTS_DIR / "validation_report.csv"
        )

    print("\nDATA QUALITY REPORT")
    print("=" * 50)

    if "quality" in reports:
        print(reports["quality"])

    print("\nOUTLIER REPORT")
    print("=" * 50)

    if "outliers" in reports:
        print(reports["outliers"])

    print("\nVALIDATION REPORT")
    print("=" * 50)

    if "validation" in reports:
        print(reports["validation"])

    return cleaned_data, reports


# --------------------------------------------------
# STATISTICAL ANALYSIS
# --------------------------------------------------

def perform_statistical_analysis(data):
    """Run descriptive and statistical analysis."""

    print("\nSTATISTICAL ANALYSIS")
    print("=" * 50)

    results = run_statistical_analysis(data)

    print("\nStatistical analysis completed.")

    return results


# --------------------------------------------------
# DATA VISUALIZATION
# --------------------------------------------------

def generate_visualizations(data):
    """Generate and save project visualizations."""

    print("\nGENERATING VISUALIZATIONS")
    print("=" * 50)

    generate_all_visualizations(
        data,
        VISUALIZATION_DIR,
    )

    print(
        f"Visualizations saved in: {VISUALIZATION_DIR}"
    )


# --------------------------------------------------
# ADVANCED ANALYTICS
# --------------------------------------------------

def perform_advanced_analysis(data):
    """Run advanced analytics and export reports."""

    print("\nRUNNING ADVANCED ANALYTICS")
    print("=" * 50)

    results = run_advanced_analytics(
        data,
        REPORTS_DIR,
    )

    print("\nAdvanced analytics completed.")

    return results


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    """Execute the complete student analytics pipeline."""

    print("\n")
    print("=" * 60)
    print("STUDENT PERFORMANCE ANALYTICS")
    print("=" * 60)

    try:
        # Step 1: Create output directories
        create_directories()

        # Step 2: Load raw dataset
        raw_data = load_student_data()

        # Step 3: Clean dataset
        cleaned_data, cleaning_reports = clean_student_data(
            raw_data
        )

        # Step 4: Statistical analysis
        statistical_results = perform_statistical_analysis(
            cleaned_data
        )

        # Step 5: Generate visualizations
        generate_visualizations(cleaned_data)

        # Step 6: Advanced analytics
        advanced_results = perform_advanced_analysis(
            cleaned_data
        )
        
        # Step 7: Machine learning
        machine_learning_results = perform_machine_learning(
            cleaned_data
        )

        # Final summary
        print("\n")
        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f"Raw records: {len(raw_data)}")
        print(f"Cleaned records: {len(cleaned_data)}")
        print(
            f"Records removed: "
            f"{len(raw_data) - len(cleaned_data)}"
        )

        print(f"\nProcessed dataset: {PROCESSED_DIR}")
        print(f"Reports: {REPORTS_DIR}")
        print(f"Visualizations: {VISUALIZATION_DIR}")

        print("\nYou can now launch the Streamlit dashboard:")
        print(
            "poetry run streamlit run dashboard/app.py"
        )

        return {
            "raw_data": raw_data,
            "cleaned_data": cleaned_data,
            "cleaning_reports": cleaning_reports,
            "statistical_results": statistical_results,
            "advanced_results": advanced_results,
            "machine_learning_results": machine_learning_results,
        }

    except Exception as error:
        print("\n")
        print("=" * 60)
        print("PIPELINE FAILED")
        print("=" * 60)
        print(f"Error: {error}")

        raise


# --------------------------------------------------
# MACHINE LEARNING
# --------------------------------------------------

def perform_machine_learning(data):
    """Train and evaluate machine learning models"""
    
    print("\nRUNNING MACHINE LEARNING")
    print("=" * 50)
    
    results = run_machine_learning(
        data,
        REPORTS_DIR
    )
    
    print("\nmachine learning completed")
    
    return results




# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
