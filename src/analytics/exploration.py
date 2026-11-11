import pandas as pd


def load_dataset(file_path):
    """
    Load student dataset from CSV.
    """
    try:
        data = pd.read_csv(file_path, sep="\t")
        print("Dataset loaded successfully")
        return data

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

def dataset_overview(data):
    """
    Display general information about the dataset
    """

    print("\nDataset overview")
    print("=" * 40)

    print(f"Number of students: {len(data)}")
    print(f"Number of columns: {len(data.columns)}")

    print("\nColumn names: ")
    print(data.columns.tolist())

    print("\nDataset information: ")
    data.info()

    print("\nFirst five records")
    print(data.head())


def missing_values(data):
    """calculate missing values in each column"""

    missing = data.isnull().sum()

    missing_percentage = (data.isnull().mean() * 100).round(2)

    result = pd.DataFrame(
        {"Missing values": missing, "missing percentage": missing_percentage}
    )

    print("\nMissing values")
    print("=" * 40)

    print(result)

    return result


def duplicate_records(data):
    """identify duplicate records"""

    duplicates = data.duplicated().sum()

    print("\nDuplicate records")
    print("=" * 40)

    print(f"Total duplicates: {duplicates}")

    return duplicates
