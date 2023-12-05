import pandas as pd
import argparse
import sys

def load_data(filepath):
    """
    Load and preprocess the dataset from an Excel file.

    This function reads an Excel file into a pandas DataFrame and performs
    necessary data type conversions and other preprocessing steps. It converts
    'year', 'day', and 'Holiday' columns to string type, and converts 'tourist',
    'Trend', 'Seasonal', and 'Resid' columns to floating-point numbers.

    Args:
        filepath (str): Path to the data file.

    Returns:
        DataFrame: A pandas DataFrame with preprocessed data.

    Raises:
        FileNotFoundError: If the file at 'filepath' is not found.
        Exception: If any other error occurs during file loading.
    """
    try:
        data = pd.read_excel(filepath)
        
        # Check for missing values in 'tourist' column
        if data['tourist'].isnull().any():
            print("Missing values found in 'tourist' column. Applying KNN imputation.")
            imputer = KNNImputer(n_neighbors=5)  # Adjust 'n_neighbors' as needed
            data['tourist'] = imputer.fit_transform(data[['tourist']]).ravel()

        # Convert columns to appropriate data types
        data["year"] = data["year"].astype(str)
        data["day"] = data["day"].astype(str)
        if 'Holiday' in data.columns:
            data['Holiday'] = data['Holiday'].astype(str)
        if 'pc_Jiuzhaigou' in data.columns:
            data["pc_Jiuzhaigou"] = data["pc_Jiuzhaigou"].astype("float64")
        if 'mob_Jiuzhaigou' in data.columns:
            data["mob_Jiuzhaigou"] = data["mob_Jiuzhaigou"].astype("float64")
        if 'pc_SichuanEpidemic' in data.columns:    
            data["pc_SichuanEpidemic"] = data["pc_SichuanEpidemic"].astype("float64")
        if 'mob_SichuanEpidemic' in data.columns:    
            data["mob_SichuanEpidemic"] = data["mob_SichuanEpidemic"].astype("float64")
        if 'pc_Jiuzhaigou' in data.columns:
            data["pc_Siguniang"] = data["pc_Siguniang"].astype("float64")
        if 'pc_Jiuzhaigou' in data.columns:
            data["mob_Siguniang"] = data["mob_Siguniang"].astype("float64")
        data["tourist"] = data["tourist"].astype("float64")
        data["Trend"]=data["Trend"].astype("float64")
        data["Seasonal"]=data["Seasonal"].astype("float64")
        data["Resid"]=data["Resid"].astype("float64")
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up argument parser for command line functionality
    parser = argparse.ArgumentParser(description='Load and preprocess data for tourism demand forecasting.')
    parser.add_argument('filepath', type=str, help='Path to the data file.')

    # Parse command line arguments
    args = parser.parse_args()

    # Load and print first few rows of the data
    data = load_data(args.filepath)
    print(data.head())  # Display the first few rows to confirm successful loading
