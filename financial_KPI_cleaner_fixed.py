import pandas as pd
import numpy as np

def clean_financial_data(input_file: str, output_file: str):
    """
    Cleans financial data from a CSV file using pandas and saves the cleaned data
    to a new CSV file.

    Steps:
    1. Load CSV file
    2. Standardize column names (lowercase, strip spaces, remove suffixes)
    3. Handle missing values (0 for cashflow items, ffill for balance sheet items)
    4. Convert numeric columns to proper dtype
    5. Drop duplicates
    6. Save cleaned dataset

    Args:
        input_file (str): The path to the raw input CSV file.
        output_file (str): The path to save the cleaned CSV file.
    """
    try:
        # 1. Load data
        df = pd.read_csv(input_file)
        print("Data loaded successfully.")

        # 2. Standardize column names
        df.columns = df.columns.str.strip().str.lower()
        df.columns = df.columns.str.replace(r'_x$|_y$', '', regex=True)

        # 3. Handle missing values
        # Fill NaNs in cashflow-like fields with 0, balance sheet with ffill
        df = df.sort_values(["stock", "enddate"])
        df = df.groupby("stock").ffill().fillna(0)

        # 4. Convert numeric columns
        for col in df.columns:
            if col not in ["stock", "enddate"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 5. Drop duplicates
        df = df.drop_duplicates(subset=["stock", "enddate"])

        # 6. Save cleaned dataset
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred during cleaning: {e}")

if __name__ == "__main__":
    input_csv = 'combined_financial_statements.csv'
    output_csv = 'cleaned_financial_data.csv'
    clean_financial_data(input_csv, output_csv)
