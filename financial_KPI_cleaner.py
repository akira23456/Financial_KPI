import pandas as pd
import numpy as np

def clean_financial_data(input_file: str, output_file: str):
    """
    Cleans financial data from a CSV file using pandas and saves the cleaned data
    to a new CSV file.

    Args:
        input_file (str): The path to the raw input CSV file.
        output_file (str): The path to save the cleaned CSV file.
    """
    try:
        # 1. Load the data from the CSV file
        df = pd.read_csv(input_file)
        print("Data loaded successfully.")

        # 2. Handle missing values
        # A common approach is to fill numerical columns with 0 or the mean.
        # This example fills all NaN values with 0.
        df.fillna(0, inplace=True)
        print("Missing values handled.")

        # 3. Rename columns for clarity and consistency with the SQL query.
        # This addresses the 'netIncome_y' column name from the original data.
        if 'netIncome_y' in df.columns:
            df.rename(columns={'netIncome_y': 'netIncome'}, inplace=True)
            print("Renamed 'netIncome_y' to 'netIncome'.")

        # 4. Ensure data types are correct.
        # It's good practice to convert relevant columns to numeric types,
        # in case they were read as strings.
        numeric_cols = [
            'grossProfit', 'totalRevenue', 'operatingIncome', 'netIncome',
            'totalCurrentAssets', 'totalCurrentLiabilities', 'inventory',
            'cash', 'shortTermInvestments', 'totalLiab', 'totalStockholderEquity',
            'ebit', 'interestExpense', 'costOfRevenue', 'netReceivables',
            'totalCashFromOperatingActivities', 'capitalExpenditures'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 5. Save the cleaned DataFrame to a new CSV file.
        df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred during cleaning: {e}")

if __name__ == "__main__":
    # Define your input and output filenames
    input_csv = 'combined_financial_statements.csv'
    output_csv = 'cleaned_financial_data.csv'

    clean_financial_data(input_csv, output_csv)
