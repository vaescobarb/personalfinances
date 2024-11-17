# personal_finances_utilities
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import calendar

from datetime import datetime

# Formatting excel file 
import os
from openpyxl import load_workbook
from openpyxl.styles import Border, Side, PatternFill
from openpyxl.formatting.rule import CellIsRule

def generate_date_tag():
    return datetime.now().strftime("%Y%m%d")

def load_money_manager_file(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and processes a financial transactions Excel file.

    This function reads an Excel file containing financial transactions, renames the columns from German to English,
    cleans the data by stripping whitespace and converting dates, filters out specific categories, and separates
    the data into expenses and income DataFrames. Each DataFrame has the 'Date' column set as the index and includes
    a 'Month' column extracted from the date.

    Parameters
    ----------
    file_path : str
        The file path to the Excel file to be loaded.

    Returns
    -------
    expenses_df : pandas.DataFrame
        A DataFrame containing processed expense transactions.

    income_df : pandas.DataFrame
        A DataFrame containing processed income transactions.

    Raises
    ------
    FileNotFoundError
        If the Excel file does not exist at the provided file path.

    KeyError
        If expected columns are missing from the Excel file.

    """
    try:
        dataframe = pd.read_excel(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file at {file_path} was not found.") from e
    
    # Define constants for category filters
    EXPENSE_TYPE = 'Ausg.'
    INCOME_TYPE = 'Ein'
    EXCLUDE_EXPENSE_CATEGORY = 'Work Travel'
    EXCLUDE_INCOME_CATEGORY = 'Business travel reimbursement'

    # Rename columns from German to English
    column_mapping = {
        'Zeitraum': 'Date',
        'Konten': 'Account',
        'Kategorie': 'Category',
        'Unterkategorie': 'Subcategory',
        'Notiz': 'Note',
        'Einnahmen/Ausgaben': 'Income/Expense',
        'Beschreibung': 'Description',
        'Betrag': 'Amount',
        'Währung': 'Currency'
    }
    missing_columns = set(column_mapping.keys()) - set(dataframe.columns)
    if missing_columns:
        raise KeyError(f"Missing columns in the Excel file: {missing_columns}")

    dataframe = dataframe.rename(columns=column_mapping)
    dataframe.columns = dataframe.columns.str.strip()

    # Convert the 'Date' column to datetime
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])

    # Strip whitespace from all string columns
    str_columns = dataframe.select_dtypes(include=['object']).columns
    dataframe[str_columns] = dataframe[str_columns].apply(lambda x: x.str.strip())

    # Helper function to process DataFrames
    def process_transactions(df, transaction_type, exclude_category):
        filtered_df = df[
            (df['Income/Expense'] == transaction_type) & (df['Category'] != exclude_category)
        ]
        filtered_df = filtered_df.set_index('Date')
        filtered_df['Month'] = filtered_df.index.to_period('M')
        return filtered_df

    # Process expenses and income
    expenses_df = process_transactions(dataframe, EXPENSE_TYPE, EXCLUDE_EXPENSE_CATEGORY)
    income_df = process_transactions(dataframe, INCOME_TYPE, EXCLUDE_INCOME_CATEGORY)

    expenses_df = expenses_df.reset_index(drop = False)
    income_df = income_df.reset_index(drop = False)

    return expenses_df, income_df