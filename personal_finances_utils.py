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


def classify_expenses(df):
    # Define your category mapping dictionary
    category_map = {
        'Food': 'NEED',
        'Miete': 'NEED',
        'Transportation': 'NEED',
        'Haircut': 'NEED',
        'Health': 'NEED',
        'Cell phone': 'NEED',
        'Paperwork': 'NEED',
        'Entgeltabschluss': 'NEED',
        'Laundry':'NEED',
        'Social Life': 'WANT',
        'Travel': 'WANT',
        'Subscription': 'WANT',
        'Apparel': 'WANT',
        'Transfer family': 'WANT',
        'Gift': 'WANT',
        'Self-development': 'WANT',
        'Household': 'WANT',
        'Culture': 'WANT',
        'Tech': 'WANT',
        'Education': 'WANT',
        'Savings': 'SAVE',
        'Investments': 'SAVE',
        'Bestand ändern':'UNKNOWN',
        'Other':'UNKNOWN'
    }
    
    # Create the new column by mapping the Category column using the dictionary
    df = df.assign(column_name = 'Type')
    df['Type'] = df['Category'].map(category_map)
    
    # Handle any unmapped categories, if necessary
    df['Type'] = df['Type'].fillna('UNKNOWN')
    
    return df

def plot_monthly_expenses(df, income_df):
    # Ensure 'Month' is a datetime column for easier grouping
    df.loc[:,'Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
    income_df.loc[:,'Month'] = pd.to_datetime(income_df['Date']).dt.to_period('M')
    
    # Group by 'Month' and 'Type' in expenses and sum 'EUR' column
    monthly_expenses = df.groupby(['Month', 'Type'])['EUR'].sum().unstack(fill_value=0)

    # Summarize total monthly expenses across all types
    total_monthly_expenses = monthly_expenses.sum(axis=1)

    # Group by 'Month' in income and sum 'EUR' column to get total monthly income
    monthly_income = income_df.groupby('Month')['EUR'].sum()

    # Calculate the SAVE value by subtracting total expenses from total income per month
    monthly_savings = monthly_income - total_monthly_expenses
    monthly_savings = monthly_savings[monthly_savings > 0]
    monthly_savings.name = 'SAVE'  # Name the Series as 'SAVE' for easy stacking

    # Add the SAVE column to the expenses DataFrame
    monthly_expenses['SAVE'] = monthly_savings
    

    # Define colors for each 'Type'
    colors = ['#D62728', '#66B2FF', '#FFBF00', '#228B22']  # NEED, UNKNOWN, WANT, SAVE 
    ax = monthly_expenses.plot(kind='bar', stacked=True, color=colors, figsize=(16, 8))

    # Calculate global total for each type (category)
    category_totals = monthly_expenses.sum(axis=0)

    # Add annotations for each global total
    for i, (expense_type, total) in enumerate(category_totals.items()):
        ax.annotate(
            f'Total {expense_type}: {total:.2f}',      # Annotation text with the total
            xy=(0.1, 1.02 - i * 0.05),                  # Position above the plot, staggered for visibility
            xycoords='axes fraction',
            ha='left',
            va='center',
            color=colors[i],                            # Use the same color as the bar
            fontweight='bold',
            fontsize=12,
            backgroundcolor="white"                     # Optional: white background for better visibility
        )

    # Adding labels and title
    plt.xlabel('Month')
    plt.ylabel('Total Amount (EUR)')
    plt.title('Monthly Expenses, Income, and Savings')
    plt.legend(title='Type')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout for better fit
    plt.grid(True)

    plt.show()


def plot_budget_status(merged_budget, show_percentage=True, month=None):
    """
    Plots a horizontal bar chart of budget performance by category/subcategory with color-coded labels and alternating row backgrounds.
    """
    color_gradient = [
        '#006400', '#1e7a1e', '#2e8b57', '#32cd32', '#adff2f',
        '#ffff00', '#ffd700', '#ffa500', '#ff4500', '#ff0000'
    ]

    def get_color(value):
        if value < 0:
            return '#ff0000'
        value = min(max(value, 0), 100)
        idx = int((1 - value / 100) * (len(color_gradient) - 1))
        return color_gradient[idx]

    # Determine data source
    if month:
        month_col = f"{month.capitalize()}_over_under"
        if month_col not in merged_budget.columns:
            raise ValueError(f"Column '{month_col}' not found in merged_budget.")
        merged_budget['value_to_plot'] = merged_budget[month_col].astype(float)
        merged_budget['bar_color'] = merged_budget['value_to_plot'].apply(
            lambda x: '#ff0000' if x < 0 else '#32cd32'
        )
        x_label = f'{month.capitalize()} Budget Status (Currency)'
        is_percentage = False
    else:
        merged_budget['percentage_of_budget'] = (
            merged_budget['year_remaining'].astype(float) / merged_budget['annually'].astype(float)
        ) * 100
        if show_percentage:
            merged_budget['value_to_plot'] = merged_budget['percentage_of_budget']
            merged_budget['bar_color'] = merged_budget['value_to_plot'].apply(get_color)
            x_label = 'Percentage of Budget (Year Remaining)'
            is_percentage = True
        else:
            merged_budget['value_to_plot'] = merged_budget['year_remaining'].astype(float)
            merged_budget['bar_color'] = merged_budget['value_to_plot'].apply(
                lambda x: '#ff0000' if x < 0 else '#32cd32'
            )
            x_label = 'Amount Remaining (Currency)'
            is_percentage = False

    # Sort and extract
    merged_budget = merged_budget.sort_values('value_to_plot', ascending=True)
    categories = merged_budget['Category'] + " - " + merged_budget['Subcategory']
    values = merged_budget['value_to_plot']
    colors = merged_budget['bar_color']

    fig, ax = plt.subplots(figsize=(16, 8))
    y_positions = np.arange(len(categories))

    # Alternating row backgrounds
    for i in range(len(categories)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=0)

    bars = ax.barh(y_positions, values, color=colors, zorder=2)
    max_val = max(values.max(), abs(values.min()))
    ax.set_xlim(-max_val * 1.3, max_val * 1.3)

    # Add bar labels
    for bar, val, color in zip(bars, values, colors):
        y = bar.get_y() + bar.get_height() / 2
        x = bar.get_width()
        label = f"{val:.0f}%" if is_percentage else f"NOK {val:,.2f}"
        align = 'left' if x >= 0 else 'right'
        offset = 8 if x >= 0 else -8
        ax.text(x + offset, y, label, va='center', ha=align, color=color, fontsize=9, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Category - Subcategory')
    ax.set_title(f'Budget Status: {month.capitalize() if month else "Annual Overview"}')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)

    plt.subplots_adjust(left=0.3)
    plt.show()