# --- Budget loading utility ---
def load_budget_sections(file_path, sheet_name='EXPENSES 2025'):
    """
    Load and concatenate regular and irregular budget sections from a budget Excel file.

    Args:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to load.

    Returns:
        pd.DataFrame: Simplified budget DataFrame with cleaned columns.
    """
    def extract_section(df, category_idx, total_idx):
        # Extract relevant columns and rows
        section = df.iloc[category_idx:total_idx, 1:6]
        section.columns = section.iloc[0]
        section = section[1:]
        section = section.reset_index(drop=True)
        return section.copy(deep=True)

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Use DataFrame.map instead of applymap for string cleaning
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    # Find indices for regular and irregular sections
    category_indices = df[df.iloc[:, 1].astype(str).str.lower() == 'category'].index
    total_indices = df[df.iloc[:, 2].astype(str).str.lower() == 'total'].index

    if len(category_indices) < 2 or len(total_indices) < 2:
        raise ValueError("Could not find both regular and irregular sections in the budget file.")

    regular = extract_section(df, category_indices[0], total_indices[0])
    irregular = extract_section(df, category_indices[1], total_indices[1])

    # Concatenate and clean
    budget = pd.concat([regular, irregular]).sort_values(by='Category')
    budget['Category'] = budget['Category'].str.strip()
    budget['Subcategory'] = budget['Subcategory'].str.strip()
    budget = budget.rename(columns={'Monthly': 'monthly', 'Annually': 'annually'})
    budget = budget.reset_index(drop=True)
    return budget
# ...existing code...

import pandas as pd
import plotly.graph_objects as go

def compute_avg_expenses_per_month(expenses_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with average expenses per month by Category.

    Produces both averages over the full date range and averages over
    the months where the category had spending.
    """
    if expenses_df.empty:
        return pd.DataFrame(columns=["Category", "Total", "Months_active", "Months_in_range", "Avg_full_range", "Avg_active_months"]).set_index("Category")

    # Ensure Date and Month columns
    expenses_df = expenses_df.copy()
    expenses_df["Date"] = pd.to_datetime(expenses_df["Date"])
    expenses_df["Month"] = expenses_df["Date"].dt.to_period("M")

    # Determine numeric amount column: prefer 'EUR', then 'Amount'
    amount_col = None
    for col in ("EUR", "Amount", "Betrag"):
        if col in expenses_df.columns:
            amount_col = col
            break
    if amount_col is None:
        # Fallback: pick first numeric column
        numeric_cols = expenses_df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric amount column found in expenses data.")
        amount_col = numeric_cols[0]

    # Treat expenses as positive numbers for totals
    expenses_df["_abs_amount"] = expenses_df[amount_col].abs().astype(float)

    # Total per category
    totals = expenses_df.groupby("Category")["_abs_amount"].sum()

    # Active months per category
    months_active = expenses_df.groupby("Category")["Month"].nunique()

    # Full months in dataset range
    month_min = expenses_df["Month"].min().to_timestamp()
    month_max = expenses_df["Month"].max().to_timestamp()
    months_in_range = pd.period_range(start=month_min, end=month_max, freq="M")
    n_months_range = len(months_in_range)

    avg_full = totals / max(n_months_range, 1)
    avg_active = totals / months_active.replace(0, 1)

    # Build monthly sums matrix (categories x months in full range) to compute mean & std
    # Ensure months in the full range are present as columns (fill missing with 0)
    monthly = (
        expenses_df.groupby(["Category", "Month"])["_abs_amount"].sum().unstack(fill_value=0)
    )

    # If some months in the full range are missing columns in `monthly`, reindex them
    all_periods = pd.period_range(start=month_min, end=month_max, freq="M")
    if not all(p in monthly.columns for p in all_periods):
        monthly = monthly.reindex(columns=all_periods, fill_value=0)

    mean_per_category = monthly.mean(axis=1)
    std_per_category = monthly.std(axis=1, ddof=0)

    result = pd.DataFrame(
        {
            "Total": totals,
            "Months_active": months_active,
            "Months_in_range": n_months_range,
            "Avg_full_range": avg_full,
            "Avg_active_months": avg_active,
            "Mean": mean_per_category,
            "Dev": std_per_category,
        }
    )

    result = result.sort_values("Avg_full_range", ascending=False)
    return result

def plot_monthly_expenses_plotly(expenses_df: pd.DataFrame, income_df: pd.DataFrame | None, amount_column: str) -> go.Figure:
    # Ensure Month period and numeric amount
    exp = expenses_df.copy()
    exp.loc[:, "Month"] = pd.to_datetime(exp["Date"]).dt.to_period("M")

    # pick amount column (prefer provided, then common names)
    amt_col = amount_column if amount_column in exp.columns else None
    if amt_col is None:
        for c in ("EUR", "Amount", "Betrag"):
            if c in exp.columns:
                amt_col = c
                break
    if amt_col is None:
        numeric_cols = exp.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No amount column found for plotting")
        amt_col = numeric_cols[0]

    exp["_amt"] = exp[amt_col].abs().astype(float)

    # Group by Month and Type (like the util function)
    if "Type" in exp.columns:
        monthly_expenses = exp.groupby(["Month", "Type"])["_amt"].sum().unstack(fill_value=0)
    else:
        monthly_expenses = exp.groupby(["Month"])["_amt"].sum().to_frame(name="Expenses")

    # Add SAVE from income if available
    if income_df is not None:
        inc = income_df.copy()
        inc.loc[:, "Month"] = pd.to_datetime(inc["Date"]).dt.to_period("M")
        monthly_income = inc.groupby("Month")[amt_col].sum()
        total_monthly_exp = monthly_expenses.sum(axis=1) if isinstance(monthly_expenses, pd.DataFrame) else monthly_expenses["Expenses"]
        monthly_savings = monthly_income - total_monthly_exp
        monthly_savings = monthly_savings[monthly_savings > 0]
        monthly_savings.name = "SAVE"
        monthly_expenses = monthly_expenses.copy()
        monthly_expenses["SAVE"] = monthly_savings.reindex(monthly_expenses.index)

    # Order columns similar to utilities: NEED, UNKNOWN, WANT, SAVE then others
    preferred = ["NEED", "UNKNOWN", "WANT", "SAVE"]
    cols = [c for c in preferred if c in monthly_expenses.columns]
    cols += [c for c in monthly_expenses.columns if c not in cols]
    monthly_expenses = monthly_expenses[cols]

    # Build x labels and color mapping
    x = [str(m) for m in monthly_expenses.index]
    color_map = {"NEED": "#D62728", "UNKNOWN": "#66B2FF", "WANT": "#FFBF00", "SAVE": "#228B22"}

    fig = go.Figure()
    for col in monthly_expenses.columns:
        y = monthly_expenses[col].fillna(0).values
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name=str(col),
                marker_color=color_map.get(col, None),
                hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f}<extra></extra>",
            )
        )

    # Annotations: total per type (staggered like the matplotlib version)
    category_totals = monthly_expenses.sum(axis=0)
    annotations = []
    for i, (etype, total) in enumerate(category_totals.items()):
        annotations.append(dict(
            xref="paper",
            yref="paper",
            x=0.01,
            y=1.02 - i * 0.05,
            xanchor="left",
            text=f"Total {etype}: {total:.2f}",
            showarrow=False,
            font=dict(color=color_map.get(etype, "black"), size=12, family="Arial",),
            bgcolor="white",
        ))

    fig.update_layout(
        barmode="stack",
        title_text="Monthly Expenses, Income, and Savings",
        xaxis_title="Month",
        yaxis_title="Total Amount (EUR)",
        legend_title_text="Type",
        annotations=annotations,
        template="plotly_white",
        height=600,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(showgrid=True)

    return fig
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

def make_yearly_expense_summary(df_expenses_year, target_year):
    import pandas as pd
    import calendar
    
    df = df_expenses_year.copy(deep=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Subcategory'] = df.apply(lambda row: row['Category'] if row['Subcategory'] == 'Other' else row['Subcategory'], axis=1)
    df['Subcategory'] = df['Subcategory'].fillna(df['Category'])
    df_filtered = df[(df['Date'].dt.year == target_year) & (df['Income/Expense'] == 'Ausg.')]
    df_filtered['Month'] = df_filtered['Date'].dt.month
    df_grouped = df_filtered.groupby(['Category', 'Subcategory', 'Month'])['NOK'].sum().reset_index()
    summary_table = df_grouped.pivot(index=['Category', 'Subcategory'], columns='Month', values='NOK')
    summary_table = summary_table.fillna(0)
    summary_table.columns = pd.to_datetime(summary_table.columns, format='%m').strftime('%B')
    summary_table = summary_table.reset_index()
    summary_table.index.name = None
    summary_table['Total'] = summary_table.iloc[:, 2:].sum(axis=1)
    category_totals = summary_table.groupby('Category').sum(numeric_only=True)
    category_totals['Subcategory'] = 'Category Total'
    summary_table = pd.concat([summary_table, category_totals.reset_index()])
    # Use DataFrame.map instead of applymap for rounding
    summary_table.iloc[:, 2:] = summary_table.iloc[:, 2:].map(lambda x: round(x, 1))
    summary_table['Category'] = summary_table['Category'].str.strip()
    summary_table['Subcategory'] = summary_table['Subcategory'].str.strip()
    all_months = list(calendar.month_name[1:])
    for month in all_months:
        if month not in summary_table.columns:
            summary_table[month] = 0
    return summary_table

def merge_expenses_and_budget(summary_table, budget_simplified, print_missing=False):
    """
    Merge expenses summary and budget, and optionally print missing combinations.
    Returns merged_budget and missing_combinations DataFrames.
    Args:
        summary_table (pd.DataFrame): Expense summary table.
        budget_simplified (pd.DataFrame): Simplified budget table.
        print_missing (bool): If True, prints missing combinations.
    Returns:
        tuple: (merged_budget, missing_combinations)
    """
    merged_budget = summary_table.merge(budget_simplified, left_on=['Category', 'Subcategory'], right_on=['Category', 'Subcategory'])
    merged_set = set(zip(merged_budget['Category'], merged_budget['Subcategory']))
    missing_combinations = budget_simplified[~budget_simplified.set_index(['Category', 'Subcategory']).index.isin(merged_set)]
    if print_missing:
        print(f"MISSING COMBINATIONS:\n{missing_combinations}")
    return merged_budget, missing_combinations

def calculate_over_under_expenditure(merged_budget, folder_path=None, file_name=None, export_xlsx=False):
    """
    Calculate over/under expenditure for each month and year remaining, with optional export to Excel.
    Args:
        merged_budget (pd.DataFrame): Merged budget DataFrame containing monthly and annual budget columns.
        folder_path (str, optional): Folder path for saving the Excel file.
        file_name (str, optional): File name for saving the Excel file.
        export_xlsx (bool, optional): If True, exports the result to Excel.
    Returns:
        pd.DataFrame: Updated DataFrame with over/under expenditure and year remaining columns.
    """
    import pandas as pd
    import calendar
    from datetime import datetime
    import os
    
    # Step 1: Calculate 'over/under expenditure' for each month using vectorized operations
    current_month = datetime.now().month
    months = [str(calendar.month_name[i]) for i in range(1, current_month + 1)]
    monthly_over_under = - merged_budget[months].subtract(merged_budget['monthly'], axis=0)
    monthly_over_under.columns = [f'{month}_over_under' for month in months]
    
    # Step 2: Concatenate the results back to the original DataFrame
    merged_budget = pd.concat([merged_budget, monthly_over_under], axis=1)
    
    # Step 3: Calculate 'year remaining' efficiently
    merged_budget['year_remaining'] = merged_budget['annually'] - merged_budget['Total']
    merged_budget = merged_budget.sort_values(by='Category', ascending=True)
    
    # Step 4: Drop month columns
    merged_budget = merged_budget.drop(columns=months)
    
    # Step 5: Optionally export to Excel
    if export_xlsx and folder_path and file_name:
        output_path = os.path.join(folder_path, 'budget_status_files', file_name.split('/')[1].split('.')[0] + '_over_under_expenditure.xlsx')
        merged_budget.to_excel(output_path, index=False, sheet_name='Budget Analysis')
    
    return merged_budget

def plot_budget_status(merged_budget, show_percentage=True, month=None, exclude_categories=None):
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

    # Exclude specified categories
    if exclude_categories:
        merged_budget['full_category'] = merged_budget['Category'] + " - " + merged_budget['Subcategory']
        merged_budget = merged_budget[~merged_budget['full_category'].isin(exclude_categories)]

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