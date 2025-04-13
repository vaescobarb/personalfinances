
import matplotlib.pyplot as plt

import numpy as np

def plot_expenses_vs_income(expenses_df, income_df):
    """
    Creates an overlapping line and bar plot of expenses and income by month.

    Parameters:
        expenses_df (pd.DataFrame): DataFrame containing expense data with columns ['Month', 'Category', 'EUR'].
        income_df (pd.DataFrame): DataFrame containing income data with columns ['Month', 'EUR'].

    Returns:
        None
    """
    # Group by month and category, and sum the expenses
    monthly_expenses = expenses_df.groupby(['Month', 'Category'])['EUR'].sum().unstack()

    # Group by month and sum the income
    monthly_income = income_df.groupby(['Month'])['EUR'].sum()

    # Assuming monthly_expenses and monthly_income are DataFrames/Series with a proper index
    categories = monthly_expenses.index  # Use the DataFrame index

    # Convert categories to a range of numbers if necessary
    x = np.arange(len(categories))  # Numeric x values for the bar plot

    plt.figure(figsize=(16, 8))

    # Create a bar plot for monthly expenses
    plt.bar(x, monthly_expenses.sum(axis=1), color='lightblue', alpha=0.6, label='Total Expenses')

    # Create a line plot for monthly income
    print(monthly_income)
    plt.plot(x, monthly_income, color='green', marker='o', markersize=2, label='Total Income', linewidth=1)

    # Set x-ticks to the actual category labels
    plt.xticks(x, categories, rotation=90)  # Rotate x-axis labels if needed

    # Add labels and title
    plt.xlabel('Months')
    plt.ylabel('Values (EUR)')
    plt.title('Overlapping Line and Bar Plot of Expenses and Income')
    plt.legend()

    # Turn on grid
    plt.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()