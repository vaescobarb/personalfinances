"""Streamlit app: Average expenses per month by category

This app loads a local Money Manager .xlsx file (same format used by
`personal_finances_utils.load_money_manager_file`) and shows a table of
average monthly expenses per category. It provides two averages:
 - average across the full time range (total / number_of_months_in_range)
 - average across active months for each category (total / months_with_spending)

Usage: run `streamlit run streamlit_avg_expenses.py` in the project root.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import io
import streamlit as st
import numpy as np
import pandas as pd
import calendar
import tempfile
from pathlib import Path
import io
import plotly.graph_objects as go

from personal_finances_utils import (
    load_money_manager_file,
    classify_expenses,
    compute_avg_expenses_per_month,
    plot_monthly_expenses_plotly,
    load_budget_sections,
)

def main() -> None:
    st.set_page_config(page_title="Average Expenses per Month", layout="wide")
    st.title("Average Expenses per Month by Category")

    st.markdown(
        """
        Upload a Money Manager Excel file (same format as used in this repo) or
        select a sample file from the repository. The app computes two averages:
        the average across the full date range and the average across the months
        where the category had spending.
        """
    )


    col1, col2, col3 = st.columns([2, 1, 1])

    uploaded = col1.file_uploader("Upload expenses .xlsx file", type=["xlsx"])
    uploaded_budget = col2.file_uploader("Upload budget .xlsx file", type=["xlsx"])

    # Offer sample files present in the repo `money_manager_data` if present
    sample_path = Path("money_manager_data")
    sample_files = []
    if sample_path.exists() and sample_path.is_dir():
        sample_files = sorted([str(p) for p in sample_path.glob("*.xlsx")])

    chosen_sample = None
    if sample_files:
        chosen_sample = col3.selectbox("Or pick a sample expenses file", ["-- none --"] + sample_files)

    # Offer sample budget files if present
    budget_sample_path = Path("budget_files")
    budget_sample_files = []
    if budget_sample_path.exists() and budget_sample_path.is_dir():
        budget_sample_files = sorted([str(p) for p in budget_sample_path.glob("*.xlsx")])

    chosen_budget_sample = None
    if budget_sample_files:
        chosen_budget_sample = col3.selectbox("Or pick a sample budget file", ["-- none --"] + budget_sample_files)

    df_expenses = None
    df_income = None
    df_budget = None

    # Load expenses
    if uploaded is not None:
        t = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        try:
            t.write(uploaded.read())
            t.flush()
            df_expenses, df_income = load_money_manager_file(t.name)
        finally:
            t.close()
    elif chosen_sample and chosen_sample != "-- none --":
        df_expenses, df_income = load_money_manager_file(chosen_sample)

    # Load budget
    if uploaded_budget is not None:
        t2 = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        try:
            t2.write(uploaded_budget.read())
            t2.flush()
            df_budget = load_budget_sections(t2.name)
        finally:
            t2.close()
    elif chosen_budget_sample and chosen_budget_sample != "-- none --":
        df_budget = load_budget_sections(chosen_budget_sample)

    if df_expenses is None:
        st.info("Upload an expenses .xlsx file or select a sample to begin.")
    
    # Tabs: Averages, Monthly tables, Plot, Budget
    if df_expenses is not None:
        df_expenses = classify_expenses(df_expenses)
        try:
            result = compute_avg_expenses_per_month(df_expenses)
        except Exception as e:
            st.error(f"Failed to compute averages: {e}")
            return

        tab_avgs, tab_tables, tab_plot, tab_budget = st.tabs(["Averages", "Monthly tables", "Plot", "Budget"])

        # --- Tab: Budget ---
        with tab_budget:
            st.subheader("Loaded Budget Table")
            if df_budget is not None:
                st.dataframe(df_budget)
                csv_budget = df_budget.to_csv(index=False).encode("utf-8")
                st.download_button("Download budget table (CSV)", csv_budget, "budget_table.csv", "text/csv")
            else:
                st.info("No budget file loaded. Upload or select a budget file to view.")

        # --- Tab: Averages ---
        with tab_avgs:
            st.subheader("Average expenses per month (by category)")
            st.write(
                "The table shows total spending, months active, number of months in the full range, and two averages."
            )

            st.dataframe(result.style.format({"Total": "{:.2f}", "Avg_full_range": "{:.2f}", "Avg_active_months": "{:.2f}"}))

            st.subheader("Bar chart: Average (full range)")
            chart_data = result[["Avg_full_range"]].rename(columns={"Avg_full_range": "Avg per month (full range)"})
            st.bar_chart(chart_data)

            # CSV download
            csv = result.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "avg_expenses_by_category.csv", "text/csv")

        # --- Tab: Monthly tables ---
        with tab_tables:
            st.subheader("Monthly expenses per year (by category)")

            # Prepare expenses data for monthly breakdown
            df = df_expenses.copy()
            df["Date"] = pd.to_datetime(df["Date"])
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month

            # Detect amount column (same logic as earlier)
            amount_col = None
            for col in ("EUR", "Amount", "Betrag"):
                if col in df.columns:
                    amount_col = col
                    break
            if amount_col is None:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) == 0:
                    st.error("No numeric amount column found for monthly table.")
                    return
                amount_col = numeric_cols[0]

            df["_abs_amount"] = df[amount_col].abs().astype(float)

            available_years = sorted(df["Year"].unique(), reverse=True)
            selected_years = st.multiselect("Select years to display", options=available_years, default=available_years)

            if selected_years:
                # Build a combined CSV for download later
                combined_frames = []

                for year in selected_years:
                    sub = df[df["Year"] == year]
                    if sub.empty:
                        st.write(f"No data for {year}")
                        continue

                    pivot = pd.pivot_table(
                        sub,
                        values="_abs_amount",
                        index="Category",
                        columns="Month",
                        aggfunc="sum",
                        fill_value=0,
                    )

                    # Ensure columns for all months 1..12
                    all_months = list(range(1, 13))
                    pivot = pivot.reindex(columns=all_months, fill_value=0)

                    # Rename columns to month abbreviations
                    pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]

                    # Add total, mean and std_dev columns across the 12 months
                    pivot["Total"] = pivot.sum(axis=1)

                    # Determine month columns by abbreviation
                    month_cols = [calendar.month_abbr[m] for m in all_months]

                    # Exclude months that are all zeros across categories when computing mean/std
                    non_zero_months = [c for c in month_cols if pivot[c].abs().sum() != 0]

                    if non_zero_months:
                        pivot["mean"] = pivot[non_zero_months].mean(axis=1)
                        pivot["median"] = pivot[non_zero_months].median(axis=1)
                        pivot["dev"] = pivot[non_zero_months].std(axis=1, ddof=0)
                    else:
                        # If every month column is all zeros, set mean/dev to 0
                        pivot["mean"] = 0.0
                        pivot["median"] = 0.0
                        pivot["dev"] = 0.0

                    # Sort by total, then reorder columns so Total, mean, dev appear first (right after Category index)
                    pivot = pivot.sort_values("Total", ascending=False)

                    # Build desired column order: Total, mean, median, dev, then months (Jan..Dec)
                    month_abbrs = [calendar.month_abbr[m] for m in all_months]
                    desired_cols = ["Total", "mean", "median", "dev"] + month_abbrs
                    # Keep only columns that exist in pivot (defensive)
                    desired_cols = [c for c in desired_cols if c in pivot.columns]
                    other_cols = [c for c in pivot.columns if c not in desired_cols]
                    pivot = pivot[desired_cols + other_cols]

                    # Append a 'Total' row by summing numeric columns vertically; set 'dev' and 'median' to NaN
                    totals = pivot.select_dtypes(include=["number"]).sum(axis=0)
                    # ensure dev and median are NaN in totals row if present
                    if "dev" in totals.index:
                        totals["dev"] = np.nan
                    if "median" in totals.index:
                        totals["median"] = np.nan
                    totals_df = totals.to_frame().T
                    totals_df.index = ["Total"]
                    pivot = pd.concat([pivot, totals_df], verify_integrity=False)

                    st.markdown(f"### {year}")
                    st.dataframe(pivot.style.format("{:.2f}"))

                    # store for combined export; ensure CSV column order: Category, Year, Total, mean, dev, months...
                    temp = pivot.reset_index()
                    temp.insert(1, "Year", year)
                    # Reorder temp columns if possible
                    cols_after_index = []
                    if "Year" in temp.columns:
                        cols_after_index.append("Year")
                    for c in ["Total", "mean", "dev"]:
                        if c in temp.columns:
                            cols_after_index.append(c)
                    # then months
                    for m in month_abbrs:
                        if m in temp.columns:
                            cols_after_index.append(m)
                    # keep any remaining columns
                    remaining = [c for c in temp.columns if c not in (['Category'] + cols_after_index)]
                    ordered = ["Category"] + cols_after_index + remaining
                    temp = temp[[c for c in ordered if c in temp.columns]]
                    combined_frames.append(temp)

                if combined_frames:
                    all_combo = pd.concat(combined_frames, ignore_index=True)
                    csv2 = all_combo.to_csv(index=False).encode("utf-8")
                    st.download_button("Download monthly table (CSV)", csv2, "monthly_expenses_by_year.csv", "text/csv")

        # --- Tab: Plot ---
        with tab_plot:
            st.subheader("Monthly stacked plot")
            # Detect amount column (same logic as earlier)
            amount_col_plot = None
            for col in ("EUR"):
                if col in df_expenses.columns:
                    amount_col_plot = col
                    break
            if amount_col_plot is None:
                numeric_cols = df_expenses.select_dtypes(include=["number"]).columns
                if len(numeric_cols) == 0:
                    st.error("No numeric amount column found for plotting.")
                    return
                amount_col_plot = numeric_cols[0]

            # date range selectors
            min_date = pd.to_datetime(df_expenses["Date"]).min().date()
            max_date = pd.to_datetime(df_expenses["Date"]).max().date()
            start_date, end_date = st.date_input(
                "Select min and max plotted dates",
                value=(min_date, max_date),
                help="Choose the date range to display (inclusive)",
            )

            # normalize inputs
            if isinstance(start_date, tuple) or isinstance(start_date, list):
                # some streams return a tuple when only one field is used
                start_date, end_date = start_date

            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)

            # Filter expenses and income by selected date range
            df_plot = df_expenses.copy()
            df_plot["Date"] = pd.to_datetime(df_plot["Date"])
            mask = (df_plot["Date"] >= start_ts) & (df_plot["Date"] <= end_ts)
            df_plot = df_plot.loc[mask]

            # Prepare monthly Period index
            if df_plot.empty:
                st.info("No data in selected date range.")
            else:
                df_plot["Month"] = df_plot["Date"].dt.to_period("M")
                # treat amounts as positive
                df_plot["_abs_amount"] = df_plot[amount_col_plot].abs().astype(float)

                # Group by Month and Type
                if "Type" in df_plot.columns:
                    monthly = df_plot.groupby(["Month", "Type"]) ["_abs_amount"].sum().unstack(fill_value=0)
                else:
                    # fallback: all expenses as single column
                    monthly = df_plot.groupby(["Month"]) ["_abs_amount"].sum().to_frame(name="Expenses")

                # Include savings if income_df is available
                if df_income is not None:
                    income_plot = df_income.copy()
                    income_plot["Date"] = pd.to_datetime(income_plot["Date"]) 
                    income_plot = income_plot[(income_plot["Date"] >= start_ts) & (income_plot["Date"] <= end_ts)]
                    if not income_plot.empty:
                        income_plot["Month"] = income_plot["Date"].dt.to_period("M")
                        monthly_income = income_plot.groupby("Month")[amount_col_plot].sum()
                        monthly_total_exp = monthly.sum(axis=1) if isinstance(monthly, pd.DataFrame) else monthly["Expenses"]
                        monthly_savings = monthly_income - monthly_total_exp
                        # add SAVE as a column aligned to monthly index
                        monthly = monthly.copy()
                        monthly["SAVE"] = monthly_savings.reindex(monthly.index, fill_value=0)

                # Plot stacked bar using Plotly via a reusable helper so it mirrors
                # personal_finances_utils.plot_monthly_expenses but is interactive.


                try:
                    fig = plot_monthly_expenses_plotly(df_plot, df_income, amount_col_plot)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to build interactive plot: {e}")


if __name__ == "__main__":
    main()
