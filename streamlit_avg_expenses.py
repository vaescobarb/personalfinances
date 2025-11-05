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
import calendar
import numpy as np
import pandas as pd
import streamlit as st

from personal_finances_utils import load_money_manager_file


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

    col1, col2 = st.columns([2, 1])

    uploaded = col1.file_uploader("Upload .xlsx file", type=["xlsx"])

    # Offer sample files present in the repo `money_manager_data` if present
    sample_path = Path("money_manager_data")
    sample_files = []
    if sample_path.exists() and sample_path.is_dir():
        sample_files = sorted([str(p) for p in sample_path.glob("*.xlsx")])

    chosen_sample = None
    if sample_files:
        chosen_sample = col2.selectbox("Or pick a sample file", ["-- none --"] + sample_files)

    df_expenses = None
    df_income = None

    if uploaded is not None:
        # Save upload to a temporary file and call the loader
        t = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        try:
            t.write(uploaded.read())
            t.flush()
            df_expenses, df_income = load_money_manager_file(t.name)
        finally:
            t.close()
    elif chosen_sample and chosen_sample != "-- none --":
        df_expenses, df_income = load_money_manager_file(chosen_sample)
    else:
        st.info("Upload an .xlsx file or select a sample to begin.")

    if df_expenses is not None:
        try:
            result = compute_avg_expenses_per_month(df_expenses)
        except Exception as e:
            st.error(f"Failed to compute averages: {e}")
            return

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

        # --- Monthly expense table per year ---
        st.markdown("---")
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
                remaining = [c for c in temp.columns if c not in (["Category"] + cols_after_index)]
                ordered = ["Category"] + cols_after_index + remaining
                temp = temp[[c for c in ordered if c in temp.columns]]
                combined_frames.append(temp)

            if combined_frames:
                all_combo = pd.concat(combined_frames, ignore_index=True)
                csv2 = all_combo.to_csv(index=False).encode("utf-8")
                st.download_button("Download monthly table (CSV)", csv2, "monthly_expenses_by_year.csv", "text/csv")


if __name__ == "__main__":
    main()
