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
from currency_conversion_utils import get_nok_conversion, load_exchange_rates

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

    tab_avgs, tab_tables, tab_plot, tab_budget, tab_budget_tracker, tab_adaptive = st.tabs(["Averages", "Monthly tables", "Plot", "Budget", "Budget-Tracker", "Adaptive Budget"])
    # --- Tab: Adaptive Budget ---
    with tab_adaptive:
        st.subheader("Adaptive Budget Calculator")
        st.markdown(
            """
            Calculate adaptive budgets based on your spending so far this year.
            The adaptive budget adjusts monthly budgets downward for remaining months
            (never increases) to match your actual spending pace.
            """
        )
        
        if df_budget is None:
            st.warning("⚠️ Please upload/select a budget file first to use adaptive budget calculator.")
        elif df_expenses is None:
            st.warning("⚠️ Please upload/select an expenses file first.")
        else:
            import personal_finances_utils as pf_utils
            from datetime import datetime, date
            
            # Date selection
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                st.markdown("**Current Date (for calculation)**")
                use_today = st.checkbox("Use today's date", value=True, key="adaptive_use_today")
                if use_today:
                    current_date = date.today()
                    st.info(f"📅 Using today: {current_date.strftime('%Y-%m-%d')}")
                else:
                    current_date = st.date_input("Select current date", value=date.today(), key="adaptive_date")
            
            with col2:
                st.markdown("**Or select manually**")
                if not use_today:
                    target_year_manual = st.selectbox("Year", range(2020, 2031), index=4, key="adaptive_year")
                    target_month_manual = st.selectbox("Month", range(1, 13), index=current_date.month - 1, key="adaptive_month")
                    current_date = date(target_year_manual, target_month_manual, 1)
                    st.info(f"📅 Selected: {current_date.strftime('%B %Y')}")
            
            with col3:
                st.markdown("**Output Settings**")
                output_folder = st.text_input(
                    "Output folder path",
                    value="./adaptive_budgets",
                    help="Where to save CSV/JSONL files. Created if doesn't exist."
                )
                export_format = st.selectbox(
                    "Export format",
                    ["CSV", "JSONL", "Both", "None"],
                    help="CSV for spreadsheets, JSONL for data pipelines"
                )
            
            # Year and month from current_date
            target_year = current_date.year
            target_month = current_date.month
            
            st.divider()
            
            # Prepare data for adaptive budget
            if st.button("🧮 Calculate Adaptive Budget", type="primary", use_container_width=True):
                try:
                    # Filter expenses for the selected year
                    df_expenses_year = df_expenses[df_expenses["Date"].dt.year == target_year].copy()
                    
                    if df_expenses_year.empty:
                        st.warning(f"No expense data found for year {target_year}")
                    else:
                        # Load exchange rates if needed
                        json_file_path = f'{target_year}_exchange_rates.json'
                        try:
                            exchange_rates = load_exchange_rates(json_file_path)
                            df_expenses_year.loc[:, 'NOK'] = df_expenses_year.apply(
                                get_nok_conversion, axis=1, exchange_rates=exchange_rates
                            )
                        except:
                            raise Exception("Failed to load exchange rates.")
                            #pass  # Continue without NOK conversion if file not found
                        
                        # Create yearly summary
                        summary_table = pf_utils.make_yearly_expense_summary(df_expenses_year, target_year)
                        
                        # Merge with budget
                        merged_budget, missing_combinations = pf_utils.merge_expenses_and_budget(
                            summary_table, df_budget, print_missing=False
                        )

                        # Prepare budget DataFrame for adaptive calculation
                        budget_for_adaptive = df_budget[['Category', 'Subcategory', 'monthly', 'annually']].copy()
                        budget_for_adaptive = budget_for_adaptive.rename(columns={'monthly': 'monthly', 'annually': 'annually'})

                        # Calculate correct totals for spent by category using the same method as in Averages tab
                        # Include both merged_budget and missing_combinations (exclude 'Category Total' rows)
                        spent_totals = merged_budget.groupby(['Category', 'Subcategory'])['Total'].sum().reset_index()

                        # Add missing combinations' totals (from summary) applying exceptions logic:
                        # For exception categories keep only 'Category Total' rows; for other categories keep non-'Category Total' rows
                        if not missing_combinations.empty:
                            exceptions = ['Apparel', 'Household']
                            mask_exception = missing_combinations['Category'].isin(exceptions)
                            mask_cat_total = missing_combinations['Subcategory'] == 'Category Total'
                            missing_filtered = missing_combinations[(mask_exception & mask_cat_total) | (~mask_exception & ~mask_cat_total)].copy()

                            if 'Total' in missing_filtered.columns:
                                missing_spent = missing_filtered[['Category', 'Subcategory', 'Total']].copy()
                                spent_totals = pd.concat([spent_totals, missing_spent], ignore_index=True)
                            else:
                                # Fallback: if Total column not present, attempt to compute from month columns
                                month_cols = [c for c in missing_filtered.columns if c not in ('Category', 'Subcategory')]
                                if month_cols:
                                    missing_filtered['Total'] = missing_filtered[month_cols].sum(axis=1)
                                    missing_spent = missing_filtered[['Category', 'Subcategory', 'Total']].copy()
                                    spent_totals = pd.concat([spent_totals, missing_spent], ignore_index=True)

                        # Aggregate again to ensure uniqueness
                        spent_totals = spent_totals.groupby(['Category', 'Subcategory'])['Total'].sum().reset_index()

                        # Determine output path and format
                        if export_format == "None":
                            output_path = None
                            format_to_use = None
                        else:
                            import os
                            os.makedirs(output_folder, exist_ok=True)
                            output_path = os.path.join(output_folder, f"{target_year}_m{target_month:02d}_adaptive")
                            if export_format == "Both":
                                format_to_use = "csv"
                            else:
                                format_to_use = export_format.lower()

                        # Calculate adaptive budget
                        adaptive_budget, metadata = pf_utils.calculate_adaptive_budget(
                            budget_df=budget_for_adaptive,
                            spent_by_category=spent_totals,
                            current_month=target_month,
                            year=target_year,
                            output_format=format_to_use,
                            output_path=output_path
                        )

                        # If "Both" format, also export as JSONL
                        if export_format == "Both":
                            adaptive_budget, _ = pf_utils.calculate_adaptive_budget(
                                budget_df=budget_for_adaptive,
                                spent_by_category=spent_totals,
                                current_month=target_month,
                                year=target_year,
                                output_format='jsonl',
                                output_path=output_path
                            )

                        # Display metrics
                        st.success("✅ Adaptive budget calculated successfully!")
                        st.divider()

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📅 Current Month", calendar.month_name[target_month])
                        col2.metric("⏳ Months Remaining", metadata['months_remaining'])
                        col3.metric("📊 Categories Affected", metadata['categories_affected'])
                        col4.metric("📉 Avg Reduction %", f"{metadata['total_reduction_percentage']:.1f}%")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("💰 Total Annual Budget", f"{metadata['total_annual_budget']:,.2f} NOK")
                        with col2:
                            # Prefer to display the aggregated spent total we computed (includes missing combinations)
                            try:
                                combined_total_spent = float(spent_totals['Total'].sum())
                            except Exception:
                                # fallback to metadata if spent_totals is not available for some reason
                                combined_total_spent = metadata.get('total_spent', 0.0)
                            st.metric("💸 Total Spent", f"{combined_total_spent:,.2f} NOK")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📈 Total Remaining", f"{metadata['total_remaining']:,.2f} NOK")
                        with col2:
                            st.metric("🎯 Calculation Date", current_date.strftime('%Y-%m-%d'))
                        
                        st.divider()
                        
                        # Show the adaptive budget table
                        st.subheader("Adaptive Budget Details")
                        
                        # Format the display
                        display_df = adaptive_budget.copy()
                        display_df = display_df.round(2)
                        
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=600,
                            column_config={
                                "Category": st.column_config.TextColumn("Category", width="medium"),
                                "Subcategory": st.column_config.TextColumn("Subcategory", width="medium"),
                                "original_monthly": st.column_config.NumberColumn("Original Monthly", format="%.2f NOK"),
                                "annually": st.column_config.NumberColumn("Annual", format="%.2f NOK"),
                                "spent_to_date": st.column_config.NumberColumn("Spent So Far", format="%.2f NOK"),
                                "remaining_annual": st.column_config.NumberColumn("Remaining Annual", format="%.2f NOK"),
                                "months_remaining": st.column_config.NumberColumn("Months Left"),
                                "adaptive_monthly": st.column_config.NumberColumn("Adaptive Monthly", format="%.2f NOK"),
                                "reduction_percentage": st.column_config.NumberColumn("Reduction %", format="%.1f%%"),
                            }
                        )
                        
                        # Display missing combinations if any, applying the same exceptions logic
                        if not missing_combinations.empty:
                            exceptions = ['Apparel', 'Household']
                            mask_exception = missing_combinations['Category'].isin(exceptions)
                            mask_cat_total = missing_combinations['Subcategory'] == 'Category Total'
                            missing_display = missing_combinations[(mask_exception & mask_cat_total) | (~mask_exception & ~mask_cat_total)].copy()
                            st.subheader("⚠️ Missing Combinations in Budget")
                            st.warning(f"The following category/subcategory combinations are in your expenses but not in the budget: {missing_display.shape[0]} combinations (Category Total kept for exceptions)")
                            if not missing_display.empty:
                                st.dataframe(missing_display, use_container_width=True)
                            else:
                                st.info("No missing combinations to display after applying exceptions filter.")
                        
                        st.divider()
                        
                        # Download buttons
                        st.subheader("📥 Download Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            csv_data = adaptive_budget.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv_data,
                                file_name=f"adaptive_budget_{target_year}_m{target_month:02d}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            import json
                            jsonl_lines = [json.dumps({"_metadata": metadata})]
                            jsonl_lines.extend([json.dumps(row.to_dict()) for _, row in adaptive_budget.iterrows()])
                            jsonl_data = '\n'.join(jsonl_lines).encode('utf-8')
                            st.download_button(
                                label="📥 Download as JSONL",
                                data=jsonl_data,
                                file_name=f"adaptive_budget_{target_year}_m{target_month:02d}.jsonl",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col3:
                            # Export path info
                            if export_path := output_path:
                                st.info(f"💾 Files also saved to:\n`{export_path}`")
                        
                        # Show categories with significant cuts
                        st.divider()
                        st.subheader("📍 Budget Adjustments Summary")
                        
                        affected = adaptive_budget[adaptive_budget['reduction_percentage'] > 0].sort_values(
                            'reduction_percentage', ascending=False
                        )
                        
                        if affected.empty:
                            st.success("✅ No budget cuts needed - you're on track!")
                        else:
                            if len(affected) <= 5:
                                st.dataframe(
                                    affected[['Category', 'Subcategory', 'original_monthly', 'adaptive_monthly', 'reduction_percentage']],
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "original_monthly": st.column_config.NumberColumn("Original Monthly", format="%.2f NOK"),
                                        "adaptive_monthly": st.column_config.NumberColumn("Adaptive Monthly", format="%.2f NOK"),
                                        "reduction_percentage": st.column_config.NumberColumn("Reduction %", format="%.1f%%"),
                                    }
                                )
                            else:
                                st.write(f"Categories with budget cuts ({len(affected)}):")
                                st.dataframe(
                                    affected[['Category', 'Subcategory', 'original_monthly', 'adaptive_monthly', 'reduction_percentage']],
                                    use_container_width=True,
                                    hide_index=True,
                                    height=300,
                                    column_config={
                                        "original_monthly": st.column_config.NumberColumn("Original Monthly", format="%.2f NOK"),
                                        "adaptive_monthly": st.column_config.NumberColumn("Adaptive Monthly", format="%.2f NOK"),
                                        "reduction_percentage": st.column_config.NumberColumn("Reduction %", format="%.1f%%"),
                                    }
                                )
                
                except Exception as e:
                    st.error(f"❌ Error calculating adaptive budget: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # --- Tab: Budget-Tracker ---
    with tab_budget_tracker:
        st.subheader("Budget Tracker")
        import personal_finances_utils as pf_utils
        # Select year from file
        if "Year" in df_expenses.columns:
            years = sorted(df_expenses["Year"].dropna().unique(), reverse=True)
        elif "Date" in df_expenses.columns:
            years = sorted(pd.to_datetime(df_expenses["Date"]).dt.year.dropna().unique(), reverse=True)
        else:
            years = []
        target_year = st.selectbox("Select year for budget tracking", years) if years else None
        if target_year is not None and df_budget is not None:
            # Prepare summary table
            df_expenses_year = df_expenses[df_expenses["Date"].dt.year == target_year]

            #################################################################################
            json_file_path = f'{target_year}_exchange_rates.json'
            exchange_rates = load_exchange_rates(json_file_path)
            df_expenses_year.loc[:,'NOK'] = df_expenses_year.apply(get_nok_conversion, axis=1, exchange_rates=exchange_rates)
            
            
            ###############################################
            summary_table = pf_utils.make_yearly_expense_summary(df_expenses_year, target_year)
            st.write("Yearly Expense Summary Table:")
            st.dataframe(summary_table)
            # Merge with budget
            merged_budget, missing_combinations = pf_utils.merge_expenses_and_budget(summary_table, df_budget, print_missing=False)
            st.write("Merged Budget Table:")
            st.dataframe(merged_budget)
            if not missing_combinations.empty:
                st.warning(f"Missing combinations in budget: {missing_combinations.shape[0]}")
                st.dataframe(missing_combinations)
            # Calculate over/under expenditure
            folder_path = None
            file_name = None
            over_under_expenditure = pf_utils.calculate_over_under_expenditure(merged_budget, folder_path, file_name, export_xlsx=False)
            st.write("Over/Under Expenditure Table:")
            st.dataframe(over_under_expenditure)
            # Plot budget status
            st.write("Budget Status Plot:")
            import matplotlib.pyplot as plt
            exclude_categories = [
                'Miete - Miete', 'Food - Groceries', 'Transfer family - Transfer family',
                'Food - Lunch', 'Transportation - Bus', 'Travel - Venezuela'
            ]
            show_percentage = st.checkbox("Show percentage of budget", value=False)
            month = st.selectbox("Select month for plot (optional)", [None] + [m.replace('_over_under','') for m in over_under_expenditure.columns if m.endswith('_over_under')])
            valid_month = month if month else None
            try:
                pf_utils.plot_budget_status(over_under_expenditure, show_percentage=show_percentage, month=valid_month, exclude_categories=exclude_categories)
                st.pyplot(plt)
            except Exception as e:
                st.error(f"Error plotting budget status: {e}")
        else:
            st.info("Please upload/select both expenses and budget files, and select a year.")

        # --- Tab: Budget ---
        with tab_budget:
            st.subheader("Loaded Budget Table")
            if df_budget is not None:
                st.dataframe(df_budget, height=800)
                csv_budget = df_budget.to_csv(index=False).encode("utf-8")
                st.download_button("Download budget table (CSV)", csv_budget, "budget_table.csv", "text/csv")

                # Aggregate by Category for semi-monthly, monthly, annually columns
                agg_cols = [c for c in ["Semi-monthly", "monthly", "annually"] if c in df_budget.columns]
                if agg_cols:
                    st.subheader("Aggregated Budget by Category")
                    budget_agg = df_budget.groupby("Category")[agg_cols].sum().reset_index()
                    st.dataframe(budget_agg, height=660)
                    # Show total monthly budget for aggregated table
                    if "monthly" in budget_agg.columns:
                        total_monthly_agg = budget_agg["monthly"].sum()
                        st.info(f"Total monthly budget (aggregated): {total_monthly_agg:,.2f}")
                    csv_agg = budget_agg.to_csv(index=False).encode("utf-8")
                    st.download_button("Download aggregated budget (CSV)", csv_agg, "budget_aggregated_by_category.csv", "text/csv")
                else:
                    st.info("No semi-monthly, monthly, or annually columns found in budget file.")
            else:
                st.info("No budget file loaded. Upload or select a budget file to view.")

        # --- Tab: Averages ---
        with tab_avgs:
            st.subheader("Average expenses per month (by category)")
            st.write(
                "Select a year and currency to view average expenses per month by category. The table shows total spending, months active, number of months in the full range, and two averages."
            )

            # Year selection
            years = sorted(df_expenses["Date"].dt.year.unique(), reverse=True)
            selected_year = st.selectbox("Select year", years, index=0)

            # Currency selection
            currency_option = st.selectbox("Select currency", ["EUR", "NOK"], index=0)

            # Filter expenses for selected year
            df_year = df_expenses[df_expenses["Date"].dt.year == selected_year].copy()

            # Apply currency conversion if NOK selected
            amount_col = None
            for col in ("EUR", "Amount", "Betrag"):
                if col in df_year.columns:
                    amount_col = col
                    break
            if amount_col is None:
                numeric_cols = df_year.select_dtypes(include=["number"]).columns
                if len(numeric_cols) == 0:
                    st.error("No numeric amount column found for averages table.")
                    return
                amount_col = numeric_cols[0]

            if currency_option == "NOK":
                json_file_path = f'{selected_year}_exchange_rates.json'
                try:
                    exchange_rates = load_exchange_rates(json_file_path)
                    df_year["NOK"] = df_year.apply(get_nok_conversion, axis=1, exchange_rates=exchange_rates)
                    used_col = "NOK"
                except Exception as e:
                    st.error(f"Failed to load exchange rates for NOK conversion: {e}")
                    return
            else:
                used_col = amount_col

            # Compute averages for selected year and currency
            try:
                result_year = compute_avg_expenses_per_month(df_year, amount_col=used_col)
            except Exception as e:
                st.error(f"Failed to compute averages: {e}")
                return

            # Report total expenses for selected year and currency
            total_expenses = df_year[used_col].abs().sum()
            st.info(f"Total expenses for {selected_year} ({currency_option}): {total_expenses:,.2f}")

            st.dataframe(result_year.style.format({"Total": "{:.2f}", "Avg_full_range": "{:.2f}", "Avg_active_months": "{:.2f}"}))

            st.subheader("Average expenses by Category and Subcategory")
            # Use make_yearly_expense_summary to build detailed subcategory view
            import personal_finances_utils as pf_utils

            try:
                summary_table = pf_utils.make_yearly_expense_summary(df_year, selected_year, amount_col=used_col)
            except Exception as e:
                st.error(f"Failed to build yearly summary: {e}")
                summary_table = pd.DataFrame()

            if not summary_table.empty:
                # Months in year (January..December)
                all_months = list(calendar.month_name[1:])

                # Compute months in range from the data
                month_min = pd.to_datetime(df_year['Date']).dt.to_period('M').min().to_timestamp()
                month_max = pd.to_datetime(df_year['Date']).dt.to_period('M').max().to_timestamp()
                months_in_range = len(pd.period_range(start=month_min, end=month_max, freq='M'))

                detailed_summary = summary_table.copy()
                # Months active (count of months with spending)
                detailed_summary['Months_Active'] = (detailed_summary[all_months] > 0).sum(axis=1)

                # Transaction counts from raw data
                tx_counts = df_year.groupby(['Category', 'Subcategory']).size().reset_index(name='Transaction_Count')
                detailed_summary = detailed_summary.merge(tx_counts, on=['Category', 'Subcategory'], how='left')
                detailed_summary['Transaction_Count'] = detailed_summary['Transaction_Count'].fillna(0).astype(int)

                detailed_summary['Avg_full_range'] = detailed_summary['Total'] / max(months_in_range, 1)
                detailed_summary['Avg_active_months'] = detailed_summary['Total'] / detailed_summary['Months_Active'].replace(0, 1)
                detailed_summary = detailed_summary.sort_values('Total', ascending=False)

                # Exclude 'Category Total' rows except for specific categories
                exceptions = ['Apparel', 'Household']
                mask_exception = detailed_summary['Category'].isin(exceptions)
                mask_cat_total = detailed_summary['Subcategory'] == 'Category Total'
                # For exception categories, keep only the 'Category Total' row;
                # for other categories, keep only rows that are not 'Category Total'.
                filtered_summary = detailed_summary[(mask_exception & mask_cat_total) | (~mask_exception & ~mask_cat_total)].copy()

                # Compute total expenses for the filtered table
                total_for_table = filtered_summary['Total'].sum()
                st.info(f"Total (Category+Subcategory table) for {selected_year} ({currency_option}): {total_for_table:,.2f}")

                st.dataframe(
                    filtered_summary.style.format({
                        'Total': '{:.2f}',
                        'Avg_full_range': '{:.2f}',
                        'Avg_active_months': '{:.2f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info('No summary data available for the selected year/currency.')

            st.subheader("Bar chart: Average (full range)")
            chart_data = result_year[["Avg_full_range"]].rename(columns={"Avg_full_range": f"Avg per month (full range) ({currency_option})"})
            st.bar_chart(chart_data)

            # CSV download
            csv = result_year.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, f"avg_expenses_by_category_{selected_year}_{currency_option}.csv", "text/csv")

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

                    # Append a 'Total' row by summing numeric columns vertically; set 'dev' to NaN
                    totals = pivot.select_dtypes(include=["number"]).sum(axis=0)
                    # ensure dev and median are NaN in totals row if present
                    if "dev" in totals.index:
                        totals["dev"] = np.nan
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
