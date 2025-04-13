import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml
import urllib.parse
from datetime import datetime
from msal import PublicClientApplication
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = "consumers"  # default fallback
SCOPES = ["Files.Read", "User.Read"]

# --- Authenticate using Device Code Flow ---
@st.cache_data(show_spinner=False)
def authenticate():
    app = PublicClientApplication(CLIENT_ID, authority=f"https://login.microsoftonline.com/{TENANT_ID}")
    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise Exception("Failed to create device flow")

    st.info(flow["message"])  # Show user instructions to authenticate
    result = app.acquire_token_by_device_flow(flow)
    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception(f"Failed to get token: {result.get('error_description')}")


# ------------------ ONEDRIVE HELPERS ------------------

def get_files_and_dates(folder_path, headers):
    encoded_path = urllib.parse.quote(folder_path)
    url = f'https://graph.microsoft.com/v1.0/me/drive/root:/{encoded_path}:/children'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get('value', [])
    else:
        st.error("Failed to fetch files.")
        st.json(response.json())
        return []

    return files
def get_latest_file(files):
    latest_file = None
    latest_time = None
    for file in files:
        file_time = datetime.strptime(file['lastModifiedDateTime'], '%Y-%m-%dT%H:%M:%SZ')
        if not latest_time or file_time > latest_time:
            latest_time = file_time
            latest_file = file
    return latest_file

def download_file(file, headers):
    file_id = file['id']
    download_url = f"https://graph.microsoft.com/v1.0/me/drive/items/{file_id}/content"
    response = requests.get(download_url, headers=headers)
    if response.status_code == 200:
        filename = file['name']
        with open(filename, 'wb') as f:
            f.write(response.content)
        return filename
    else:
        st.error(f"Failed to download {file['name']}")
        return None
    
# --- Load data and display ---
def load_latest_budget_file():
    access_token = authenticate()
    headers = {"Authorization": f"Bearer {access_token}"}

    folder_path = "Documentos/PERSONAL FINANCES/budget_status_files"
    files = get_files_and_dates(folder_path, headers)
    latest = get_latest_file(files)

    if latest:
        filename = download_file(latest, headers)
        if filename:
            return pd.read_excel(filename)
    return None


# ------------------ BUDGET PLOTTING FUNCTION ------------------

def plot_budget_status(df, show_percentage=True, month=None):
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

    if month:
        month_col = f"{month.capitalize()}_over_under"
        if month_col not in df.columns:
            st.error(f"Column '{month_col}' not found.")
            return
        df['value_to_plot'] = df[month_col].astype(float)
        df['bar_color'] = df['value_to_plot'].apply(lambda x: '#ff0000' if x < 0 else '#32cd32')
        x_label = f'{month.capitalize()} Budget Status (Currency)'
        is_percentage = False
    else:
        df['percentage_of_budget'] = (
            df['year_remaining'].astype(float) / df['annually'].astype(float)
        ) * 100
        if show_percentage:
            df['value_to_plot'] = df['percentage_of_budget']
            df['bar_color'] = df['value_to_plot'].apply(get_color)
            x_label = 'Percentage of Budget (Year Remaining)'
            is_percentage = True
        else:
            df['value_to_plot'] = df['year_remaining'].astype(float)
            df['bar_color'] = df['value_to_plot'].apply(
                lambda x: '#ff0000' if x < 0 else '#32cd32'
            )
            x_label = 'Amount Remaining (Currency)'
            is_percentage = False

    df = df.sort_values('value_to_plot', ascending=True)
    categories = df['Category'] + " - " + df['Subcategory']
    values = df['value_to_plot']
    colors = df['bar_color']
    y_positions = np.arange(len(categories))

    fig, ax = plt.subplots(figsize=(16, 8))

    for i in range(len(categories)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=0)

    bars = ax.barh(y_positions, values, color=colors, zorder=2)
    max_val = max(values.max(), abs(values.min()))
    ax.set_xlim(-max_val * 1.3, max_val * 1.3)

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
    st.pyplot(fig)

# ------------------ STREAMLIT MAIN APP ------------------

st.set_page_config(page_title="Personal Finance Dashboard", layout="wide")
st.title("📊 Personal Budget Visualizer (from OneDrive)")

df = load_latest_budget_file()

if df is not None:
    st.success("Budget file loaded from OneDrive!")

    # Sidebar
    view_mode = st.sidebar.radio("View mode", ["Annual", "Specific Month"])
    if view_mode == "Annual":
        show_percentage = st.sidebar.checkbox("Show percentage view", value=True)
        plot_budget_status(df, show_percentage=show_percentage)
    else:
        month = st.sidebar.selectbox("Select Month", [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        plot_budget_status(df, month=month)

else:
    st.warning("No budget data available.")
