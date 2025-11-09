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

from personal_finances_utils import (
    load_money_manager_file,
    classify_expenses,
    compute_avg_expenses_per_month,
    plot_monthly_expenses_plotly,
)
