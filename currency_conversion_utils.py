import pandas as pd
from datetime import timedelta
from datetime import datetime
import json
import os

def get_previous_monday(date):
    # Check if the input is a string and convert it to a date object
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    
    # Check if the given date is a Monday (weekday() returns 0 for Monday)
    if date.weekday() == 0:
        return date.date()  # Return just the date part
    else:
        # Calculate the previous Monday
        days_since_monday = date.weekday()
        previous_monday = date - timedelta(days=days_since_monday)
        return previous_monday.date()  # Return just the date part


# Function to get the NOK conversion using the previous Monday's exchange rate
def get_nok_conversion(row, exchange_rates):
    if row['Currency'] != 'NOK':
        previous_monday = get_previous_monday(row['Date'])
        previous_monday_str = str(previous_monday)
        if previous_monday_str in exchange_rates:
            rate = exchange_rates[previous_monday_str]
            return row['EUR'] * rate
    return row['Amount']  # For NOK or if no conversion, return 'Amount'

def load_exchange_rates(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {}