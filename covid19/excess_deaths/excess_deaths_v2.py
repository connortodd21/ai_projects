import csv
import sys
import os
import errno
import pathlib
from datetime import datetime
import pandas as pd

list_of_states = ["Alabama", "Alaska", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming", "United States"]
list_of_states_temp = list_of_states[0:11]
list_of_months = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
list_of_years = list(range(2015, 2022))

######################################################
# 
# Data pulled from https://www.cdc.gov/nchs/nvss/vsrr/covid19/excess_deaths.htm#data-tables
#
######################################################



"""
Plan: 
init_data_dict()
For each row:
    get the month, year, state, and data_type
    if year is before target_year:
        update data_dict[state][year][month][age_group][previous_deaths]
        update data_dict[state][year][age_group][previous_deaths]
        update data_dict[state][year][week][age_group][previous_deaths]
    elif year is target_year:
        update data_dict[state][year][month][current_deaths]
        update data_dict[state][year][age_group][current_deaths]
        update data_dict[state][year][week][age_group][current_deaths]
"""

def init_data_dict(columns):
    data = {}
    for state in list_of_states:
        data[state] = {}
        for year in list_of_years:
            data[state][year] = {}
            for month in list_of_months.values():
                data[state][year][month] = {}
            for 
    return data

def count_excess_deaths(data, data_type, target_year, columns, important_columns):
    data_dict = init_data_dict(columns)
    return 0

"""
Data tyes:
Unweighted
Predicted (weighted)
"""

def run(file, year_input, starting_month, stopping_month, data_type):
    df = pd.read_csv(file)
    columns = df.columns.to_list()


csv_path = "/Users/connortodd/personal_projects/ai_projects/covid19/excess_deaths/new_data.csv"
run(csv_path, 2020, 3, 6, "Predicted (weighted)")

######################################################
# 
# Future goal: Make more modular
# Do one pass through the data and calculate everything in one pass (death by state by year, month, age group, etc)
# One giant pass to calculate a bunch of stats (make helper functions for each)
# Right now each run() call traverses the entire dataset, super inefficient. Make it so that we can just add more helper functions
# Either that or be more precise about how to traverse the dataset
#
# Stretch goal:
# Add some statistical analysis to try and predict future data (maybe a RNN if possible?)
#
######################################################