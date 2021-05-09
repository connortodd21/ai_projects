import csv
import sys
import os
import errno
import pathlib
from datetime import datetime

list_of_states = ["Alabama", "Alaska", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming", "United States"]
list_of_states_temp = list_of_states[0:11]
list_of_months = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
######################################################
# 
# Data pulled from https://www.cdc.gov/nchs/nvss/vsrr/covid19/excess_deaths.htm#data-tables
#
######################################################

"""
labels: need the index of data for Jurisdiction, Type, Date, Deaths, Year
"""
def count_excess_deaths(data, data_type, labels, month_start, year_start, state, excess_deaths_year, months):
    deaths_before_year = 0
    deaths_in_year = 0
    deaths_per_year = {}
    list_of_years = []
    i = 0
    for row in data:
        year = row[labels["Date"]][6:11]
        month = int(row[labels["Date"]][month_start:month_start+2])
        if year not in list_of_years:
            list_of_years.append(year)
            deaths_per_year[year] = 0
        if row[labels["Jurisdiction"]] == state and row[labels["Type"]] == data_type:
            deaths_str = row[labels["Deaths"]]
            if deaths_str != '':
                deaths = int(deaths_str)
                if int(year) < 2020:
                    # look through past years (2015-2019) data, calculate deaths per year before 2020 as baseline
                    if month in months:
                        deaths_per_year[year] += int(deaths)
                        deaths_before_year += int(deaths)
                elif int(year) == excess_deaths_year:
                    # 2020 excess deaths
                    if month in months:
                        deaths_in_year += int(deaths)
                        deaths_per_year[year] += int(deaths)
        i += 1
    deaths_before_year /= max(sum(int(yr) < 2020 for yr in list_of_years), 1)
    return (int(deaths_in_year - deaths_before_year), deaths_in_year / deaths_before_year, deaths_per_year, list_of_years)

"""
Data tyes:
Unweighted
Predicted (weighted)
"""

def run(file, year_input, starting_month, stopping_month, data_type):
    currentMonth = datetime.now().month
    with open(csv_path, newline='',  encoding='utf-8-sig') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        weekly_labels = next(data)
        labels = {
            "Jurisdiction": weekly_labels.index("Jurisdiction"), 
            "Type": weekly_labels.index("Type"), 
            "Date": weekly_labels.index("Week Ending Date"), 
            "Deaths": weekly_labels.index("Number of Deaths"), 
            }
        weight = data_type
        results_list = []
        for state in list_of_states:
            # print(f"Calculating {weight} statistics for {state}")
            excess_deaths, excess_death_percent, deaths_per_year, list_of_years = count_excess_deaths(data, weight, labels, 0, 6, state, year_input, list(range(starting_month,stopping_month + 1)))
            # excess_deaths, excess_death_percent, deaths_per_year, list_of_years = old_count_excess_deaths(data, weight, labels, 0, state)
            results_list.append({"state": state, "data": [excess_deaths, excess_death_percent]})
            fname = pathlib.Path(csv_path)
            assert fname.exists(), f'No such file: {fname}'
            csvfile.seek(0)
            next(data)
        
        results_sorted_by_raw_excess_deaths = sorted(results_list, key=lambda k: k["data"][0], reverse=True)
        results_sorted_by_excess_death_percent = sorted(results_list, key=lambda k: k["data"][1], reverse=True)
        dash = '\u2500' * 170
        if starting_month == stopping_month:
            output_file_name = f'results/{year_input}_{list_of_months[starting_month]}_excess_deaths.txt'
        else:
            output_file_name = f'results/{year_input}_{list_of_months[starting_month]}_to_{list_of_months[stopping_month]}_excess_deaths.txt'
        if not os.path.exists(os.path.dirname(output_file_name)):
            try:
                os.makedirs(os.path.dirname(output_file_name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(output_file_name, 'w') as output_file:
            sys.stdout = output_file
            if starting_month == stopping_month:
                print(f"\n{weight} excess deaths in {year_input} during {list_of_months[starting_month]}\n")
            else:
                print(f"\n{weight} excess deaths in {year_input} between {list_of_months[starting_month]} and {list_of_months[stopping_month]}\n")
            print(f"NOTE: {list_of_months[(currentMonth+11) % 12] if currentMonth > 1 else list_of_months[(currentMonth+11)] } and {list_of_months[currentMonth]} DATA LIKELY INCOMPLETE. ANY STATE WITH 0.0% CHANGE IN EXCESS DEATHS HAS NOT REPORTED ANY DATA\n")
            print("{a:<55s}{c:^20s}{i:^5s}{c:^20s}{b:<60s}".format(a="States sorted by excess deaths", b="States sorted by excess death percent increase", c="|", i="i"))
            print(f"{dash}")
            i = 1
            for (raw_state), (pct_state) in zip(results_sorted_by_raw_excess_deaths, results_sorted_by_excess_death_percent):
                print("{:<55s}{:^20s}{i:^5d}{:^20s}{:<60s}".format(f"{raw_state['state']} had {raw_state['data'][0]} excess deaths ({round(raw_state['data'][1],2)}% change)", "|", "|" ,f"{pct_state['state']} had {pct_state['data'][0]} excess deaths ({round(pct_state['data'][1],2)}% change)", i=i))
                i+=1

            print(f"\n\ndata current as of {datetime.fromtimestamp(fname.stat().st_ctime)}")

csv_path = "/Users/connortodd/personal_projects/ai_projects/covid19/excess_deaths/new_data.csv"
# csv_path = "/Users/connortodd/personal_projects/ai_projects/covid19/excess_deaths/Weekly_counts_of_deaths_by_jurisdiction_and_age_group.csv"

# run(csv_path, 2020, 1, 12, "Predicted (weighted)")
# run(csv_path, 2020, 1, 10, "Predicted (weighted)")
# run(csv_path, 2021, 1, 1, "Predicted (weighted)")
# run(csv_path, 2020, 3, 12, "Predicted (weighted)")
# run(csv_path, 2020, 4, 12, "Predicted (weighted)")
# run(csv_path, 2020, 3, 6, "Predicted (weighted)")
# run(csv_path, 2020, 6, 12, "Predicted (weighted)")
# run(csv_path, 2020, 12, 12, "Predicted (weighted)")

run(csv_path, 2021, 1, 2, "Predicted (weighted)")

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