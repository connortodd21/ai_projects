import csv
import sys
import pathlib
import datetime

list_of_states = ["Alaska", "Alabama", "Arkansas", "Arizona", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming", "United States"]
list_of_states_temp = list_of_states[0:11]
######################################################
# 
# You'll need to download the csv files at https://www.cdc.gov/nchs/nvss/vsrr/covid19/excess_deaths.htm#data-tables
#
######################################################

"""
labels: need the index of data for Jurisdiction, Type, Date, Deaths, Year
"""
def count_excess_deaths(data, data_type, labels, month_start, state):
    deaths_before_2020 = 0
    deaths_in_2020 = 0
    deaths_per_year = {}
    list_of_years = []
    i = 0
    for row in data:
        year = row[labels["Year"]]
        month = row[labels["Date"]][month_start:month_start+2]
        if year not in list_of_years:
            list_of_years.append(year)
            deaths_per_year[year] = 0
        if row[labels["Jurisdiction"]] == state and row[labels["Type"]] == data_type:
            deaths_str = row[labels["Deaths"]]
            if deaths_str != '':
                deaths = int(deaths_str)
                if year != "2020":
                    # look through past years data, calculate deaths per year before 2020
                    deaths_per_year[year] += int(deaths)
                    if month != "12":
                        deaths_before_2020 += int(deaths)
                else:
                    # 2020 excess deaths
                    if month != "12":
                        deaths_in_2020 += int(deaths)
                        deaths_per_year[year] += int(deaths)
        i += 1
    deaths_before_2020 /= max(len(list_of_years) - 1, 1)
    return (int(deaths_in_2020 - deaths_before_2020), deaths_in_2020 / deaths_before_2020, deaths_per_year, list_of_years)

"""
Data tyes:
Unweighted
Predicted (weighted)
"""

csv_path = "/Users/connortodd/personal_projects/ai_projects/covid19/excess_deaths/Weekly_counts_of_deaths_by_jurisdiction_and_age_group.csv"

with open(csv_path, newline='',  encoding='utf-8-sig') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    weekly_labels = next(data)
    labels = {
        "Jurisdiction": weekly_labels.index("Jurisdiction"), 
        "Type": weekly_labels.index("Type"), 
        "Date": weekly_labels.index("Week Ending Date"), 
        "Deaths": weekly_labels.index("Number of Deaths"), 
        "Year": weekly_labels.index("Year")
        }
    weight = "Unweighted"
    results_list = []
    for state in list_of_states:
        # print(f"Calculating {weight} statistics for {state}")
        excess_deaths, excess_death_percent, deaths_per_year, list_of_years = count_excess_deaths(data, weight, labels, 0, state)
        results_list.append({"state": state, "data": [excess_deaths, excess_death_percent]})
        fname = pathlib.Path(csv_path)
        assert fname.exists(), f'No such file: {fname}'
        csvfile.seek(0)
        next(data)
    
    results_sorted_by_raw_excess_deaths = sorted(results_list, key=lambda k: k["data"][0], reverse=True)
    results_sorted_by_excess_death_percent = sorted(results_list, key=lambda k: k["data"][1], reverse=True)
    dash = '\u2500' * 170
    print("{a:<55s}{c:^20s}{i:^5s}{c:^20s}{b:<60s}".format(a="States sorted by excess deaths", b="States sorted by excess death percent increase", c="|", i="i"))
    print(f"{dash}")
    i = 1
    for (raw_state), (pct_state) in zip(results_sorted_by_raw_excess_deaths, results_sorted_by_excess_death_percent):
        print("{:<55s}{:^20s}{i:^5d}{:^20s}{:<60s}".format(f"{raw_state['state']} had {raw_state['data'][0]} excess deaths ({round(raw_state['data'][1],2)}% change)", "|", "|" ,f"{pct_state['state']} had {pct_state['data'][0]} excess deaths ({round(pct_state['data'][1],2)}% change)", i=i))
        i+=1
    print(f"\n\ndata current as of {datetime.datetime.fromtimestamp(fname.stat().st_ctime)}")