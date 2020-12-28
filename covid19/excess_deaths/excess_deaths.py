import csv
import sys

######################################################
# 
# You'll need to download the csv files at https://www.cdc.gov/nchs/nvss/vsrr/covid19/excess_deaths.htm#data-tables
#
######################################################

"""
labels: need the index of data for state, weight, date, deaths
"""
def count_excess_deaths(data, data_type, labels, month_start):
    deaths_in_month = {}
    deaths_in_month[data_type] = {}
    deaths_in_month_2020 = {}
    deaths_in_month_2020[data_type] = {}
    excess_deaths_2020 = {}
    i = 0
    for row in data:
        if row[labels[0]] == "United States" and row[labels[1]] == data_type:
            deaths_str = row[labels[3]]
            month = row[labels[2]][month_start:month_start+2]
            if month == "12":
                continue
            if deaths_str != '':
                deaths = int(deaths_str)
                if row[7] != "2020":
                    # look through past years data, calculate average deaths per year before 2020
                    if month in deaths_in_month[data_type].keys():
                        deaths_in_month[data_type][month] += int(deaths)
                    else:
                        deaths_in_month[data_type][month] = int(deaths)
                else:
                    # 2020 excess deaths
                    if month in deaths_in_month_2020[data_type].keys():
                        deaths_in_month_2020[data_type][month] += int(deaths)
                    else:
                        deaths_in_month_2020[data_type][month] = int(deaths)
            i += 1
    display_weighted_data = data_type
    for month, deaths in deaths_in_month[display_weighted_data].items():
        deaths_in_month[display_weighted_data][month] /= 5
    for (month1, _), (month2, _) in zip(deaths_in_month[display_weighted_data].items(), deaths_in_month_2020[display_weighted_data].items()):
        excess_deaths_2020[month1] = int(deaths_in_month_2020[display_weighted_data][month2] - deaths_in_month[display_weighted_data][month1])
    return sum(excess_deaths_2020.values())

"""
Data tyes:
Unweighted
Predicted (weighted)
"""

with open('/Users/connortodd/personal_projects/ai_projects/covid19/excess_deaths/Weekly_counts_of_deaths_by_jurisdiction_and_age_group.csv', newline='',  encoding='utf-8-sig') as csvfile:
    weekly_count_excess_deaths = csv.reader(csvfile, delimiter=',')
    weekly_labels = next(weekly_count_excess_deaths)
    labels = [weekly_labels.index("Jurisdiction"), weekly_labels.index("Type"), weekly_labels.index("Week Ending Date"), weekly_labels.index("Number of Deaths")]
    print(f"\n\nExcess deaths from Januray to November using weekly deaths by jurisdiction data: {count_excess_deaths(weekly_count_excess_deaths, 'Unweighted', labels, 0)}\n\n")
