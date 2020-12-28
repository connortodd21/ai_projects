import csv
import sys

######################################################
# 
# You'll need to download the csv files at https://www.cdc.gov/nchs/nvss/vsrr/covid19/excess_deaths.htm#data-tables
#
######################################################

"""
labels: need the index of data for Jurisdiction, Type, Date, Deaths, Year
"""
def count_excess_deaths(data, data_type, labels, month_start):
    deaths_before_2020 = 0
    deaths_in_2020 = 0
    list_of_years = []
    i = 0
    for row in data:
        if row[labels["Year"]] not in list_of_years:
            list_of_years.append(row[labels["Year"]])
        if row[labels["Jurisdiction"]] == "United States" and row[labels["Type"]] == data_type:
            deaths_str = row[labels["Deaths"]]
            month = row[labels["Date"]][month_start:month_start+2]
            if month == "12":
                continue
            if deaths_str != '':
                deaths = int(deaths_str)
                if row[7] != "2020":
                    # look through past years data, calculate average deaths per year before 2020
                    deaths_before_2020 += int(deaths)
                else:
                    # 2020 excess deaths
                    deaths_in_2020 += int(deaths)
            i += 1
    deaths_before_2020 /= len(list_of_years) - 1
    return int(deaths_in_2020 - deaths_before_2020)

"""
Data tyes:
Unweighted
Predicted (weighted)
"""

with open('/Users/connortodd/personal_projects/ai_projects/covid19/excess_deaths/Weekly_counts_of_deaths_by_jurisdiction_and_age_group.csv', newline='',  encoding='utf-8-sig') as csvfile:
    weekly_count_excess_deaths = csv.reader(csvfile, delimiter=',')
    weekly_labels = next(weekly_count_excess_deaths)
    labels = {
        "Jurisdiction": weekly_labels.index("Jurisdiction"), 
        "Type": weekly_labels.index("Type"), 
        "Date": weekly_labels.index("Week Ending Date"), 
        "Deaths": weekly_labels.index("Number of Deaths"), 
        "Year": weekly_labels.index("Year")
        }
    print(f"\n\nExcess deaths from Januray to November using weekly deaths by jurisdiction data: {count_excess_deaths(weekly_count_excess_deaths, 'Unweighted', labels, 0)}\n\n")
    print("data current as of 12/27/20")