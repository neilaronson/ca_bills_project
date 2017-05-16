"""Get historical data tables of legislators and when they served from Wikipedia page.
Convert tables into pandas data frames and then add them to mySQL DB"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import get_sql

def pandafy_table(table):
    """Takes in an HTML table as BeautifulSoup object and outputs a pandas df version of the table"""
    all_rows = table.find_all("tr")[2:-2]
    results = make_tuple_list(all_rows)
    tbl = make_standard_table(results)
    return pd.DataFrame(tbl).set_index([0])

def make_tuple_list(all_rows):
    """Takes in list containing all rows of table. Each element of list is a
    BeautifulSoup object that contains all the columns.

    Outputs a list of tuples representing the table in the following form:
    for every row: (row number, td number, repeats(rowspan), text)
    """
    results = []
    for i, row in enumerate(all_rows):
        results_row = []
        for j, item in enumerate(row.find_all('td')):
            rowspan = 1
            if item.has_key("rowspan"):
                rowspan = item["rowspan"]
            results_row.append((i, j, int(rowspan), item.text))
        results.append(results_row)
    return results

def make_standard_table(list_of_tuples):
    """Takes the output of make_tuple_list (list of tuples) and reformats it to
    a list with the follwing form:
        [session, 1st, 2nd, 3rd...]
    where the numbers are the districts

    It accounts for uneven rowspans in the html table by copying the row above
    for the appropriate number of rows
    """

    final_results = []

    rowspans_above = [item[2] for item in list_of_tuples[0]]
    result_row0 = list_of_tuples[0]
    final_results.append([item[3] for item in result_row0])

    for i, result_row in enumerate(list_of_tuples):
        if i == 0:
            continue
        final_results_row = []

        new_spans = [item[2] for item in result_row]
        updated_spans = [span - 1 for span in rowspans_above]
        usc = list(updated_spans) #make copy so not changing list while iterating

        for col_index, span in enumerate(updated_spans):
            if span == 0: #time to change rowspan with current entry
                final_results_row.append(result_row.pop(0)[3])
                usc[col_index] = new_spans.pop(0)
            else:
                previous = final_results[i-1][col_index]
                final_results_row.append(previous)

        final_results.append(final_results_row)
        rowspans_above = usc

    return final_results

def main():
    page = requests.get("https://en.wikipedia.org/wiki/Members_of_the_California_State_Legislature").content
    soup = BeautifulSoup(page, "html.parser")
    senate_table = soup.find_all('table')[1]
    assembly_table = soup.find_all('table')[2]
    assembly_df = pandafy_table(assembly_table)
    senate_df = pandafy_table(senate_table)
    write_to_sql(assembly_df, senate_df)

if __name__ == '__main__':
    main()
