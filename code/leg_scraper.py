import requests
from bs4 import BeautifulSoup
import pandas as pd

page = requests.get("https://en.wikipedia.org/wiki/Members_of_the_California_State_Legislature").content

soup = BeautifulSoup(page, "html.parser")

senate_table = soup.find_all('table')[1]

headers = senate_table.find_all("tr")[1]

all_rows = senate_table.find_all("tr")[2:-2]

headers = [header.text for header in headers.find_all('td')]

# all_rows = [[item.text for item in row.find_all('td')] for row in all_rows]

results = []
for i, row in enumerate(all_rows):
    results_row = []
    for j, item in enumerate(row.find_all('td')):
        rowspan = 1
        if item.has_key("rowspan"):
            rowspan = item["rowspan"]
        results_row.append((i, j, int(rowspan), item.text))
    results.append(results_row)

final_results = []
# [session, 1st, 2nd, 3rd...]
rowspans_above = [item[2] for item in results[0]]
result_row0 = results[0]
final_results.append([item[3] for item in result_row0])

for i, result_row in enumerate(results):
    if i == 0:
        continue
    final_results_row = []

    new_spans = [item[2] for item in result_row]
    updated_spans = [span - 1 for span in rowspans_above]
    usc = list(updated_spans)
    for col_index, span in enumerate(updated_spans):
        if span == 0: #time to change rowspan with current entry
            final_results_row.append(result_row.pop(0)[3])
            usc[col_index] = new_spans.pop(0)
        else:
            previous = final_results[i-1][col_index]
            final_results_row.append(previous)
    final_results.append(final_results_row)

    rowspans_above = usc




# for every row:
# (row number, td number, repeats, text)

# ses
# for row in senate_table.find_all("tr"):
#     if row.find("td"):
#         print "new year"
#         print row.text.split('\n')
        # for col in row.find_all("td"):
        #     print col.text
