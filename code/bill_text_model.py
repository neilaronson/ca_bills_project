from timeit import default_timer as timer
start = timer()

import get_sql
import pandas as pd
from bs4 import BeautifulSoup

def get_bill_content(xml):
    soup = BeautifulSoup(xml, "xml")
    results = [raw.text for raw in soup.find_all('Content')]
    text = " ".join(results)
    return text

query = """select earliest.bill_id, bv.bill_version_id as earliest_bvid, bill_xml from
(select bill_id, min(bill_version_action_date) as earliest_date from bill_version_tbl
group by bill_id) earliest
join bill_version_tbl bv on (earliest.bill_id=bv.bill_id and earliest.earliest_date=bv.bill_version_action_date)"""

df = get_sql.get_df(query)
df['bill_content'] = df['bill_xml'].apply(get_bill_content)

end = timer()
print end-start
