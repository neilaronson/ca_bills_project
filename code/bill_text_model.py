import get_sql
import pandas as pd
from bs4 import BeautifulSoup
from textblob import TextBlob
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

def get_bill_text(xml, field):
    soup = BeautifulSoup(xml, "xml")
    results = [raw.text for raw in soup.find_all(field)]
    text = " ".join(results)
    return text

def process(df, column_name):
    bill_soup = df[column_name].values
    bill_content = [TextBlob(get_bill_text(soup, 'Content')).lower() for soup in bill_soup]
    bill_words = [bill.words for bill in bill_content]
    bill_words_stemmed = [wordlist.stem() for wordlist in bill_words]
    return bill_words_stemmed

def process2(df, column_name):
    bill_soup = df[column_name].values
    bill_content = [get_bill_text(soup, 'Content') for soup in bill_soup]
    return bill_content

def tokenize(text):
    bill_content = TextBlob(text).lower()
    bill_words = bill_content.words
    bill_words_stemmed = [wordlist.stem() for wordlist in bill_words]
    return bill_words_stemmed

query = """select earliest.bill_id, bv.bill_version_id as earliest_bvid, bill_xml, passed from
(select bill_id, min(bill_version_action_date) as earliest_date from bill_version_tbl
group by bill_id) earliest
join bill_version_tbl bv on (earliest.bill_id=bv.bill_id and earliest.earliest_date=bv.bill_version_action_date)
join bill_tbl b on earliest.bill_id=b.bill_id"""

df = get_sql.get_df(query)
content = process2(df, 'bill_xml')
y = df.passed.values

#list_of_processed_bills = process(df, 'bill_xml')

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfidf_mat = tfidf.fit_transform(content, y)

nb = MultinomialNB()
nb.fit(tfidf_mat, y)
print cross_val_score(nb, tfidf_mat, y, scoring='f1')
