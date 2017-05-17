from data_prep import DataPrep
from textblob import TextBlob

def store_tfidf(**kwargs):
    prep = DataPrep(filepath='../data/full_data.csv')
    prep.prepare(cache_tfidf=True, **kwargs)

def tokenize(text):
    """Tokenize and stem a block of text"""
    bill_content = TextBlob(text).lower()
    bill_words = bill_content.words
    bill_words_stemmed = [wordlist.stem() for wordlist in bill_words]
    return bill_words_stemmed

def main():
    tfidf_params = {'stop_words':'english', 'max_features':8000, 'max_df':.8, 'ngram_range':(1,3)}
    store_tfidf(**tfidf_params)

if __name__ == '__main__':
    main()
