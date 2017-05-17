import pandas as pd
import get_sql
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.decomposition import NMF
import cPickle as pickle
from datetime import datetime

class DataPrep(object):
    """An object to clean and wrangle data into format for a model"""

    def __init__(self, query=None, filepath=None, training=True, predict=False, amendment_model=False):
        """Reads in data

        Args:
            filepath (str): location of file with csv data
        """
        if training:
            if query:
                self.df = get_sql.get_df(query)
            elif filepath:
                self.df = pd.read_csv(filepath)
            elif amendment_model:
                self.df = self.amendment_data()
            else: # default queries
                self.df = self.default_data()

    def default_data(self):
        bills_query = """SELECT b.bill_id, bv.bill_version_id, b.session_year, b.session_num, measure_type, bv.urgency, datediff(bv.bill_version_action_date,sd.start_date) as days_since_start, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee, b.passed
            FROM bill_tbl b
            left join bill_version_tbl bv
            on b.bill_id=bv.bill_id and bv.bill_version_id like '%INT'
            join start_dates sd on b.session_year=sd.session_year and b.session_num=sd.session_num
            where b.measure_type in ('AB' , 'SB') and b.session_year < '2015'"""
        bills_df = get_sql.get_df(bills_query)


        authors_query = """SELECT
                bv.bill_id, l.author_name, l.party
            FROM
                bill_version_authors_tbl bva
                    LEFT JOIN
                legislator_tbl l ON bva.name = l.author_name
                    AND l.session_year = bva.session_year
                    JOIN
                bill_version_tbl bv ON bv.bill_version_id = bva.bill_version_id
            WHERE
                contribution = 'LEAD_AUTHOR'
            		AND bv.bill_id < '2015'
                    AND bva.bill_version_id LIKE '%INT'
                    AND (bv.bill_id LIKE '%AB%'
                    OR bv.bill_id LIKE '%SB%')"""
        authors_df = get_sql.get_df(authors_query)
        authors_df = self.aggregate_authors_df(authors_df)

        earliest_version_query = """select bv.bill_id, bv.bill_xml from bill_version_tbl bv
            where bv.bill_version_id like '%INT'
            """
        text_df = get_sql.get_df(earliest_version_query)

        bill_authors_merged_df = pd.merge(bills_df, authors_df, on='bill_id')
        bill_authors_text_df = pd.merge(bill_authors_merged_df, text_df, on='bill_id')

        return bill_authors_text_df

    def amendment_data(self):
        bills_query = """SELECT b.bill_id, bv.bill_version_id, b.session_year, b.session_num, measure_type, bv.urgency, datediff(bv.bill_version_action_date,sd.start_date) as days_since_start, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee, b.passed
            FROM bill_tbl b
            join bill_version_tbl bv
            on b.bill_id=bv.bill_id and bv.bill_version_id like '%AMD'
            join start_dates sd on b.session_year=sd.session_year and b.session_num=sd.session_num
            where b.measure_type in ('AB' , 'SB') and b.session_year < '2015'"""
        bills_df = get_sql.get_df(bills_query)
        return bills_df

    def aggregate_authors_df(self, authors_df):
        authors_df['party'] = authors_df['party'].fillna('COM')
        party_df = authors_df[['bill_id', 'party']].groupby('bill_id').agg(agg_parties)
        cosponsor_count_df = authors_df[['bill_id', 'party']].groupby('bill_id').count()
        cosponsor_count_df = cosponsor_count_df.rename(columns={'party': 'n_authors'})
        cosponsor_count_df['committee'] = (cosponsor_count_df['n_authors']==0).astype(int)
        merged_df = pd.merge(party_df, cosponsor_count_df, left_index=True, right_index=True).reset_index()
        return merged_df

    def dummify(self, columns, regression=False):
        """Create dummy columns for categorical variables"""
        if regression:
            dummies = pd.get_dummies(self.df[columns], columns=columns,
                                    prefix=columns, drop_first=True)
        else:
            dummies = pd.get_dummies(self.df[columns], columns=columns,
                                    prefix=columns)
        self.df = self.df.drop(columns, axis=1)
        self.df = pd.concat([self.df,dummies], axis=1)

    def drop_some_cols(self, columns):
        """Simply drop columns (as list) from the dataframe"""
        for col in columns:
            self.df = self.df.drop(col,axis=1)

    def drop_na(self):
        """Generic method to drop all rows with NA's in any column."""
        before = self.df.shape[0]
        self.df = self.df.dropna(axis=0, how='any')
        after = self.df.shape[0]
        self.df = self.df.reset_index()
        print "dropped {} rows".format(before-after)

    def bucket_vote_required(self):
        """Separate all bills that require a simple majority from other types of bills that require more than
        a simple majority. There should really only be 2/3, so this function adds the
        handful of other vote types together"""
        self.df['vote_required'] = self.df['vote_required'].apply(lambda vote: 0 if vote=='Majority' else 1)

    def make_session_type(self):
        """Uses session_num to make a column indicating if it was regular session (0) or Extraordinary session(1)"""
        self.df['session_type'] = self.df['session_num'].apply(lambda session: 0 if session=='0' else 1)

    # def get_latent_topic_mat(self, n_components, use_cached_tfidf, cache_tfidf, X_data=None):
    #     tfidf_mat =
    #     nmf_mat = self.get_nmf_mat(tfidf_mat, n_components)
    #     print "nmf complete"
    #     return nmf_mat

    def run_tfidf(self, use_cached_tfidf, cache_tfidf, X_data=None, **kwargs):
        """Apply TFIDF and get back transformed matrix"""
        if use_cached_tfidf:
            with open(use_cached_tfidf) as p:
                tfidf_contents = pickle.load(p)
                tfidf_mat = tfidf_contents[1]
            print "loaded tfidf"
        else: #not using a cached tfidf, will have to generate
            tfidf = TfidfVectorizer(**kwargs)
            tfidf_mat = tfidf.fit_transform(X_data)
            if cache_tfidf:
                current_time = datetime.now().strftime(format='%m-%d-%y-%H-%M')
                filename = "../data/cached_tfidf_"+current_time+".pkl"
                with open(filename, 'w') as p:
                    pickle.dump([tfidf, tfidf_mat], p)
            print "tfidf complete"
        return tfidf_mat

    def get_nmf_mat(self, X_data, n_components):
        nmf = NMF(n_components=n_components)
        W = nmf.fit_transform(X_data)
        return W

    def process_text(self, column_name, field, use_cached_processing=None, cache_processing=False):
        """Run each text row of column_name through BS to extract content from the specified XML field"""
        if use_cached_processing:
            with open(use_cached_processing) as p:
                bill_content = pickle.load(p)
                print "loaded processed text"
        else:
            bill_soup = self.df[column_name].values
            bill_content = [self.get_bill_text(soup, field) for soup in bill_soup]
            if cache_processing:
                current_time = datetime.now().strftime(format='%m-%d-%y-%H-%M')
                filename = "../data/cached_processed_text_"+current_time+".pkl"
                with open(filename, 'w') as p:
                    pickle.dump(bill_content, p)
            print "processed text"
        return bill_content


    def get_bill_text(self, xml, field):
        """Finds all tags of field in given xml and returns them as one string
        separated by space if there's more than one"""
        soup = BeautifulSoup(xml, "xml")
        results = [raw.text for raw in soup.find_all(field)]
        text = " ".join(results)
        return text

    def process_and_tfidf(self, use_cached_processing=None, use_cached_tfidf=None, cache_processing=False, cache_tfidf=False, **kwargs):
        if cache_tfidf:  #make sure to cache processing if caching tfidf
            cache_processing=True
        if not use_cached_tfidf:
            X = self.process_text('bill_xml', 'Content', use_cached_processing, cache_processing)
            tfidf_mat = self.run_tfidf(use_cached_tfidf, cache_tfidf, X_data=X, **kwargs)
        else: #using cache, don't need to process
            tfidf_mat = self.run_tfidf(use_cached_tfidf, cache_tfidf)
        return tfidf_mat

    def add_latent_topics(self, n_components, use_cached_processing=None, use_cached_tfidf=None, cache_processing=False, cache_tfidf=False):
        tfidf_mat = self.process_and_tfidf(use_cached_processing, use_cached_tfidf, cache_processing, cache_tfidf, tokenizer=tokenize, stop_words='english', max_features=2000)
        ltm = self.get_nmf_mat(tfidf_mat, n_components)

        # ltm_df = pd.DataFrame(ltm, index=range(self.df.shape[0]))
        # self.df.index=range(self.df.shape[0])
        # newdf = pd.concat([self.df, ltm_df], axis=1)
        ltm_df = pd.DataFrame(ltm)
        self.df = pd.concat([self.df, ltm_df], axis=1)

    def random_subset(self, nrows_to_keep):
        np.random.seed(123)
        keepers = np.random.choice(range(self.df.shape[0]), size=nrows_to_keep, replace=False)
        self.df = self.df.iloc[keepers,:]

    def prepare(self, regression=False, predict=False, test=False, n_components=2):
        """Executes all cleaning methods in proper order. If regression, remove one
        dummy column and scale numeric columns for regularization"""
        self.drop_na()
        self.make_session_type()
        self.df = self.df[['party', 'passed', 'bill_xml']]
        if regression:
            self.dummify(['party'], regression=True)
        else:
            self.dummify(['party'])
        # if regression:
        #     self.dummify(['urgency', 'taxlevy', 'appropriation', 'party'], regression=True)
        # else:
        #     self.dummify(['urgency', 'taxlevy', 'appropriation', 'party'])
        # self.bucket_vote_required()

        # add latent topics
        self.add_latent_topics(n_components, use_cached_tfidf='../data/cached_tfidf_05-17-17-02-39.pkl')

        # todrop = [u'bill_id', u'session_year', u'session_num', u'measure_type', u'fiscal_committee', u'bill_version_id', u'bill_xml']
        todrop = [u'bill_xml']
        self.drop_some_cols(todrop)

        y = self.df.pop('passed').values
        X = self.df.values

        return X, y

def agg_parties(list_of_parties):
    """All Dem, Repub, Both, or Committee"""
    list_of_parties = list_of_parties.tolist()
    if all(party=="DEM" for party in list_of_parties):
        return "ALL_DEM"
    elif all(party=="REP" for party in list_of_parties):
        return "ALL_REP"
    elif all(party=="COM" for party in list_of_parties):
        return "COM"
    else:
        return "BOTH"

def tokenize(text):
    """Tokenize and stem a block of text"""
    bill_content = TextBlob(text).lower()
    bill_words = bill_content.words
    bill_words_stemmed = [wordlist.stem() for wordlist in bill_words]
    return bill_words_stemmed
