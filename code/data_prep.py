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
                bv.bill_id, l.author_name, l.party, l.district, bva.session_year
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

        n_amendments_query = """select bv1.bill_version_id, bv1.bill_id, count(bv2.bill_version_id) as n_prev_versions from bill_version_tbl bv1
            join bill_version_tbl bv2 on bv1.bill_id=bv2.bill_id
            where bv1.bill_version_id like '%AMD' and bv2.version_num > bv1.version_num
            group by bv1.bill_version_id"""
        n_amendments_df = get_sql.get_df(n_amendments_query)

        merged_df = pd.merge(bills_df, n_amendments_df, on='bill_version_id')
        merged_df = merged_df.drop('bill_id_y', axis=1)
        merged_df = merged_df.rename(columns={'bill_id_x': 'bill_id'})
        merged_df = self.add_authors(merged_df)

        return merged_df

    def add_authors(self, df):
        authors_amendment_query = """SELECT
                bv.bill_id, bv.bill_version_id, bva.session_year, l.author_name, l.legislator_name, l.district, l.party
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
                    AND bva.bill_version_id LIKE '%AMD'
                    AND (bv.bill_id LIKE '%AB%'
                    OR bv.bill_id LIKE '%SB%')"""

        authors_amendment_df = get_sql.get_df(authors_amendment_query)
        authors_amendment_df['party'] = authors_amendment_df['party'].fillna('COM')
        authors_amendment_seniority_df = self.add_seniority(authors_amendment_df)

        authors_amendment_seniority_df = authors_amendment_seniority_df.drop(['bill_id', 'session_year', 'district', 'author_name', 'legislator_name', 'name'], axis=1)
        authors_amendment_seniority_df = authors_amendment_seniority_df.groupby('bill_version_id').agg({'party': agg_parties, 'nterms': 'mean'}).reset_index()
        authors_amendment_seniority_df.nterms[authors_amendment_seniority_df['nterms'].isnull()] = -1000
        return pd.merge(df, authors_amendment_seniority_df, on='bill_version_id')

    def add_seniority(self, df):

        all_seniority = self.make_seniority_data()


        return pd.merge(df, all_seniority, on=['district', 'session_year'], how='left')

    def make_seniority_data(self):
        assembly_df = pd.read_csv('../data/assembly.csv')
        assembly_df = assembly_df.rename(columns={'0': 'session_year'})
        assembly_df_melted = pd.melt(assembly_df, id_vars='session_year', var_name='district', value_name='name')
        assembly_joined = pd.merge(assembly_df_melted, assembly_df_melted, on='name')
        assembly_joined_conditioned = assembly_joined[assembly_joined.session_year_x >= assembly_joined.session_year_y]
        assembly_seniority = assembly_joined_conditioned.groupby(['session_year_x', 'district_x', 'name']).count()['session_year_y'].reset_index()
        assembly_seniority = assembly_seniority.rename(columns={'session_year_x': 'session_year', 'district_x': 'district', 'session_year_y':'nterms'})
        assembly_seniority['district'] = 'AD' + assembly_seniority['district']
        assembly_seniority['district'] = assembly_seniority['district'].apply(lambda x: x[:2]+str(0)+x[2] if len(x)==3 else x)
        assembly_seniority['session_year'] = assembly_seniority['session_year'].str.replace('-', '')

        senate_df = pd.read_csv('../data/senate.csv')
        senate_df = senate_df.iloc[21:]
        senate_df = senate_df.rename(columns={'0': 'session_year'})
        senate_df_melted = pd.melt(senate_df, id_vars='session_year', var_name='district', value_name='name')
        senate_joined = pd.merge(senate_df_melted, senate_df_melted, on='name')
        senate_joined_conditioned = senate_joined[senate_joined.session_year_x >= senate_joined.session_year_y]
        senate_seniority = senate_joined_conditioned.groupby(['session_year_x', 'district_x', 'name']).count()['session_year_y'].reset_index()
        senate_seniority = senate_seniority.rename(columns={'session_year_x': 'session_year', 'district_x': 'district', 'session_year_y':'nterms'})
        senate_seniority['district'] = 'SD' + senate_seniority['district']
        senate_seniority['district'] = senate_seniority['district'].apply(lambda x: x[:2]+str(0)+x[2] if len(x)==3 else x)
        senate_seniority['session_year'] = senate_seniority['session_year'].str.replace('-', '')

        all_seniority = pd.concat([assembly_seniority, senate_seniority])

        aswd = all_seniority[['session_year', 'name', 'nterms']]
        wo_district = pd.merge(aswd, aswd, on='name')
        wo_district = wo_district[wo_district.session_year_x >= wo_district.session_year_y]
        total_seniority = wo_district.groupby(['session_year_x', 'name']).count()['session_year_y'].reset_index()
        total_seniority = total_seniority.rename(columns={'session_year_x': 'session_year', 'session_year_y':'nterms'})

        final_seniority = pd.merge(all_seniority, total_seniority, on=['name', 'session_year'])
        final_seniority = final_seniority.rename(columns={'nterms_x': 'nterms_in_house', 'nterms_y': 'nterms'})
        return final_seniority


    def aggregate_authors_df(self, authors_df):
        all_seniority = self.make_seniority_data()

        authors_df['party'] = authors_df['party'].fillna('COM')
        authors_party_seniority_df = pd.merge(authors_df, all_seniority, on=['district', 'session_year'], how='left')

        grouped_party_seniority = authors_party_seniority_df[['bill_id', 'party', 'nterms']].groupby('bill_id')
        grouped_party_seniority = grouped_party_seniority.agg({'party': agg_parties, 'nterms': 'mean'}).reset_index()

        grouped_party_seniority.nterms[grouped_party_seniority['nterms'].isnull()] = -1000

        # cosponsor_count_df = authors_df[['bill_id', 'party']].groupby('bill_id').count()
        # cosponsor_count_df = cosponsor_count_df.rename(columns={'party': 'n_authors'})
        # cosponsor_count_df['committee'] = (cosponsor_count_df['n_authors']==0).astype(int)
        # merged_df = pd.merge(party_df, cosponsor_count_df, left_index=True, right_index=True).reset_index()
        return grouped_party_seniority

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
        if (before-after) > 0:
            self.df = self.df.reset_index()
            self.df = self.df.drop('index', axis=1)
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

    def run_tfidf(self, use_cached_tfidf, cache_tfidf, X_data=None, **tfidfargs):
        """Apply TFIDF and get back transformed matrix"""
        if use_cached_tfidf:
            with open(use_cached_tfidf) as p:
                tfidf_contents = pickle.load(p)
                tfidf_mat = tfidf_contents[1]
            print "loaded tfidf"
        else: #not using a cached tfidf, will have to generate
            tfidf = TfidfVectorizer(tokenizer=tokenize, **tfidfargs)
            tfidf_mat = tfidf.fit_transform(X_data)
            if cache_tfidf:
                current_time = datetime.now().strftime(format='%m-%d-%y-%H-%M')
                filename = "../data/cached_tfidf_"+current_time+".pkl"
                with open(filename, 'w') as p:
                    pickle.dump([tfidf, tfidf_mat], p)
                print "pickled tfidf file at {}".format(filename)
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
                print "pickled processed text file at {}".format(filename)
            print "processed text"
        return bill_content


    def get_bill_text(self, xml, field):
        """Finds all tags of field in given xml and returns them as one string
        separated by space if there's more than one"""
        soup = BeautifulSoup(xml, "xml")
        results = [raw.text for raw in soup.find_all(field)]
        text = " ".join(results)
        return text

    def process_and_tfidf(self, use_cached_processing=None, use_cached_tfidf=None, cache_processing=False, cache_tfidf=False, **tfidfargs):
        if cache_tfidf:  #make sure to cache processing if caching tfidf
            cache_processing=True
        if not use_cached_tfidf:
            X = self.process_text('bill_xml', 'Content', use_cached_processing, cache_processing)
            tfidf_mat = self.run_tfidf(use_cached_tfidf, cache_tfidf, X_data=X, **tfidfargs)
        else: #using cache, don't need to process
            tfidf_mat = self.run_tfidf(use_cached_tfidf, cache_tfidf)
        return tfidf_mat

    def add_latent_topics(self, n_components, use_cached_processing=None, use_cached_tfidf=None, cache_processing=False, cache_tfidf=False, **tfidfargs):
        tfidf_mat = self.process_and_tfidf(use_cached_processing, use_cached_tfidf, cache_processing, cache_tfidf, **tfidfargs)
        ltm = self.get_nmf_mat(tfidf_mat, n_components)
        ltm_df = pd.DataFrame(ltm)
        self.df = pd.concat([self.df, ltm_df], axis=1)

    def random_subset(self, nrows_to_keep):
        np.random.seed(123)
        keepers = np.random.choice(range(self.df.shape[0]), size=nrows_to_keep, replace=False)
        self.df = self.df.iloc[keepers,:]

    def prepare(self, regression=False, n_components=2, use_cached_processing=None, use_cached_tfidf=None, cache_processing=False, cache_tfidf=False, **tfidfargs):
        """Executes all cleaning methods in proper order. If regression, remove one
        dummy column and scale numeric columns for regularization"""
        import ipdb; ipdb.set_trace()
        self.drop_na()
        self.make_session_type()
        self.df = self.df[['party', 'passed', 'bill_xml', 'nterms']]
        # self.df['text_length'] = [len(content) for content in self.process_text('bill_xml', 'Content', use_cached_processing='../data/cached_processed_text_05-19-17-00-54.pkl')]
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
        #self.add_latent_topics(n_components,  use_cached_processing, use_cached_tfidf, cache_processing, cache_tfidf, **tfidfargs)

        # todrop = [u'bill_id', u'session_year', u'session_num', u'measure_type', u'fiscal_committee', u'bill_version_id', u'bill_xml']
        todrop = [u'bill_xml']
        self.drop_some_cols(todrop)

        y = self.df.pop('passed').values
        print "Using these features: {}".format(", ".join(self.df.columns))
        X = self.df.values

        return X, y

    def bucket_n_amendments(self, cutoff):
        self.df.n_prev_versions = self.df.n_prev_versions.apply(lambda n: 0 if n < cutoff else 1)

    def prepare_amendment_model(self, regression=False):
        """Executes all preparation for input into amendment model"""
        self.drop_na()
        self.make_session_type()
        if regression:
            self.dummify(['urgency', 'taxlevy', 'appropriation', 'party'], regression=True)
        else:
            self.dummify(['urgency', 'taxlevy', 'appropriation', 'party'])
        self.bucket_vote_required()
        todrop = [u'session_year', u'session_num', u'measure_type', u'fiscal_committee', u'bill_version_id']
        # ['days_since_start', 'vote_required', 'n_prev_versions', 'nterms', 'session_type', 'urgency_No', 'urgency_Yes', 'taxlevy_No', 'taxlevy_Yes',  'appropriation_Yes']
        self.drop_some_cols(todrop)
        print "Using these features: {}".format(", ".join(self.df.columns))

        return self.df


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
