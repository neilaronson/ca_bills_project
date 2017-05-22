import pandas as pd
import get_sql
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from sklearn.decomposition import NMF
import cPickle as pickle
from datetime import datetime
from multiprocessing import Pool, cpu_count
import sql_queries

class DataPrep(object):
    """An object to clean and wrangle data into format for a model"""

    def __init__(self, query=None, filepath=None, training=True, predict=False, amendment_model=False):
        """Reads in data, either from csv, a specific SQL query, or takes a series of pre-defined steps to
        get data for the introduction model (by default) or the amendment model.

        Args:
            query (str): SQL query to execute to get data (optional, currently not used)
            filepath (str): location of file with csv data (optional)
            training (bool): whether to use train or test mode.
            predict (bool): whether to prepare unseen data to predict upon
            amendment_model (bool): whether to take the steps to build features for amendment model
        """
        if training:
            if query:
                self.df = get_sql.get_df(query)
            elif filepath:
                reader = pd.read_csv(filepath, chunksize=1000, encoding='utf-8')
                self.df = pd.DataFrame()
                chunks = [chunk for chunk in reader]
                self.df = pd.concat(chunks)
                nrows = self.df.shape[0]
                print "loaded csv, {} rows".format(nrows)
            elif amendment_model:
                self.df = self.amendment_data()
                # uncomment next life if you want to use the bill content the basis for features
                # self.df['content'] = [content for content in self.process_text('bill_xml', 'Content')]
            else: # default queries
                self.df = self.intro_data()
        else: #test mode
            if amendment_model:
                self.df = self.amendment_data(test=True)
            else:
                self.df = self.intro_data(test=True)

    def intro_data(self, test=False):
        """This executes all necessary queries and read all necessary files to build the master feature matrix
        for the introduction model. Currently, these features are:
            ['bill_id', 'bill_version_id', 'session_year', 'session_num',
       'measure_type', 'urgency', 'days_since_start', 'appropriation',
       'vote_required', 'taxlevy', 'fiscal_committee', 'passed', 'party',
       'nterms', 'success_rate', 'bill_xml']
       """

        if test:
            bills_query = sql_queries.intro_bills_query_test()
        else:
            bills_query = sql_queries.intro_bills_query()
        bills_df = get_sql.get_df(bills_query)

        bill_authors_merged_df = self.add_authors(bills_df, "intro")

        earliest_version_query = sql_queries.intro_earliest_version_query()
        text_df = get_sql.get_df(earliest_version_query)

        # bill_authors_merged_df = pd.merge(bills_df, authors_df, on='bill_id')
        bill_authors_text_df = pd.merge(bill_authors_merged_df, text_df, on='bill_id')

        return bill_authors_text_df

    def amendment_data(self, test=True):
        """This executes all necessary queries and read all necessary files to build the master feature matrix
        for the amendment model. Currently, these features are:
            ['bill_version_id', 'bill_xml', 'bill_id', 'session_year',
       'session_num', 'measure_type', 'urgency', 'days_since_start',
       'appropriation', 'vote_required', 'taxlevy', 'fiscal_committee',
       'passed', 'n_prev_versions', 'party', 'nterms', 'success_rate',
       'n_prev_votes']
       """
        if test:
            bills_query = sql_queries.amd_bills_query_test()
        else:
            bills_query = sql_queries.amd_bills_query()
        bills_df = get_sql.get_df(bills_query)

        n_amendments_query = sql_queries.amd_n_amendments_query()
        n_amendments_df = get_sql.get_df(n_amendments_query)

        merged_df = pd.merge(bills_df, n_amendments_df, on='bill_version_id')
        merged_df = merged_df.drop('bill_id_y', axis=1)
        merged_df = merged_df.rename(columns={'bill_id_x': 'bill_id'})
        merged_df = self.add_authors(merged_df, "amendment")

        text_query = sql_queries.amd_text_query()
        text_df = get_sql.get_df(text_query)
        merged_df = pd.merge(text_df, merged_df, on='bill_version_id')

        previous_committees_query = sql_queries.amd_prev_com_query()
        prev_com = get_sql.get_df(previous_committees_query)

        prev_com_count = prev_com.groupby('bill_version_id').count().reset_index()[['bill_version_id', 'SCID']]
        prev_com_count = prev_com_count.rename(columns={'SCID': 'n_prev_votes'})
        merged_df = pd.merge(merged_df, prev_com_count, on='bill_version_id', how='left')
        merged_df['n_prev_votes'] = merged_df['n_prev_votes'].fillna(value=0)

        return merged_df

    def add_authors(self, df, model):
        if model == "amendment":
            authors_query = sql_queries.amd_authors_query()
        elif model =="intro":
            authors_query = sql_queries.intro_authors_query()

        authors_df = get_sql.get_df(authors_query)
        authors_df = self.add_seniority(authors_df)
        authors_df = self.add_success_rate_amd(authors_df)
        authors_df = self.aggregate_authors_df(authors_df, model)
        if model == "amendment":
            merged_df = pd.merge(df, authors_df, on='bill_version_id')
        elif model =="intro":
            merged_df = pd.merge(df, authors_df, on='bill_id')
        return merged_df


    def add_success_rate_amd(self, df):
        success_rate_query = """select name as author_name, session_year, count(passed) as n_amended, sum(passed) as n_passed from (SELECT bva.name, bva.session_year, bva.bill_id, b.passed FROM capublic.bill_version_authors_tbl bva
            join bill_tbl b on b.bill_id=bva.bill_id
            where bill_version_id like '%AMD' and (b.bill_id like '%SB%' or b.bill_id like '%AB%')
            group by bva.name, bva.session_year, bva.bill_id
            order by name) t
            group by name, session_year"""
        success_rate = get_sql.get_df(success_rate_query)
        success_rate['success_rate'] = success_rate['n_passed'] / success_rate['n_amended']
        success_rate['session_year'] = success_rate['session_year'].apply(make_next_session_year)
        merged_df = pd.merge(df, success_rate, on=['author_name', 'session_year'], how='left')
        return merged_df


    def add_success_rate(self, df):
        success_rate_query = """SELECT  bva.name as author_name, bva.session_year, sum(b.passed) as n_passed, count(b.passed) as n_introduced
            FROM capublic.bill_version_authors_tbl bva
            join legislator_tbl l on bva.name=l.author_name and bva.session_year=l.session_year
            join bill_tbl b on b.bill_id=bva.bill_id
            where bill_version_id like '%INT' and b.measure_type in ('SB', 'AB') and bva.contribution='LEAD_AUTHOR' and bva.session_year < '20152016'
            group by  bva.name, bva.session_year
            order by bva.name, bva.session_year"""
        success_rate = get_sql.get_df(success_rate_query)

        # success_rate_cum = success_rate.groupby(['author_name', 'session_year']).sum().groupby(level=[0]).cumsum().reset_index()
        # success_rate_cum['success_rate'] = success_rate_cum['n_passed'] / success_rate_cum['n_introduced']
        # success_rate_cum['w_success_rate'] = success_rate_cum['success_rate'] * success_rate_cum['n_introduced']
        success_rate['success_rate'] = success_rate['n_passed'] / success_rate['n_introduced']

        # success_rate_cum['session_year'] = success_rate_cum['session_year'].apply(make_next_session_year)
        success_rate['session_year'] = success_rate['session_year'].apply(make_next_session_year)
        merged_df = pd.merge(df, success_rate, on=['author_name', 'session_year'], how='left')

        return merged_df


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


    def aggregate_authors_df(self, authors_df, model):
        if model == "intro":
            # all_seniority = self.make_seniority_data()

            authors_df['party'] = authors_df['party'].fillna('COM')
            # authors_party_seniority_df = pd.merge(authors_df, all_seniority, on=['district', 'session_year'], how='left')

            grouped_party_seniority = authors_df[['bill_id', 'party', 'nterms', 'success_rate']].groupby('bill_id')
            grouped_party_seniority = grouped_party_seniority.agg({'party': agg_parties, 'nterms': 'mean', 'success_rate': 'mean'}).reset_index()

            grouped_party_seniority.nterms[grouped_party_seniority['nterms'].isnull()] = -1000
            grouped_party_seniority.success_rate[grouped_party_seniority['success_rate'].isnull()] = -1000

            return grouped_party_seniority
        else:
            authors_df['party'] = authors_df['party'].fillna('COM')
            authors_df = authors_df.drop(['bill_id', 'session_year', 'district', 'author_name', 'legislator_name', 'name'], axis=1)
            authors_df = authors_df.groupby('bill_version_id').agg({'party': agg_parties, 'nterms': 'mean', 'success_rate': 'mean'}).reset_index()
            authors_df.nterms = authors_df.nterms.fillna(-1000)
            authors_df.success_rate = authors_df.success_rate.fillna(-1000)
            return authors_df

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

    def make_corpus(self, X_data):
        """Creates corpus lazily for tfidf and shows progress report for every 100 documents"""
        for i, bill in enumerate(X_data):
            if i % 100 == 0:
                print "on document {} of {}".format(i, len(X_data))
            yield bill

    def run_tfidf(self, use_cached_tfidf, cache_tfidf, X_data=None, identifier=None, **tfidfargs):
        """Apply TFIDF and get back transformed matrix"""
        if use_cached_tfidf:
            with open(use_cached_tfidf) as p:
                tfidf_contents = pickle.load(p)
                tfidf_mat = tfidf_contents[1]
            print "loaded tfidf"
        else: #not using a cached tfidf, will have to generate
            identifiers = self.df[identifier].values
            corpus = self.make_corpus(X_data)
            tfidf = TfidfVectorizer(tokenizer=tokenize, **tfidfargs)
            tfidf_mat = tfidf.fit_transform(corpus)
            if cache_tfidf:
                current_time = datetime.now().strftime(format='%m-%d-%y-%H-%M')
                filename = "/home/ubuntu/extra/data/cached_tfidf_"+current_time+".pkl"
                with open(filename, 'w') as p:
                    pickle.dump([tfidf, tfidf_mat, identifiers], p)
                print "pickled tfidf file at {}".format(filename)
            print "tfidf complete"
        return tfidf_mat

    def get_nmf_mat(self, X_data, n_components):
        """Returns sklearn's W matrix from NMF with the given number of latent topics"""
        nmf = NMF(n_components=n_components)
        W = nmf.fit_transform(X_data)
        return W

    def process_text(self, column_name, field, identifier, use_cached_processing=None, cache_processing=False):
        """Run each text row of column_name through BS to extract content from the specified XML field"""
        if use_cached_processing:
            with open(use_cached_processing) as p:
                bill_content = pickle.load(p)
                print "loaded processed text"
        else:
            identifiers = self.df[identifier].values
            bill_soup = self.df[column_name].values
            bill_content = [self.get_bill_text(soup, field) for soup in bill_soup]
            if cache_processing:
                current_time = datetime.now().strftime(format='%m-%d-%y-%H-%M')
                filename = "../data/cached_processed_text_"+current_time+".pkl"
                with open(filename, 'w') as p:
                    pickle.dump(zip(identifiers, bill_content), p)
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

    def process_and_tfidf(self, use_cached_processing=None, use_cached_tfidf=None, cache_processing=False, cache_tfidf=False, identifier=None, **tfidfargs):
        """Processes text through BeautifulSoup (if necessary) to extract an XML field and then runs tfidf
        on all bill's text"""
        if cache_tfidf and not use_cached_processing:  #make sure to cache processing if caching tfidf
            cache_processing=True
        if not use_cached_tfidf:
            if 'content' not in self.df.columns:
                processed = self.process_text('bill_xml', 'Content', use_cached_processing, cache_processing)
                self.df['content'] = processed
            self.df['content'][self.df['content'].isnull()] = " "
            X = self.df.content.values
            tfidf_mat = self.run_tfidf(use_cached_tfidf, cache_tfidf, X_data=X, identifier=identifier, **tfidfargs)
        else: #using cache, don't need to process
            tfidf_mat = self.run_tfidf(use_cached_tfidf, cache_tfidf)
        return tfidf_mat

    def add_latent_topics(self, n_components, use_cached_processing=None, use_cached_tfidf=None, cache_processing=False, cache_tfidf=False, **tfidfargs):
        """Adds latent topics to feature matrix"""
        tfidf_mat = self.process_and_tfidf(use_cached_processing, use_cached_tfidf, cache_processing, cache_tfidf, **tfidfargs)
        ltm = self.get_nmf_mat(tfidf_mat, n_components)
        col_names = ["topic_"+str(i) for i in range(n_components)]
        ltm_df = pd.DataFrame(ltm, columns=col_names)
        self.df = pd.concat([self.df, ltm_df], axis=1)

    def random_subset(self, nrows_to_keep):
        """Make a random sample of the dataframe stored in self.df"""
        np.random.seed(123)
        keepers = np.random.choice(range(self.df.shape[0]), size=nrows_to_keep, replace=False)
        self.df = self.df.iloc[keepers,:]

    def prepare(self, save=False, regression=False, n_components=2, use_cached_tfidf=None, cache_tfidf=False, **tfidfargs):
        """Executes all cleaning methods in proper order. If regression, remove one
        dummy column and scale numeric columns for regularization"""
        self.drop_na()
        self.make_session_type()
        # self.df = self.df[['party', 'passed', 'bill_xml']]
        # self.df['text_length'] = [len(content) for content in self.process_text('bill_xml', 'Content', use_cached_processing='../data/cached_processed_text_05-19-17-00-54.pkl')]
        if regression:
            self.dummify(['party', 'urgency', 'appropriation', 'taxlevy', 'fiscal_committee'], regression=True)
        else:
            self.dummify(['party', 'urgency', 'appropriation', 'taxlevy', 'fiscal_committee'])
        self.bucket_vote_required()

        # add latent topics
        # if use_text:
        #     self.add_latent_topics(n_components,  use_cached_processing, use_cached_tfidf, cache_processing, cache_tfidf, **tfidfargs)



        if save:
            current_time = datetime.now().strftime(format='%m-%d-%y-%H-%M')
            filename = "/home/ubuntu/extra/data/intro_data_" + current_time + ".csv"
            self.df.to_csv(filename, index=False, encoding='utf-8')

        # todrop = [u'bill_id', u'session_year', u'session_num', u'measure_type', u'fiscal_committee', u'bill_version_id', u'bill_xml']

        if 'bill_xml' in self.df.columns:
            todrop = [u'bill_xml']
            self.drop_some_cols(todrop)

        # y = self.df.pop('passed').values
        # print "Using these features: {}".format(", ".join([str(col) for col in self.df.columns]))
        # X = self.df.values

        # return X, y

    def subset(self, features, return_df=False):
        """Allows the user to specify which features from the feature matrix will be used.
        Features must already be processed (ie in numeric format)"""
        if 'passed' not in features:
            features.append('passed')
        self.df = self.df[features]
        print "Using these features: {}".format(", ".join(self.df.columns))
        if return_df:
            return self.df
        y = self.df.pop('passed').values
        X = self.df.values
        return X, y

    def bucket_n_amendments(self, cutoff):
        self.df.n_prev_versions = self.df.n_prev_versions.apply(lambda n: 0 if n < cutoff else 1)

    def check_for_stems(self, stems_filepath):
        """Checks to see whether certain stems are present in each row"""
        stems = pd.read_csv(stems_filepath, encoding='utf-8', names=['stems'])['stems'].values.tolist()
        stems = "|".join(stems)
        self.df['stemmed_text'] = applyParallel(self.df['content'].values, tokenize_join)
        self.df['n_bad_stems'] = self.df['stemmed_text'].str.count(stems)
        self.drop_some_cols(['stemmed_text'])

    def prepare_amendment_model(self, n_components=None, save=False, regression=False):
        """Executes all preparation for input into amendment model"""


        if n_components:
            self.add_latent_topics(n_components,  use_cached_tfidf='../data/cached_tfidf_05-19-17-20-07.pkl')

        self.drop_na()
        self.make_session_type()
        if regression:
            self.dummify(['party', 'urgency', 'appropriation', 'taxlevy', 'fiscal_committee'], regression=True)
        else:
            self.dummify(['party', 'urgency', 'appropriation', 'taxlevy', 'fiscal_committee'])
        self.bucket_vote_required()

        if save:
            current_time = datetime.now().strftime(format='%m-%d-%y-%H-%M')
            filename = "/home/ubuntu/extra/data/amendment_data_" + current_time + ".csv"
            self.df.to_csv(filename, index=False, encoding='utf-8')

        # todrop = [u'session_year', u'session_num', u'measure_type', u'bill_version_id', 'days_since_start', 'vote_required', 'n_prev_versions', 'nterms', 'session_type',
        # 'urgency_No', 'urgency_Yes', 'appropriation_No', 'appropriation_Yes', 'taxlevy_No', 'taxlevy_Yes', 'fiscal_committee_No', 'fiscal_committee_Yes', 'bill_xml', 'content']
        # ['days_since_start', 'vote_required', 'n_prev_versions', 'nterms', 'session_type', 'urgency_No', 'urgency_Yes', 'taxlevy_No', 'taxlevy_Yes',  'appropriation_Yes']
        if 'bill_xml' in self.df.columns:
            todrop = [u'bill_xml']
            self.drop_some_cols(todrop)

        # print "Using these features: {}".format(", ".join([str(col) for col in self.df.columns]))
        #
        # return self.df

def make_next_session_year(sy):
    """Change the session year to be the next session year. This is done so that when joining
    the information together about success rate, only the previous year's success rate is used."""
    session_year_start = int(sy[:4]) + 2
    session_year_end = int(sy[4:]) + 2
    return str(session_year_start) + str(session_year_end)



def applyParallel(x, func):
    """Applies a function to a pandas series in parallel"""
    print "beginning multiprocessing"
    pool = Pool(processes=cpu_count())
    results = pool.map(func, x)
    pool.close()

    return pd.Series(results)

def tokenize_join(text):
    """Join together a tokenized text into one string so that it can quickly be
    searched with a regular expression"""
    return " ".join(tokenize(text))


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
    bill_words_stemmed = [word.stem() for word in bill_words if word.isalpha()]
    return bill_words_stemmed
