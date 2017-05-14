import pandas as pd
import get_sql


class DataCleaning(object):
    """An object to clean and wrangle data into format for a model"""

    def __init__(self, query=None, filepath=None, training=True, predict=False):
        """Reads in data

        Args:
            filepath (str): location of file with csv data
        """

        if training:
            if query:
                self.df = get_sql.get_df(query)
            elif filepath:
                self.df = pd.read_csv(filepath)
            else: # default queries
                self.df = self.default_data()

    def default_data(self):
        bills_query = """SELECT b.bill_id, e.earliest_bvid, b.session_year, b.session_num, measure_type, e.urgency, datediff(e.earliest_date,sd.start_date) as days_since_start, e.appropriation, e.vote_required, e.taxlevy, e.fiscal_committee, b.passed
            FROM bill_tbl b
            left join (select earliest.bill_id, earliest.earliest_date, bv.bill_version_id as earliest_bvid, bv.urgency, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee from
        				(select bill_id, min(bill_version_action_date) as earliest_date from bill_version_tbl
        				group by bill_id) earliest
        				join bill_version_tbl bv on (earliest.bill_id=bv.bill_id and earliest.earliest_date=bv.bill_version_action_date)) e
            on b.bill_id=e.bill_id
            join start_dates sd on b.session_year=sd.session_year and b.session_num=sd.session_num
            where b.measure_type in ('AB' , 'SB')"""
        bills_df = get_sql.get_df(bills_query)

        authors_query = """select bv.bill_id, count(l.author_name) as n_authors from bill_version_authors_tbl bva
            left join legislator_tbl l on bva.name=l.author_name and l.session_year=bva.session_year
            join bill_version_tbl bv on bv.bill_version_id=bva.bill_version_id
            where contribution='LEAD_AUTHOR' and bva.bill_version_id like '%INT' and (bv.bill_id like '%AB%' or bv.bill_id like '%SB%')
            group by bill_id"""
        authors_df = get_sql.get_df(authors_query)
        authors_df = self.aggregate_authors_df(authors_df)
        merged_df = pd.merge(bills_df, authors_df, on='bill_id')
        return merged_df

    def aggregate_authors_df(self, authors_df):
        #authors_df['n_authors'] = authors_df['n_authors'].apply(lambda n: n if n > 0 else np.nan)
        authors_df['committee'] = (authors_df['n_authors']==0).astype(int)
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
        print "dropped {} rows".format(before-after)

    def bucket_vote_required(self):
        """Separate all bills that require a simple majority from other types of bills that require more than
        a simple majority. There should really only be 2/3, so this function adds the
        handful of other vote types together"""
        self.df['vote_required'] = self.df['vote_required'].apply(lambda vote: 0 if vote=='Majority' else 1)

    def make_session_type(self):
        """Uses session_num to make a column indicating if it was regular session (0) or Extraordinary session(1)"""
        self.df['session_type'] = self.df['session_num'].apply(lambda session: 0 if session=='0' else 1)

    def clean(self, regression=False, predict=False, test=False):
        """Executes all cleaning methods in proper order. If regression, remove one
        dummy column and scale numeric columns for regularization"""
        #import ipdb; ipdb.set_trace()
        self.drop_na()
        self.make_session_type()
        if regression:
            self.dummify(['urgency', 'taxlevy', 'appropriation'], regression=True)
        else:
            self.dummify(['urgency', 'taxlevy', 'appropriation'])
        self.bucket_vote_required()
        todrop = [u'bill_id', u'session_year', u'session_num', u'measure_type', u'fiscal_committee', u'earliest_bvid']
        self.drop_some_cols(todrop)

        #import ipdb; ipdb.set_trace()
        y = self.df.pop('passed').values
        X = self.df.values

        return X, y
