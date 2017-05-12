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
        self.drop_na()
        self.make_session_type()
        if regression:
            self.dummify(['urgency', 'taxlevy', 'appropriation'], regression=True)
        else:
            self.dummify(['urgency', 'taxlevy', 'appropriation'])
        self.bucket_vote_required()
        todrop = [u'bill_id', u'session_year', u'session_num', u'measure_type', u'fiscal_committee']
        self.drop_some_cols(todrop)

        #import ipdb; ipdb.set_trace()
        y = self.df.pop('passed').values
        X = self.df.values

        return X, y
