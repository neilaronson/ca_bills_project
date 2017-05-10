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

    def make_target_variable(self):
        """Create the target variable of whether a bill passed both houses or not"""
        self.df['measure_state'] = self.df['measure_state'].apply(lambda state: 1 if state in ('Chaptered', 'Enrolled') else 0)

    def dummify(self, columns):
        """Create dummy columns for categorical variables"""
        dummies = pd.get_dummies(self.df[columns], columns=columns,
                                prefix=columns, dummy_na=True)
        self.df = self.df.drop(columns, axis=1)
        self.df = pd.concat([self.df,dummies], axis=1)

    def drop_some_cols(self, columns):
        """Simply drop columns (as list) from the dataframe"""
        for col in columns:
            self.df = self.df.drop(col,axis=1)

    def clean(self, regression=False, predict=False, test=False):
        """Executes all cleaning methods in proper order. If regression, remove one
        dummy column and scale numeric columns for regularization"""
        self.make_target_variable()
        self.dummify(['urgency'])
        todrop = [u'bill_id', u'session_year', u'session_num', u'measure_type',
       u'measure_num', u'chapter_year', u'chapter_num',
       u'latest_bill_version_id']
        self.drop_some_cols(todrop)
        y = self.df.pop('measure_state').values
        X = self.df.values

        return X, y
