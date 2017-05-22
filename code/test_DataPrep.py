import unittest
from data_prep import DataPrep
import pandas as pd
import pandas.util.testing as pdt

class DataPrepTest(unittest.TestCase):
    def test_read_csv(self):
        test_csv = "/home/ubuntu/ca_bills_project/data/extra/intro_data_raw_sample_5_21.csv"
        nrows = 100
        dp = DataPrep(filepath=test_csv)
        self.assertEqual(nrows, dp.df.shape[0])

class DataPrepIntroTest(unittest.TestCase):

    def setUp(self):
        self.dp = DataPrep()

    def test_intro_model_data_shape(self):
        right_shape = (34001, 16)
        dp = DataPrep()
        self.assertEqual(right_shape, self.dp.df.shape)

    def test_intro_model_data(self):
        raw_intro_data_csv = "/home/ubuntu/ca_bills_project/data/extra/intro_data_raw_5_21.csv"
        reader = pd.read_csv(raw_intro_data_csv, chunksize=1000, encoding='utf-8', dtype={'session_year': object, 'session_num': object})
        df = pd.DataFrame()
        chunks = [chunk for chunk in reader]
        df = pd.concat(chunks)
        # from nose.tools import set_trace; set_trace()
        taxlevy = df['taxlevy']
        self.assertTrue(taxlevy.equals(self.dp.df['taxlevy']))

class DataPrepAmdTest(unittest.TestCase):

    def setUp(self):
        self.dpa = DataPrep(amendment_model=True)

    def test_amd_model_data(self):
        right_shape = (73559, 18)
        self.assertEqual(right_shape, self.dpa.df.shape)

    def test_amd_model_data(self):
        raw_amd_data_csv = "/home/ubuntu/ca_bills_project/data/extra/amendment_raw_data_5_21.csv"
        reader = pd.read_csv(raw_amd_data_csv, chunksize=1000, encoding='utf-8')
        df = pd.DataFrame()
        chunks = [chunk for chunk in reader]
        df = pd.concat(chunks)
        days_since_start = df['days_since_start']
        self.assertTrue(days_since_start.equals(self.dpa.df['days_since_start']))
