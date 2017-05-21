import unittest
from data_prep import DataPrep

class DataPrepTest(unittest.TestCase):

    def test_read_csv(self):
        test_csv = "/home/ubuntu/ca_bills_project/data/extra/intro_data_raw_sample_5_21.csv"
        nrows = 100
        dp = DataPrep(filepath=test_csv)
        self.assertEqual(nrows, dp.df.shape[0])

    def test_intro_model_data(self):
        right_shape = (34001, 16)
        dp = DataPrep()
        self.assertEqual(right_shape, dp.df.shape)
        
