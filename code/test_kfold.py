import unittest
from bill_kfold import bill_kfold
import pandas as pd
import numpy as np

class KFoldTest(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv('../data/test_amendment_data.csv')
        self.df = self.df.drop('passed', axis=1)
        self.nrows = self.df.shape[0]
        self.kfolder = bill_kfold(self.df, nfolds=5)


    def test_lengths(self):
        """Make sure that the largest fold is no more than 5% bigger than the smallest fold"""
        lengths_of_folds = [len(fold[0]) + len(fold[1]) for fold in self.kfolder]
        max_fold = float(max(lengths_of_folds))
        min_fold = min(lengths_of_folds)
        percent_dif = (max_fold-min_fold)/min_fold
        self.assertLessEqual(percent_dif, .05)


    def test_indices_add_up(self):
        """Make sure that the total number of indices in a fold is equal to the input number of rows"""
        fold = self.kfolder.next()
        total = len(fold[0]) + len(fold[1])
        self.assertEqual(self.nrows, total)

    def test_bill_integrity(self):
        """Make sure that in all folds no bill_id is present in both the training and test indices"""
        folds = list(self.kfolder)
        for fold in folds:
            # get bill ids of training set and validation set
            bill_ids_training = self.df[self.df.index.isin(fold[0])].bill_id.values
            bill_ids_validation = self.df[self.df.index.isin(fold[1])].bill_id.values
            # check for overlap
            overlap = np.intersect1d(bill_ids_training, bill_ids_validation)
            self.assertEqual(len(overlap), 0)
