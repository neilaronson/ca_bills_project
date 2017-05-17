from data_prep import DataPrep
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('../data/test_amendment_data.csv')
    df = df.drop('passed', axis=1)
    bill_kfold(df)

def bill_kfold(X, nfolds=3):
    """Make folds such that no bill is spread out over more one fold"""
    length = X.shape[0]
    unique_bill_ids = X.bill_id.unique()
    # shuffle all bills
    np.random.shuffle(unique_bill_ids)
    # split into n groups
    bill_groups = np.array_split(unique_bill_ids, nfolds)
    # go through each group and find the original indices
    # that correspond to all the versions of the billid
    folds = [X[X.bill_id.isin(group)] for group in bill_groups]
    # yield each group of indices
    for i, fold in enumerate(folds):
        indices = [np.array(x.index) for x in folds[:i]] + [np.array(x.index) for x in folds[i+1:]]
        training_indices = np.concatenate(indices)
        yield (training_indices, np.array(folds[i].index))


if __name__ == '__main__':
    main()
