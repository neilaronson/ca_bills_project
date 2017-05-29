from data_prep import DataPrep
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, r2_score, mean_squared_error
from sklearn.decomposition import NMF
from datetime import datetime
import numpy as np
import pandas as pd
import re
import cPickle as pickle
from multiprocessing import Pool, cpu_count
import traceback
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class ModelChooser(object):
    """This ModelChooser object takes in cleaned data and provides methods necessary to train and predict
    multiple types of models. The goal is to make a clean, customizable interface for comparing models"""

    def __init__(self, list_of_models):
        """Args:
                list_of_models (list): contains sklearn models to be tried
        """
        self.list_of_models = list_of_models
        self.trained_models = []

    def print_results(self, regressor=False):
        """Prints out the cross-validated estimate of test error. If classification, it will print f1, recall and precision
        If regression, it will print R2 and MSE

        Args:
            regressor (bool): whether ModelChooser is evaluating a regression model. Defaults to False, meaning classification model
        """
        if regressor:
            for i, model in enumerate(self.list_of_models):
                print "Model: ", model
                print "R2 score: ", self.r2_scores[i]
                print "MSE: ", self.mse_scores[i]
        else:
            for i, model in enumerate(self.list_of_models):
                print "Model: ", model
                print "F1 score: ", self.f1_scores[i]
                print "recall score: ", self.recall_scores[i]
                print "precision score: ", self.precision_scores[i]
                print "accuracy score: ", self.accuracy_scores[i]

    def fit_predict(self, x_data, y_data, custom_kfold=None, regressor=False):
        """This method is meant to be called as a one-stop method for fitting
         models and evaluating model performance

         Args:
            x_data (numpy array): training data
            y_data (numpy array): labels for training data
            custom_kfold (object or sequence): a custom k-fold cross validation object or sequence of
                train/test splits. Defaults to None, which will use sklearn's defaults
            regressor (bool): whether ModelChooser is evaluating a regression model. Defaults to False, meaning classification model
        """
        if regressor:
            self.r2_scores, self.mse_scores =  self.predict_and_cv_score_regression(x_data, y_data, custom_kfold)
        else:
            self.f1_scores, self.recall_scores, self.precision_scores, self.accuracy_scores = self.predict_and_cv_score(x_data, y_data, custom_kfold)


    def train(self, x_data, y_data):
        """Goes through each model in self.list_of_models and trains it. This function is meant to be used
        once optimized hyperparameters are already found, for use in training a final model before test set evaluation

        Args:
            x_data (numpy array): training data
            y_data (numpy array): labels for training data
         """
        for model in self.list_of_models:
            model.fit(x_data, y_data)
            self.trained_models.append(model)

    def grid_search(self, x_data, y_data, tuning_params, custom_kfold=None):
        """Performs a grid search to find the best hyperparameters for each model in self.list_of_models and
        reports on the results

        Args:
            x_data (numpy array): training data
            y_data (numpy array): labels for training data
            tuning_params (list of dicts): each entry in the list must line up with the order of models in self.list_of_models.
                Each entry is a dict containing hyperparameter name as key and list of values to try as values
            custom_kfold (object or sequence): a custom k-fold cross validation object or sequence of
                train/test splits. Defaults to None, which will use sklearn's defaults
            """
        for i, model in enumerate(self.list_of_models):
            grid = GridSearchCV(model, tuning_params[i], cv=custom_kfold, scoring='f1', n_jobs=-1, verbose=3)
            grid.fit(x_data, y_data)
            params = grid.best_params_
            trained_model = grid.best_estimator_
            self.trained_models.append(trained_model)
            p = re.compile(r"(.*)\(.*")
            model_name = re.match(p, str(trained_model)).group(1)
            print "for {} model, best parameters were: {}".format(model_name, params)
            print "its f1 score was: {} \n".format(grid.cv_results_['mean_test_score'][0])


    def predict_and_cv_score(self, x_data, y_data, custom_kfold=None):
        """Used by fit_predict to return model evaluation metrics through cross-validation, specifically for classification models

        Args:
            x_data (numpy array): training data
            y_data (numpy array): labels for training data
            custom_kfold (object or sequence): a custom k-fold cross validation object or sequence of
                train/test splits. Defaults to None, which will use sklearn's defaults
        """
        f1_scores = []
        recall_scores = []
        precision_scores = []
        accuracy_scores = []
        for model in self.list_of_models:
            f1_scores.append(cross_val_score(model, x_data, y_data, cv=custom_kfold, scoring='f1').mean())
            recall_scores.append(cross_val_score(model, x_data, y_data, cv=custom_kfold, scoring='recall').mean())
            precision_scores.append(cross_val_score(model, x_data, y_data, cv=custom_kfold, scoring='precision').mean())
            accuracy_scores.append(cross_val_score(model, x_data, y_data, cv=custom_kfold, scoring='accuracy').mean())
        return f1_scores, recall_scores, precision_scores, accuracy_scores

    def predict_and_cv_score_regression(self, x_data, y_data, custom_kfold=None):
        """Used by fit_predict to return model evaluation metrics through cross-validation, specifically for regression models

        Args:
            x_data (numpy array): training data
            y_data (numpy array): labels for training data
            custom_kfold (object or sequence): a custom k-fold cross validation object or sequence of
                train/test splits. Defaults to None, which will use sklearn's defaults
        """
        r2_scores = []
        mse_scores = []
        for model in self.list_of_models:
            r2_scores.append(cross_val_score(model, x_data, y_data, cv=custom_kfold, scoring='r2').mean())
            mse_scores.append(cross_val_score(model, x_data, y_data, cv=custom_kfold, scoring='neg_mean_squared_error').mean())
        return r2_scores, mse_scores

    def score(self, x_test, y_test, regressor=False):
        """This score function is meant to be used only for test data. One best hyperparameters
        are chosen through CV, use this method to get actual test error

        Args:
            x_data (numpy array): training data
            y_data (numpy array): labels for training data
            regressor (bool): whether ModelChooser is evaluating a regression model. Defaults to False, meaning classification model
            """
        if regressor:
            r2_scores = []
            mse_scores = []
            for model in self.list_of_models:
                predictions = model.predict(x_test)
                self.r2_scores.append(r2_score(y_test, predictions))
                self.mse_scores.append(mean_squared_error(y_test, predictions))
            self.print_results(regressor=True)
        else:
            self.f1_scores = []
            self.recall_scores = []
            self.precision_scores = []
            self.accuracy_scores = []
            for model in self.list_of_models:
                predictions = model.predict(x_test)
                self.f1_scores.append(f1_score(y_test, predictions))
                self.recall_scores.append(recall_score(y_test, predictions))
                self.precision_scores.append(precision_score(y_test, predictions))
                self.accuracy_scores.append(accuracy_score(y_test, predictions))
            self.print_results()

def test_intro_model():
    """Runs the intro model from start to finish and saves the trained model"""
    k = 100 # number of latent topics
    prep = DataPrep(filepath='/home/ubuntu/ca_bills_project/data/extra/topic_intro_data_05-23-17-08-23.csv')
    prep.prepare()

    features =  [u'days_since_start', u'session_type', u'party_ALL_DEM', u'party_ALL_REP',
       u'party_BOTH', 'party_COM', u'urgency_No', u'urgency_Yes',
        u'taxlevy_No',
       u'taxlevy_Yes']
    topic_features = ["topic_"+str(x) for x in range(k)]
    features += topic_features
    X_train, y_train = prep.subset(features)

    baseline = DummyClassifier(strategy='stratified')

    rf = RandomForestClassifier(max_features=0.1, n_estimators=1000, max_depth=8, n_jobs=-1)
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.05)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4)

    mc = ModelChooser([gb])

    mc.train(X_train, y_train)

    save_model(mc.list_of_models[0], "intro_model_100_topics_gb.pkl")

def try_latent_topics_intro_model(k):
    """Used to figure out the optimal number of latent topics to get the best F1 score for the intro model.

    Args:
        k (int): number of latent topics
    """
    highest_f1 = 0
    print "start time: {}".format(datetime.now())
    print "using {} latent topics".format(k)
    prep = DataPrep(filepath='/home/ubuntu/ca_bills_project/data/extra/intro_data_w_content_5_22.csv')
    prep.prepare(n_components=k, use_cached_tfidf='/home/ubuntu/ca_bills_project/data/extra/cached_tfidf_real_05-23-17-05-28.pkl')
    topic_features = ["topic_"+str(x) for x in range(k)]
    features = topic_features
    X_train, y_train = prep.subset(features)
    print "regular data prep complete"
    print topic_features


    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()

    mc = ModelChooser([rf, gb])
    mc.fit_predict(X_train, y_train)
    mc.print_results()

    for i, score in enumerate(mc.f1_scores):
        if score > highest_f1:
            highest_f1 = score
            best_n_latent_features = k
            if i == 0:
                best_model_type = "Random Forest"
            else:
                best_model_type = "Gradient Booster"


    print "end time: {}".format(datetime.now())
    print "-"*10
    results =  "f1 score was {} with {} latent features on {} model".format(highest_f1, best_n_latent_features, best_model_type)
    print results
    return results

def try_latent_topics_parallel(min_, max_, step):
    """Use multiprocessing on all cores to try different numbers of latent topics in parallel.

    Args:
        min_ (int): minimum number of latent topics
        max_ (int): upper bound of latent topics to try
        step (int): how many ints to step by in range
    """
    print "beginning multiprocessing"
    arange = range(min_, max_, step)
    pool = Pool(processes=cpu_count())
    results = pool.map(latent_topics_multiprocessing_helper, arange)
    pool.close()

    return results

def score_intro_model():
    """Open the pickle of the saved intro model and score against the test set"""
    k = 100
    features =  [u'days_since_start', u'session_type', u'party_ALL_DEM', u'party_ALL_REP',
       u'party_BOTH', 'party_COM', u'urgency_No', u'urgency_Yes',
        u'taxlevy_No',
       u'taxlevy_Yes']
    topic_features = ["topic_"+str(x) for x in range(k)]
    features += topic_features

    trained_model_file = "/home/ubuntu/ca_bills_project/data/extra/intro_model_100_topics_rf_10000trees.pkl"
    with open(trained_model_file) as p:
        model = pickle.load(p)
    mc = ModelChooser([model])
    dp = DataPrep(training=False)
    dp.prepare(n_components=k, use_cached_nmf='/home/ubuntu/ca_bills_project/data/extra/nmf_100_05-23-17-08-23.pkl',
        use_cached_tfidf="/home/ubuntu/ca_bills_project/data/extra/cached_tfidf_real_05-23-17-05-28.pkl", cache_tfidf=True, test=True)
    X_test, y_test = dp.subset(features)


    mc.score(X_test, y_test)

def save_model(model, filename=None):
    """Helper function to save a traind model as a pickle

    Args:
        model (sklearn model): a trained model
        filename (string): what to save the pickle as. defaults to None, which causes model to be saved under
            a default filename of model_name + current_time
    """
    if not filename:
        p = re.compile(r"(.*)\(.*")
        model_name = re.match(p, str(model)).group(1)
        current_time = datetime.now().strftime(format='%m-%d-%y-%H-%M')
        name = "/home/ubuntu/ca_bills_project/data/extra/"+model_name+current_time+".pkl"
    else:
        name = "/home/ubuntu/ca_bills_project/data/extra/"+filename
    with open(name, 'w') as p:
        pickle.dump(model, p)

def test_intro_model_n_amd():
    """This function was to test if there was any signal in the data that exists about a bill when it is first
    introduced that could determine how many amendments it would ultimately have. It did not do well (R2 of .05)
    so was not pursued further
    """
    prep = DataPrep(filepath='/home/ubuntu/ca_bills_project/data/extra/intro_data_w_content_5_22.csv')
    n=100
    prep.prepare(n_components=n, use_cached_tfidf='/home/ubuntu/ca_bills_project/data/extra/cached_tfidf_real_05-23-17-05-28.pkl')
    features = [

              u'days_since_start',
              u'vote_required',
                     u'nterms',         u'success_rate',
                      u'n_amd',         u'session_type',
              u'party_ALL_DEM',        u'party_ALL_REP',
                 u'party_BOTH',            u'party_COM',
                 u'urgency_No',          u'urgency_Yes',
           u'appropriation_No',    u'appropriation_Yes',
                 u'taxlevy_No',          u'taxlevy_Yes',
        u'fiscal_committee_No', u'fiscal_committee_Yes']
    topic_features = ["topic_"+str(k) for k in range(n)]
    features += topic_features
    X_train, y_train = prep.subset(features, dep_var='n_amd')

    baseline = DummyRegressor()

    gb = GradientBoostingRegressor()

    mc = ModelChooser([baseline, gb])
    mc.fit_predict(X_train, y_train, regressor=True)
    mc.print_results(regressor=True)

def latent_topics_multiprocessing_helper(k):
    """This function simply tries to call try_latent_topics_intro_model and reports back the error
    if there is one. It was added because multiprocessing does not report the line traceback error

    Args:
        k (int): number of latent topics to try
    """
    try:
        return try_latent_topics_intro_model(k)
    except Exception:
        print("Exception in worker:")
        traceback.print_exc()
        raise

def latent_topics_predict_party():
    """This function was to see if the latent topics of a bill when first introduced could determine the party of the
    authors. On this initial attempt, an F1 of 0 was discovered, so it was not pursued further
    """
    k = 100
    prep = DataPrep(filepath='/home/ubuntu/ca_bills_project/data/extra/intro_data_w_content_5_22.csv')
    prep.prepare_predict_party(n_components=k, use_cached_tfidf='/home/ubuntu/ca_bills_project/data/extra/cached_tfidf_real_05-23-17-05-28.pkl')

    topic_features = ["topic_"+str(x) for x in range(k)]
    features = ['party'] + topic_features
    X_train, y_train = prep.subset(features, dep_var='party')

    baseline = DummyClassifier(strategy='stratified')

    ada = AdaBoostClassifier(learning_rate=0.1)

    mc = ModelChooser([baseline, ada])
    mc.fit_predict(X_train, y_train)
    mc.print_results()


def grid_search_intro_model_with_latent_topics(k):
    """Perform grid search for intro model (predicting whether a bill will pass based on information
    available when first introduced). Tries parameters for Random Forest, Gradient Boosting and
    AdaBoost classifiers

    Args:
        k (int): number of latent topics
    """
    if k == 100: # there exists a saved file already if using 100 latent topics
        prep = DataPrep(filepath='/home/ubuntu/ca_bills_project/data/extra/topic_intro_data_05-23-17-08-23.csv')
        prep.prepare()
    else:
        prep = DataPrep(filepath='/home/ubuntu/ca_bills_project/data/extra/intro_data_w_content_5_22.csv')
        prep.prepare(n_components=k, use_cached_tfidf='/home/ubuntu/ca_bills_project/data/extra/cached_tfidf_real_05-23-17-05-28.pkl', save=True)

    topic_features = ["topic_"+str(x) for x in range(k)]
    features = [u'days_since_start', u'session_type', u'party_ALL_DEM', u'party_ALL_REP',
       u'party_BOTH', u'party_COM', u'urgency_No', u'urgency_Yes',
       u'taxlevy_No', u'taxlevy_Yes']
    features += topic_features
    X_train, y_train = prep.subset(features)

    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    ada = AdaBoostClassifier()

    mc = ModelChooser([rf, gb, ada])

    tuning_params = [   {'max_features': [.1, .5, .7], 'max_depth': [5, 8, 10], 'n_estimators': [100000]},
                        {'learning_rate': [.1, .05], 'max_depth': [2, 4], 'n_estimators': [100, 500]},
                        {'learning_rate': [.1, .05], 'n_estimators': [100, 500]}]

    mc.grid_search(X_train, y_train, tuning_params)

def plot_feature_importance(feature_names, feature_importances, n=20):
    """Plots the top n feature importances for each model. This is done
    with sklearn's feature_importances_ attribute.

    Args:
        feature_names (list): list of the names of features in the same order as feature_importances
        feature_importances (numpy array): array of the importance of features in the same order as feature_names
        n (int): top number of features to include in plot (defaults to 20)
    """
    feature_names = np.array(feature_names)
    top_nx = np.argsort(feature_importances)[:-n-1:-1]
    feat_import = feature_importances[top_nx] # now sorted
    feat_import = feat_import / feat_import.max()
    feature_names = feature_names[top_nx]
    fig = plt.figure()
    x_ind = np.arange(n)
    plt.barh(x_ind, feat_import, height=.3, align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, feature_names)
    plt.savefig('/home/ubuntu/ca_bills_project/graphs/feature_importances.png', dpi=300)

def feature_importance(model, feature_names):
    """Gets the feature importance of a model, lines it up with feature names, creates a dataframe of these features
    sorted by importance and calls plot_feature_importance to plot them

    Args:
        model (sklearn model): a trained model
        feature_names (list): list of the names of features
    """
    features_list = []
    imp_list = []
    feat_imps = model.feature_importances_
    for j, importance in enumerate(feat_imps):
        features_list.append(feature_names[j])
        imp_list.append(importance)
    features_df = pd.DataFrame(data=zip(features_list,imp_list), columns=['feature', 'importance'])
    print features_df.sort_values('importance', ascending=False)
    plot_feature_importance(feature_names, feat_imps)

def make_plots():
    """Make feature importance and partial dependence plots for intro model"""
    prep = DataPrep(filepath='/home/ubuntu/ca_bills_project/data/extra/topic_intro_data_05-23-17-08-23.csv')
    prep.prepare()
    k = 100
    trained_model_file = "/home/ubuntu/ca_bills_project/data/extra/intro_model_100_topics_rf_10000trees.pkl"
    with open(trained_model_file) as p:
        model = pickle.load(p)
    print "loaded model"
    features =  [u'days_since_start', u'session_type', u'party_ALL_DEM', u'party_ALL_REP',
       u'party_BOTH', 'party_COM', u'urgency_No', u'urgency_Yes',
        u'taxlevy_No',
       u'taxlevy_Yes']
    topic_features = ["topic_"+str(x) for x in range(k)]
    features += topic_features
    X_train, y_train = prep.subset(features)
    feature_importance(model, features)
    feature_subset_indices = [73, 13]
    gb_file = "/home/ubuntu/ca_bills_project/data/extra/intro_model_100_topics_gb.pkl"
    with open(gb_file) as p:
        gb = pickle.load(p)
    make_partial_dependence(gb, X_train, y_train, features, feature_subset_indices)

def make_partial_dependence(model, X_train, y_train, feature_names, feature_subset_indices):
    """Plot the partial dependence of given features for a gradient boosted model

    Args:
        model (sklearn model): a trained model
        X_train (numpy array): training data
        y_train (numpy array): label data
        feature_names (list): list of the names of features
        feature_subset_indices (list): the indices of which features to plot for the partial dependence plot
    """
    fig, axes = plot_partial_dependence(model, X_train, feature_subset_indices, feature_names=feature_names, n_jobs=1)
    fig.suptitle('Partial dependence of topics')
    plt.savefig('/home/ubuntu/ca_bills_project/graphs/partial_dep.png', dpi=300)



def main():
    test_intro_model()
    results =  try_latent_topics_parallel(10, 400, 10)
    print results
    grid_search_intro_model_with_latent_topics(100)
    score_intro_model()
    test_intro_model_n_amd()
    make_plots()

if __name__ == '__main__':
    main()
