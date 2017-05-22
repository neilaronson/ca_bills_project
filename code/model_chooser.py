from data_prep import DataPrep
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from datetime import datetime
from bill_kfold import bill_kfold
import re

class ModelChooser(object):
    """This ModelChooser object takes in cleaned data and provides methods necessary to train and predict
    multiple types of models. The goal is to make a clean, customizable interface for comparing models"""

    def __init__(self, list_of_models):
        """Args:
                list_of_models (list): contains uninstantiated sklearn models to be tried
        """
        self.list_of_models = list_of_models
        self.trained_models = []

    def print_cv_results(self, X_data, y_data):
        """Prints out the cross-validated estimate of test error for f1, recall and precision"""
        for i, model in enumerate(self.list_of_models):
            print "Model: ", model
            print "F1 score: ", self.f1_scores[i]
            print "recall score: ", self.recall_scores[i]
            print "precision score: ", self.precision_scores[i]
            print "accuracy score: ", self.accuracy_scores[i]

    def fit_predict(self, x_data, y_data, custom_kfold=None):
        """This method is meant to be called in main as a one-stop method for fitting
        the model and generating predictions for cross-validation test error estimation"""
        self.f1_scores, self.recall_scores, self.precision_scores, self.accuracy_scores = self.predict_and_cv_score(x_data, y_data, custom_kfold)


    def train(self, x_data, y_data):
        """Goes through each model in self.list_of_models, finds best hyperparameters, then instantiates each model
        with its best hyperparameters. It also reports on what these hyperparameters were"""
        for model in self.list_of_models:
            model.fit(x_data, y_data)
            self.trained_models.append(model)

    def grid_search(self, x_data, y_data, tuning_params, custom_kfold=None):
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
        """Used by fit_predict to return model evaluation metrics through cross-validation"""
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

    def score(self, x_test, y_test):
        """This score function is meant to be used only for test data. One best hyperparameters
        are chosen through CV, use this method to get actual test error"""
        scores = []
        for model in self.list_of_models:
            predictions = model.predict(x_test)
            scores.append(f1_score(y_test, predictions))
        return scores



def main():
    #try_latent_topics_intro_model(3,3)
    # test_amendment_model()
    # test_intro_model()
    # grid_search_intro_model()
    # test_latent_topics_amd_model(10, 11)
    # grid_search_amd_model()
    score_intro_model()

def grid_search_intro_model():
    prep = DataPrep(filepath='../data/intro_data_05-19-17-21-41.csv')
    bkf = bill_kfold(prep.df)
    features =  [u'days_since_start', u'session_type', u'party_ALL_DEM', u'party_ALL_REP',
       u'party_BOTH', u'party_COM', u'urgency_No', u'urgency_Yes',
       u'taxlevy_No', u'taxlevy_Yes']
    X_train, y_train = prep.subset(features)

    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    ada = AdaBoostClassifier()

    mc = ModelChooser([rf, gb, ada])

    tuning_params = [   {'max_features': [.8], 'max_depth': [1, 5], 'n_estimators': [10000]},
                        {'learning_rate': [.1, .08, .05], 'max_depth': [1, 5]},
                        {'learning_rate': [.1, .08, .05]}]

    mc.grid_search(X_train, y_train, tuning_params, custom_kfold=bkf)

def test_intro_model():
    prep = DataPrep(filepath='/home/ubuntu/extra/data/intro_data_05-21-17-18-46.csv')
    features =  [u'days_since_start', u'session_type', u'party_ALL_DEM', u'party_ALL_REP',
       u'party_BOTH', 'party_COM', u'urgency_No', u'urgency_Yes',
        u'taxlevy_No',
       u'taxlevy_Yes', 'success_rate']
    X_train, y_train = prep.subset(features)


    baseline = DummyClassifier(strategy='stratified')

    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    ada = AdaBoostClassifier(learning_rate=0.1)

    mc = ModelChooser([baseline, ada])
    mc.fit_predict(X_train, y_train)
    mc.print_cv_results(X_train, y_train)


def test_latent_topics_amd_model(min_n, max_n):
    highest_f1 = 0
    for k in range(min_n, max_n+1):
        print "start time: {}".format(datetime.now())
        print "using {} latent topics".format(k)
        prep = DataPrep(filepath='../data/amendment_10_topics.csv')
        features = [str(i) for i in range(10)]
        features.insert(0, 'bill_id')
        features = ['bill_id', "1", "3", "5", "6", "7", "8", "9"]
        df = prep.subset(features, return_df=True)
        bkf = bill_kfold(df)
        y_train = df.pop('passed').values
        X_train = df.values[:,1:]
        # X_train, y_train = prep.prepare_amendment_model(n_components=k)

        print "regular data prep complete"

        # reg_prep = DataPrep()
        # X_train_reg, y_train_reg = reg_prep.prepare(regression=True, n_components=k)
        # print "regression data prep complete"

        rf = RandomForestClassifier(n_jobs=-1, n_estimators=1000)
        gb = GradientBoostingClassifier()

        mc = ModelChooser([rf, gb])
        mc.fit_predict(X_train, y_train)
        mc.print_cv_results(X_train, y_train)

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
    print "best f1 score was {} with {} latent features on {} model".format(highest_f1, best_n_latent_features, best_model_type)

def test_amendment_model():
    # prep = DataPrep(filepath='/home/ubuntu/extra/data/amendment_data_nprevvotes.csv')
    prep = DataPrep(filepath="/home/ubuntu/extra/data/amendment_data_05-21-17-20-05.csv")
    # df = prep.prepare_amendment_model(save=True)
    # features = [u'bill_id', u'A0',
    #            u'A1', u'A10', u'A11', u'A12', u'A13', u'A14', u'A15', u'A16', u'A17',
    #            u'A18', u'A19', u'A2', u'A20', u'A21', u'A22', u'A24', u'A25', u'A26',
    #            u'A27', u'A28', u'A29', u'A3', u'A30', u'A31', u'A4', u'A5', u'A6',
    #            u'A7', u'A8', u'A9', u'AE', u'S0', u'S1', u'S10', u'S11', u'S12',
    #            u'S13', u'S14', u'S15', u'S16', u'S17', u'S18', u'S19', u'S2', u'S20',
    #            u'S21', u'S22', u'S3', u'S4', u'S5', u'S6', u'S7', u'S8', u'S9']
    # features = ['bill_id', 'days_since_start', 'vote_required', 'n_prev_versions', 'nterms', 'session_type', 'urgency_No', 'urgency_Yes', 'taxlevy_No',
    #             'taxlevy_Yes', 'appropriation_No', 'appropriation_Yes', 'party_ALL_DEM', 'party_ALL_REP', 'party_BOTH', 'party_COM', 'success_rate']
    features = ['bill_id', 'success_rate', 'party_ALL_DEM', 'party_ALL_REP', 'party_BOTH', 'party_COM', 'session_type', 'days_since_start', 'n_prev_versions'
                'n_prev_votes', 'taxlevy_No', 'taxlevy_Yes', 'urgency_No', 'urgency_Yes']
    df = prep.subset(features, return_df=True)
    bkf = bill_kfold(df)
    #df = df.drop('bill_id', axis=1)
    y_train = df.pop('passed').values
    X_train = df.values[:,1:]

    baseline = DummyClassifier(strategy='most_frequent')
    gb = GradientBoostingClassifier()
    rf = RandomForestClassifier(n_estimators=100, max_features=.8, n_jobs=-1)
    mc = ModelChooser([baseline, gb, rf])



    # tuning_params = [{'strategy': ['most_frequent']}, {'n_estimators': [10]}, {'n_estimators': [10]}]
    #
    # mc.grid_search(X_train, y_train, tuning_params, custom_kfold=bkf)


    mc.fit_predict(X_train, y_train, custom_kfold=bkf)
    mc.print_cv_results(X_train, y_train)

def grid_search_amd_model():
    prep = DataPrep(filepath="/home/ubuntu/extra/data/amendment_data_05-21-17-20-05.csv")
    features = ['bill_id', 'success_rate', 'party_ALL_DEM', 'party_ALL_REP', 'party_BOTH', 'party_COM', 'session_type', 'days_since_start', 'n_prev_versions',
                'n_prev_votes', 'taxlevy_No', 'taxlevy_Yes', 'urgency_No', 'urgency_Yes']
    df = prep.subset(features, return_df=True)
    bkf = bill_kfold(df)
    y_train = df.pop('passed').values
    X_train = df.values[:,1:]

    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    ada = AdaBoostClassifier()

    mc = ModelChooser([rf, gb, ada])

    tuning_params = [   {'max_features': [.8], 'max_depth': [5], 'n_estimators': [100]},
                        {'learning_rate': [.1, .05, .01], 'max_depth': [1, 2, 3]},
                        {'learning_rate': [.1, .05, .01]}]

    mc.grid_search(X_train, y_train, tuning_params, custom_kfold=bkf)


def try_latent_topics_intro_model(min_n, max_n):
    highest_f1 = 0
    for k in range(min_n, max_n+1):
        print "start time: {}".format(datetime.now())
        print "using {} latent topics".format(k)
        prep = DataPrep(filepath='../data/subset_data.csv')
        #prep = DataPrep()
        X_train, y_train = prep.prepare(n_components=k)
        print "regular data prep complete"

        # reg_prep = DataPrep()
        # X_train_reg, y_train_reg = reg_prep.prepare(regression=True, n_components=k)
        # print "regression data prep complete"

        rf = RandomForestClassifier()
        gb = GradientBoostingClassifier()

        mc = ModelChooser([rf, gb])
        mc.fit_predict(X_train, y_train)
        mc.print_cv_results(X_train, y_train)

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
    print "best f1 score was {} with {} latent features on {} model".format(highest_f1, best_n_latent_features, best_model_type)

def score_intro_model():
    prep = DataPrep(filepath='/home/ubuntu/extra/data/intro_data_05-21-17-18-46.csv')
    features = [u'days_since_start', u'session_type', u'party_ALL_DEM', u'party_ALL_REP',
       u'party_BOTH', u'party_COM', u'urgency_No', u'urgency_Yes',
       u'taxlevy_No', u'taxlevy_Yes']
    X_train, y_train = prep.subset(features)

    baseline = DummyClassifier(strategy='stratified')
    ada = AdaBoostClassifier(learning_rate=0.1)

    mc = ModelChooser([baseline, ada])
    mc.train(X_train, y_train)

    dp = DataPrep(training=False)
    dp.prepare()
    features = [u'days_since_start', u'session_type', u'party_ALL_DEM', u'party_ALL_REP',
       u'party_BOTH', u'party_COM', u'urgency_No', u'urgency_Yes',
       u'taxlevy_No', u'taxlevy_Yes']
    X_test, y_test = dp.subset(features)


    score = mc.score(X_test, y_test)
    print "Test set scores: \n"
    print "F1: {}".format(score)


if __name__ == '__main__':
    main()
