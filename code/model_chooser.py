from data_prep import DataPrep
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
        """Prints out the cross-validated estimate of test error for f1, recall and precision
        It also calls the get_variable_imp to report on the variable importance for each model"""
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
            model = model()
            model.fit(x_data, y_data)
            self.trained_models.append(model)

    def grid_search(self, x_data, y_data, tuning_params, custom_kfold=None):
        for i, model in enumerate(self.list_of_models):
            grid = GridSearchCV(model, tuning_params[i], cv=custom_kfold, scoring='f1')
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
        for model in self.trained_models:
            predictions = model.predict(x_test)
            scores.append(f1_score(y_test, predictions))
        return scores



def main():
    # try_latent_topics_intro_model(3,4)
    test_amendment_model()

def test_amendment_model():
    prep = DataPrep(amendment_model=True)
    df = prep.prepare_amendment_model()
    bkf = bill_kfold(df)
    #df = df.drop('bill_id', axis=1)
    y_train = df.pop('passed').values
    X_train = df.values[:,1:]

    baseline = DummyClassifier()
    gb = GradientBoostingClassifier()

    rf = RandomForestClassifier()
    mc = ModelChooser([baseline, rf])

    tuning_params = [{'strategy': ['most_frequent']}, {'n_estimators': [10]}, {'n_estimators': [10]}]

    mc.grid_search(X_train, y_train, tuning_params, custom_kfold=bkf)
    #
    #
    # mc.fit_predict(X_train, y_train, custom_kfold=bkf)
    # mc.print_cv_results(X_train, y_train)


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

        # logr = LogisticRegression
        # regression_mc = ModelChooser([logr])
        # regression_mc.fit_predict(X_train_reg, y_train_reg)
        # regression_mc.print_cv_results(X_train_reg, y_train_reg)
        print "end time: {}".format(datetime.now())
        print "-"*10
    print "best f1 score was {} with {} latent features on {} model".format(highest_f1, best_n_latent_features, best_model_type)


if __name__ == '__main__':
    main()
