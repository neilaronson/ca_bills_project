from data_cleaning import DataCleaning
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class Pipeline(object):
    """This pipeline object takes in cleaned data and provides methods necessary to train and predict
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
        for i, model in enumerate(self.trained_models):
            print "Model: ", model
            print "F1 score: ", self.f1_scores[i]
            print "recall score: ", self.recall_scores[i]
            print "precision score: ", self.precision_scores[i]
            print "accuracy score: ", self.accuracy_scores[i]

    def fit_predict(self, x_data, y_data):
        """This method is meant to be called in main as a one-stop method for fitting
        the model and generating predictions for cross-validation test error estimation"""
        self.train(x_data, y_data)
        self.f1_scores, self.recall_scores, self.precision_scores, self.accuracy_scores = self.predict_and_cv_score(x_data, y_data)

    def train(self, x_data, y_data):
        """Goes through each model in self.list_of_models, finds best hyperparameters, then instantiates each model
        with its best hyperparameters. It also reports on what these hyperparameters were"""
        for model in self.list_of_models:
            model = model()
            model.fit(x_data, y_data)
            self.trained_models.append(model)


    def predict_and_cv_score(self, x_data, y_data):
        """Used by fit_predict to return model evaluation metrics through cross-validation"""
        f1_scores = []
        recall_scores = []
        precision_scores = []
        accuracy_scores = []
        for model in self.trained_models:
            f1_scores.append(cross_val_score(model, x_data, y_data, scoring='f1').mean())
            recall_scores.append(cross_val_score(model, x_data, y_data, scoring='recall').mean())
            precision_scores.append(cross_val_score(model, x_data, y_data, scoring='precision').mean())
            accuracy_scores.append(cross_val_score(model, x_data, y_data, scoring='accuracy').mean())
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
    query = """SELECT b.bill_id, b.session_year, session_num, measure_type, e.urgency, datediff(e.earliest_date,sd.start_date) as days_since_start, e.appropriation, e.vote_required, e.taxlevy, e.fiscal_committee, b.passed
    FROM bill_tbl b
    left join (select earliest.bill_id, earliest.earliest_date, bv.bill_version_id as earliest_bvid, bv.urgency, bv.appropriation, bv.vote_required, bv.taxlevy, bv.fiscal_committee from
				(select bill_id, min(bill_version_action_date) as earliest_date from bill_version_tbl
				group by bill_id) earliest
				join bill_version_tbl bv on (earliest.bill_id=bv.bill_id and earliest.earliest_date=bv.bill_version_action_date)) e
    on b.bill_id=e.bill_id
    join start_dates sd on b.session_year=sd.session_year
    where b.measure_type in ('AB' , 'SB')"""
    cleaner = DataCleaning(query)
    X_train, y_train = cleaner.clean()

    reg_cleaner = DataCleaning(query)
    X_train_reg, y_train_reg = reg_cleaner.clean(regression=True)

    baseline = DummyClassifier
    rf = RandomForestClassifier
    gb = GradientBoostingClassifier

    pipe = Pipeline([baseline, rf, gb])
    pipe.fit_predict(X_train, y_train)
    pipe.print_cv_results(X_train, y_train)

    logr = LogisticRegression
    regression_pipe = Pipeline([logr])
    regression_pipe.fit_predict(X_train_reg, y_train_reg)
    regression_pipe.print_cv_results(X_train_reg, y_train_reg)



if __name__ == '__main__':
    main()
