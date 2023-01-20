import os.path
import sys
import yaml
from datetime import datetime
import time
from slugify import slugify
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.feature_selection import chi2, f_classif

from lr_modeling.lr_functions import *
from modeling.other_functions import read_data
import pickle

# Develop Logistic-Regression models
# Example: python3.7 -m lr_modeling.lr_models.py settings.yaml
# This script is based on:
#  - https://github.com/tasosnent/BeyondMeSH

if len(sys.argv) == 2:
    settings_file = sys.argv[1]
    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "Run with settings.py file at: " + settings_file)
else:
    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "No settings.py file found as argument, try to find the file in the project folder.")
    settings_file = "./settings.yaml"

# Read the settings
settings = {}

with open(settings_file) as parameters:
    settings = yaml.safe_load(parameters)

print('settings:' ,settings)

workingPath = settings["datasetPath"]
experiment_suffix = ''
# Suffixes are used for different versions of a dataset
if "experiment_suffix" in settings.keys():
    experiment_suffix = settings["experiment_suffix"]
detailsfgNL = pd.read_csv(settings["datasetPath"] + os.path.sep + settings["detailsfgNL"])
target_labels = list(detailsfgNL["Descr. UI"])
print("Run for target_labels:",str(target_labels))
testFileRaw = settings["testFileRaw"]
test_csv = workingPath + os.path.sep + testFileRaw[:testFileRaw.rfind(".")] + ".csv"
trainFileRaw = settings["trainFileRaw"]

training_csv = workingPath + os.path.sep + trainFileRaw[:trainFileRaw.rfind(".")] + experiment_suffix + ".csv"


base_experiment_path = workingPath + os.path.sep + slugify(str(datetime.now()))

if not os.path.exists(base_experiment_path):
    os.makedirs(base_experiment_path)
    print("Working folder: ",base_experiment_path)

with open(base_experiment_path + os.path.sep + "settings.yaml", 'w') as outfile:
    yaml.dump(settings, outfile, default_flow_style=False)

seed_values = settings["seed_vals"]

# Keep all predictions and golden to create the majority vote version
agg_predictions = {}
agg_golden = {}

for seed in seed_values:
    print("Run for seed: ", seed)
    seed = int(seed)
    agg_report = pd.DataFrame(columns = ['label','precision','recall','f1-score','support'])

    # create seed folder
    experiment_path = base_experiment_path + os.path.sep + str(seed)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # print("Undersampling the dataset")
    for target_label in target_labels:
        target_label = [target_label]
        print("\t Run for :", target_label)
        train_input_data = read_data(training_csv, target_label, use_cuis = settings["use_CUIs"])
        print('\ttotal train_input_data:', train_input_data.shape)
        positives = train_input_data[target_label].sum().values[0]
        negatives = train_input_data.loc[train_input_data[target_label[0]] == 0]
        # Keep negative instances as many as balance_n-times the positive ones.
        if "balance_n" in settings.keys():
            balance_n = settings["balance_n"]
            max_negative_instances = positives * balance_n
            if len(negatives) > max_negative_instances:
                negatives_to_remove = negatives.sample(n=int(len(negatives) - positives), random_state=seed)
                train_input_data = train_input_data[~train_input_data['pmid'].isin(negatives_to_remove['pmid'].tolist())]
                print('\tUnder-sampled dataset size', train_input_data.shape)
            else:
                print('\tNo under-sampling needed for this label.')
        else:
            print('\tNo under-sampling configured in this experiment (balance_n in settings)')

        # print("Tokenize dataset")
        # Tokenize
        train_counts, feature_names = tokenizeArticleText(train_input_data["text"])
        feature_names = pd.DataFrame({"token": feature_names})
        train_tfidf = getTFIDF(train_counts)
        train_labels = train_input_data[target_label]
        statistic = chi2
        if settings["FS_statistic"] == "f_classif":
            statistic = f_classif
        X_df = getFeatureWeights(train_tfidf, train_labels, feature_names, target_label, statistic)
        # Print basic statistics
        # Sort descending by Max Weight across all labels
        X_df = X_df.sort_values([MaxWeights], ascending=0)
        # print(X_df.head)

        features = min(int(settings["features_max"]), len(feature_names))
        # c = 1
        # Select top features/token only and transform tfidf matrix and tokens DataFrame accordingly
        tfidf_selectedFeatures, count_selectedFeatures, tokens_selectedFeatures = getTopFeatures(train_tfidf, X_df, train_counts,
                                                                                                             feature_names, features)

        top_token_csv = experiment_path + os.path.sep + "tokens_" + target_label[0] + ".txt"
        tokens_selectedFeatures.to_csv(top_token_csv, index=False)

        clf = OneVsRestClassifier(LogisticRegression())

        if settings['grid_search']:
            C= np.arange(float(settings["estimator_C_min"]), float(settings["estimator_C_max"]), float(settings["estimator_C_step"])).tolist()
            # print("C:",C)
            # Perform grid search for regularization level
            parameters = {
                "estimator__random_state": [seed],
                "estimator__penalty": ['l2'],
                "estimator__solver": ['liblinear'],
                'estimator__C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 1000000000]
            }

            scorer = make_scorer(f1_score, average='macro')
            clf = GridSearchCV(clf, parameters, cv=3, scoring=scorer)

            print("\ttrain tfidf shape:", tfidf_selectedFeatures.shape)

            clf.fit(tfidf_selectedFeatures, train_labels)

            print('\t' + 'Best : ' + str(clf.best_params_))

            with open(experiment_path + os.path.sep + "best_" + target_label[0] + ".txt", 'a') as f:
                f.write("clf.best_params_: \n")
                f.write(str(clf.best_params_))
                f.write("\nclf.cv_results_['mean_test_score']: \n")
                f.write(str(clf.cv_results_['mean_test_score']))
                f.write("\nclf.cv_results_['std_test_score']: \n")
                f.write(str(clf.cv_results_['std_test_score']))
                f.write("\nclf.cv_results_:\n")
                f.write(str(clf.cv_results_))
        else:
            clf.set_params(estimator__penalty="l2")
            # clf.set_params(estimator__solver='lbfgs')
            clf.set_params(estimator__solver='liblinear')
            clf.set_params(estimator__C=float(settings["estimator_C_min"]))
            clf.fit(tfidf_selectedFeatures, train_labels)

        # print("Load test data")

        test_input_data = read_data(test_csv, target_label, use_cuis = settings["use_CUIs"])
        test_labels = test_input_data[target_label]
        # Convert vocabulary from DataFrame to a Dictionary
        vocabulary = tokens_selectedFeatures.to_dict()["token"]
        # Invert it to be adequate for use by Tokenizer
        inv_vocabulary = {v: k for k, v in vocabulary.items()}
        # Tokenize
        test_counts, feature_names = tokenizeArticleText(test_input_data["text"], inv_vocabulary)
        # print("test count shape:", test_counts.shape)

        # print("Get TFIDF")
        test_tfidf = getTFIDF(test_counts, count_selectedFeatures)
        print("\ttest tfidf shape:", test_tfidf.shape)
        X_test_tfidf = test_tfidf.toarray()
        # print("predict")
        predicted = clf.predict(X_test_tfidf)

        # Note: filtering of predictions is not needed for single-label models where all "invalid" articles are removed from the test dataset

        if target_label[0] in agg_predictions.keys():
            agg_predictions[target_label[0]][str(seed)] = predicted
        else:
            agg_predictions[target_label[0]] = {str(seed):predicted}
            agg_golden[target_label[0]] = test_labels

        report = classification_report(test_labels, predicted, output_dict=True)

        print(report)
        positives_key = '1'
        if not positives_key in report.keys():
            positives_key = '1.0'
        scores_on_positives = report[positives_key]
        agg_report = agg_report.append({'label' : target_label[0],
                           'precision' : scores_on_positives['precision'],
                           'recall' : scores_on_positives['recall'],
                           'f1-score' : scores_on_positives['f1-score'],
                           'support' : scores_on_positives['support']  }, ignore_index=True)

    macro_avg = agg_report.mean()
    agg_report = agg_report.append({'label': 'macro avg',
                                    'precision': macro_avg['precision'],
                                    'recall': macro_avg['recall'],
                                    'f1-score': macro_avg['f1-score'],
                                    'support': macro_avg['support']}, ignore_index=True)
    std = agg_report.std()
    agg_report = agg_report.append({'label': 'std',
                                    'precision': std['precision'],
                                    'recall': std['recall'],
                                    'f1-score': std['f1-score'],
                                    'support': std['support']}, ignore_index=True)
    var = agg_report.var()
    agg_report = agg_report.append({'label': 'var',
                                    'precision': var['precision'],
                                    'recall': var['recall'],
                                    'f1-score': var['f1-score'],
                                    'support': var['support']}, ignore_index=True)
    # print(agg_report)

    agg_report_csv = experiment_path + os.path.sep + "report.csv"
    print("\tsave report at : ",agg_report_csv)
    agg_report.to_csv(agg_report_csv, index=False)

# Create the majority vote version
agg_mv_report = pd.DataFrame(columns = ['label','precision','recall','f1-score','support'])

for label in agg_predictions.keys():
    predictions = agg_predictions[label]
    golden = agg_golden[label]
    doc_num = len(golden)
    new_predictions = []

    for doc in range(doc_num):
        positive = 0
        for seed in predictions.keys():
            seed_prediction = predictions[seed]
            positive += seed_prediction[doc]
        if positive >= (len(predictions.keys()) / 2):
            new_predictions.append(1)
        else:
            new_predictions.append(0)

    new_predictions_array = np.array(new_predictions)

    mv_predictions_file = base_experiment_path + os.path.sep + label + "_preditions_mv.pkl"
    with open(mv_predictions_file, 'wb') as f:
        pickle.dump(new_predictions_array, f)

    report_json = classification_report(golden, new_predictions_array, output_dict=True)

    positives_key = '1'
    if not positives_key in report_json.keys():
        positives_key = '1.0'

    scores_on_positives = report_json[positives_key]
    agg_mv_report = agg_mv_report.append({'label' : label,
                       'precision' : scores_on_positives['precision'],
                       'recall' : scores_on_positives['recall'],
                       'f1-score' : scores_on_positives['f1-score'],
                       'support' : scores_on_positives['support']  }, ignore_index=True)

macro_avg = agg_mv_report.mean()
agg_mv_report = agg_mv_report.append({'label': 'macro avg',
                                'precision': macro_avg['precision'],
                                'recall': macro_avg['recall'],
                                'f1-score': macro_avg['f1-score'],
                                'support': macro_avg['support']}, ignore_index=True)
std = agg_mv_report.std()
agg_mv_report = agg_mv_report.append({'label': 'std',
                                'precision': std['precision'],
                                'recall': std['recall'],
                                'f1-score': std['f1-score'],
                                'support': std['support']}, ignore_index=True)
var = agg_mv_report.var()
agg_mv_report = agg_mv_report.append({'label': 'var',
                                'precision': var['precision'],
                                'recall': var['recall'],
                                'f1-score': var['f1-score'],
                                'support': var['support']}, ignore_index=True)

agg_mv_report_csv = base_experiment_path + os.path.sep + "all_report_mv.csv"
print("save report at : ",agg_mv_report_csv)
agg_mv_report.to_csv(agg_mv_report_csv, index=False)
