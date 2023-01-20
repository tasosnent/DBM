import os.path
import sys
import yaml
from datetime import datetime
import time
import gc
from sklearn.metrics import classification_report, multilabel_confusion_matrix

# Evaluate Logistic-Regression models based on stored predictions (e.g. D052246_preditions_mv.pkl for label D052246)

from lr_modeling.lr_functions import *
from modeling.other_functions import read_data
import pickle

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
# settings = yaml.load(settings_file, Loader=yaml.FullLoader)
with open(settings_file) as parameters:
    settings = yaml.safe_load(parameters)

print('settings:' ,settings)
workingPath = settings["datasetPath"]
detailsfgNL = pd.read_csv(settings["datasetPath"] + os.path.sep + settings["detailsfgNL"])
target_labels = list(detailsfgNL["Descr. UI"])
print("Run for target_labels:",str(target_labels))
testFileRaw = settings["testFileRaw"]
test_csv = workingPath + os.path.sep + testFileRaw[:testFileRaw.rfind(".")] + ".csv"

base_experiment_path = settings["base_experiment_path"]

agg_mv_report = pd.DataFrame(columns = ['label','precision','recall','f1-score','support',"tn","fp","fn","tp"])

for label in target_labels:
    label_list = [label]
    print("\t Run for :", label_list)
    new_predictions_array = None
    sotred_predictions_file = base_experiment_path + os.path.sep + label +"_preditions_mv.pkl"
    print("\t sotred_predictions_file :", sotred_predictions_file)
    with open(sotred_predictions_file, 'rb') as f:
        new_predictions_array = pickle.load(f)
    test_input_data = read_data(test_csv, label_list, use_cuis=settings["use_CUIs"])
    test_labels = test_input_data[label_list]

    report_json = classification_report(test_labels, new_predictions_array, output_dict=True)
    matrix = multilabel_confusion_matrix(test_labels, new_predictions_array)
    print(report_json)
    positives_key = '1'
    if not positives_key in report_json.keys():
        positives_key = '1.0'

    scores_on_positives = report_json[positives_key]
    agg_mv_report = agg_mv_report.append({'label': label,
                                          'precision': scores_on_positives['precision'],
                                          'recall': scores_on_positives['recall'],
                                          'f1-score': scores_on_positives['f1-score'],
                                          'support': scores_on_positives['support'],
                                         "tn": int(matrix[1][0][0]),
                                         "fp": int(matrix[1][0][1]),
                                         "fn": int(matrix[1][1][0]),
                                         "tp": int(matrix[1][1][1])},
                                         ignore_index=True)

    del(new_predictions_array)
    del(test_input_data)
    del(test_labels)
    gc.collect()

agg_mv_report_csv = base_experiment_path + os.path.sep + "all_report_mv_full.csv"
print("save report at : ", agg_mv_report_csv)
agg_mv_report.to_csv(agg_mv_report_csv, index=False)