import os
import pandas as pd
import pickle
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from modeling.other_functions import save_evaluation_report

# Evaluate DBM models based on stored predictions (e.g. D052246_preditions_mv.pkl for label D052246)

def get_measures_from_json(report_json):
    content_row = []
    if "1.0" in report_json.keys():  # Case of Binary classification: We get the measures on the positive class, not the macro-averaged one, as the latter is calculated on positive and negative in the same class
        content_row = content_row + [report_json["1.0"]["precision"],
                                     report_json["1.0"]["recall"],
                                     report_json["1.0"]["f1-score"],
                                     None, None,
                                     # We also use the same values in the micro-averaged column
                                     report_json["1.0"]["precision"],
                                     report_json["1.0"]["recall"],
                                     report_json["1.0"]["f1-score"]]
    else:
        content_row = content_row + [report_json["macro avg"]["precision"],
                                     report_json["macro avg"]["recall"],
                                     report_json["macro avg"]["f1-score"]]
        if "std" in report_json.keys():
            content_row = content_row + [report_json["std"]["f1-score"],
                                 report_json["var"]["f1-score"]]
        else:
            content_row = content_row + [None, None]
        if "micro avg" in report_json.keys():
            content_row = content_row + [report_json["micro avg"]["precision"],
                                         report_json["micro avg"]["recall"],
                                         report_json["micro avg"]["f1-score"]]
        else:
            content_row = content_row + [None, None, None]
    return content_row

# root_folder where the different DBM dataset folders (as created by Datasets.py) are stored.
root_folder = "\home\RetroData"
years = {
    # The specific paths for each dataset to be considered
    "2007": root_folder + os.path.sep + "Dataset_SI_2007_2022...",
    "2008": root_folder + os.path.sep + "Dataset_SI_2008_2022...",
    "2009": root_folder + os.path.sep + "Dataset_SI_2009_2022..."
    # ...
}
for year in years:
    folder = years[year]
    dataset_folder = root_folder + os.path.sep + folder
    golden_file = dataset_folder + os.path.sep + "test_y.pkl"
    detailsfgNL_file = dataset_folder + os.path.sep + "UseCasesSelected_"+str(year)+".csv"
    # Define which CSV files to be evaluated here:
    # minority (ALO3)
    mv_predictions_report_csv_file = dataset_folder + os.path.sep + "report_minority_full.csv"
    new_predictions_file = dataset_folder + os.path.sep + "label_matrix_test_minority_voter_"+str(year)+"_filtered.csv"
    # majority
    # mv_predictions_report_csv_file = dataset_folder + os.path.sep + "report_majority_full.csv"
    # new_predictions_file = dataset_folder + os.path.sep + "label_matrix_test_majority_voter_"+str(year)+"_filtered.csv"
    # label_model
    # mv_predictions_report_csv_file = dataset_folder + os.path.sep + "report_label_model_full.csv"
    # new_predictions_file = dataset_folder + os.path.sep + "label_matrix_test_label_model_"+str(year)+"_filtered.csv"
    # concept occurrence
    # mv_predictions_report_csv_file = dataset_folder + os.path.sep + "report_concept_occurrence_full.csv"
    # new_predictions_file = dataset_folder + os.path.sep + "label_matrix_test_concept_occurrence_label_"+str(year)+"_filtered.csv"
    # synonyms_lowercase
    # mv_predictions_report_csv_file = dataset_folder + os.path.sep + "report_synonyms_lowercase_full.csv"
    # new_predictions_file = dataset_folder + os.path.sep + "label_matrix_test_synonyms_lowercase_"+str(year)+"_filtered.csv"
    # name_exact_lowercase
    # mv_predictions_report_csv_file = dataset_folder + os.path.sep + "report_name_lowercase_full.csv"
    # new_predictions_file = dataset_folder + os.path.sep + "label_matrix_test_name_exact_lowercase_"+str(year)+"_filtered.csv"
    # print(detailsfgNL_file)
    detailsfgNL = pd.read_csv(detailsfgNL_file)
    fgNL = list(detailsfgNL["Descr. UI"])

    with open(golden_file, 'rb') as f:
        golden = pickle.load(f)
    pred = pd.read_csv(new_predictions_file)
    pred = pred[fgNL]
    pred = pred.to_numpy()

    report_json = classification_report(golden, pred, output_dict=True)
    matrix = multilabel_confusion_matrix(golden, pred)

    save_evaluation_report(report_json, fgNL, None, mv_predictions_report_csv_file, matrix)

    report_df = pd.read_csv(mv_predictions_report_csv_file)