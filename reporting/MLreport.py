import os
import json
import csv
import yaml
import pandas as pd
from pathlib import Path
import pickle
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from modeling.other_functions import save_evaluation_report
from contextlib import redirect_stdout

# Aggregate the predictions from different models (seeds) into a majority-vote prediction and the results from different datasets into a single one.

# Read the settings
settings_file = open("MLsettings.yaml")
settings = yaml.load(settings_file, Loader=yaml.FullLoader)

multi_seed_experiments = False;
if "multi_seed_experiments" in settings.keys():
    multi_seed_experiments = settings["multi_seed_experiments"]
dataset_folder = settings["dataset_folder"]
model_ep = None
if "model_ep" in settings.keys():
    model_ep_list = settings["model_ep"]

def aggregate_multi_seed_results(experiment_folder, sumup_writer):
    print("Run for ", experiment_folder )

    experiments_csv_file = experiment_folder + os.path.sep + "experiments.csv"
    if model_ep is not None:
        experiments_csv_file = experiment_folder  + os.path.sep + model_ep + "_experiments.csv"

    multi_seed_experiment_folder = str(Path(experiment_folder).parents[1])
    seed_experiment_folder = str(Path(experiment_folder).parents[0])
    # read WS baselines
    WS_labels_csv_file = multi_seed_experiment_folder + os.path.sep + "report_minority_full.csv"
    ws_df =pd.read_csv(WS_labels_csv_file)
    macro_avg_ws = ws_df.loc[ws_df['label'] == '1'] # for the case of datasets with just one labelQ binary classification
    micro_avg_ws = macro_avg_ws
    if len(macro_avg_ws) == 0:
        # This is a normal case of dataset with multiple labels
        macro_avg_ws = ws_df.loc[ws_df['label'] == 'macro avg']
        micro_avg_ws = ws_df.loc[ws_df['label'] == 'micro avg']
    # print(macro_avg_ws)

    # get folder paths

    dict_labels_csv_file = multi_seed_experiment_folder + os.path.sep + "report_dictLabel.csv"
    dict_df =pd.read_csv(dict_labels_csv_file)
    macro_avg_dict = dict_df.loc[dict_df['label'] == '1'] # for the case of datasets with just one labelQ binary classification
    micro_avg_dict = macro_avg_dict
    if len(macro_avg_dict) == 0:
        # This is a normal case of dataset with multiple labels
        macro_avg_dict = dict_df.loc[dict_df['label'] == 'macro avg']
        micro_avg_dict = dict_df.loc[dict_df['label'] == 'micro avg']

    # read settings
    experiment_setting_file = seed_experiment_folder + os.path.sep + "settings.yaml"
    experiment_settings = yaml.load(open(experiment_setting_file), Loader=yaml.FullLoader)
    detailsfgNL_file = multi_seed_experiment_folder + os.path.sep + experiment_settings["detailsfgNL"]
    # print(detailsfgNL_file)
    detailsfgNL = pd.read_csv(detailsfgNL_file)
    fgNL = list(detailsfgNL["Descr. UI"])
    # print(fgNL)
    seeds = []
    with open(experiments_csv_file, 'w', newline='', encoding='utf-8') as f:
        # create the csv writer
        writer = csv.writer(f)
        header_row = ["Dir", "learning_rate", "seed", "Epochs", "loss_func_name", "modelName",
                      "ma-p", "ma-r", "ma-f1", "ma-f1 std", "ma-f1 var",
                      "mi-p", "mi-r", "mi-f1",
                      "ma-p-val", "ma-r-val", "ma-f1-val", ]
        writer.writerow(header_row)
        model_eps = []
        if model_ep is not None:
            if model_ep == "both":
                model_eps.append("best")
                model_eps.append("prev")
                # model_eps.append("current")
            else:
                model_eps.append(model_ep)
        else:
            model_eps.append(None)
        for dir in os.listdir(experiment_folder):
            root = experiment_folder + os.path.sep + dir
            if os.path.isdir(root):
                seed_report = root + os.path.sep + "report.json"
                seed_val_report = root + os.path.sep + "val_report.json"
                for ep in model_eps:
                    if ep is not None:
                        seed_report = root  + os.path.sep + ep + os.path.sep + "report.json"
                        seed_val_report = root  + os.path.sep + ep + os.path.sep + "val_report.json"
                    content_row = []
                    seed = root.replace(experiment_folder + os.path.sep, '')
                    experiment_complete = True;
                    if os.path.isfile(seed_report):
                        seeds.append(seed)
                        # print("\t\t seed ", seed, " experiment complete.")
                        label_report_file = open(seed_report)
                        # print(label_report )
                        # print(label)
                        report_json = json.load(label_report_file)
                        content_row = content_row + [root, experiment_settings["learning_rate"], seed, experiment_settings["epochs"],experiment_settings["loss_func_name"],experiment_settings["modelName"]]
                        content_row = content_row + get_measures_from_json(report_json)
                    else:
                        experiment_complete = False;
                        print("No evaluation report found here: ", seed_report)
                    if os.path.isfile(seed_val_report):
                        val_label_report_file = open(seed_val_report)
                        val_report_json = json.load(val_label_report_file)
                        content_row = content_row + [val_report_json["macro avg"]["precision"],
                                                     val_report_json["macro avg"]["recall"],
                                                     val_report_json["macro avg"]["f1-score"]]
                    else:
                        experiment_complete = False;
                        print("No validation report found here: ", seed_val_report)
                    if content_row != [] and experiment_complete:
                        writer.writerow(content_row)
                    else:
                        print("\t\t seed ", seed, " experiment is still in progress for", ep , "epoch.")

    print("\t Aggregate results for ", len(seeds), " seeds: ", seeds)

    df = pd.read_csv(experiments_csv_file)

    df['rank'] = df['ma-f1-val'].rank()
    max = df.max()

    # Create ensemble predictions
    all_dirs = df['Dir'].tolist()
    # beyond_avg_on_val_dirs = beyond_avg_on_val['Dir'].tolist()
    # top_rank_val_dirs = top_rank_val['Dir'].tolist()

    if len(all_dirs) > 1:
        mv_all_report_json = create_ensemble_predictions(all_dirs, fgNL, "all")
        content_row = [experiment_folder, experiment_settings["learning_rate"], experiment_settings["epochs"], len(df), "-",experiment_settings["balance_ns"],experiment_settings["loss_func_name"],experiment_settings["modelName"],
            # Weak Label baseline
           macro_avg_ws["precision"].squeeze(),macro_avg_ws["recall"].squeeze(),macro_avg_ws["f1-score"].squeeze(),
           micro_avg_ws["precision"].squeeze(), micro_avg_ws["recall"].squeeze(), micro_avg_ws["f1-score"].squeeze(),
           # Dict baseline
           macro_avg_dict["precision"].squeeze(), macro_avg_dict["recall"].squeeze(),
           macro_avg_dict["f1-score"].squeeze(),
           micro_avg_dict["precision"].squeeze(), micro_avg_dict["recall"].squeeze(),
           micro_avg_dict["f1-score"].squeeze(),
           max["ma-p"], max["ma-r"], max["ma-f1"], max["mi-p"], max["mi-r"],
           max["mi-f1"],
           ]
        content_row = content_row + get_measures_from_json(mv_all_report_json)
        sumup_writer.writerow(content_row)
    else :
        print("No multiple experiments to aggregate!")

    return ws_df, dict_df, fgNL

def get_measures_from_json(report_json):
    # Adds values for 8 columns
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

def create_ensemble_predictions(dirs, fgNL, experiments, ensemble_type ="mv"):
    # mv: for "majority vote"
    original_predictions = {}
    new_predictions = []
    doc_num = 0
    label_num = 0
    for dir in dirs:
        if os.path.isdir(dir):
            seed_predictions_files = {}
            if model_ep is not None:
                if model_ep == "both":
                    seed_predictions_files[dir + os.path.sep + "best"]= dir + os.path.sep + "best" + os.path.sep + "preditions_filtered.pkl"
                    seed_predictions_files[dir + os.path.sep + "perv"]= dir + os.path.sep + "prev" + os.path.sep + "preditions_filtered.pkl"
                else:
                    seed_predictions_files[dir + os.path.sep + model_ep]= dir + os.path.sep + model_ep + os.path.sep + "preditions_filtered.pkl"
            if not seed_predictions_files:
                seed_predictions_files[dir] = dir + os.path.sep + "preditions_filtered.pkl"
            for seed_predictions_dir, seed_predictions_file in seed_predictions_files.items():
                with open(seed_predictions_file, 'rb') as f:
                    seed_predictions = pickle.load(f)
                    original_predictions[seed_predictions_dir] = seed_predictions
                    doc_num = len(seed_predictions)
                    label_num = len(seed_predictions[0])
        else:
            print("This is not a valid/accessible dir: ", dir)

    # print(len(original_predictions))
    for doc in range(doc_num):
        doc_new_prediction = []
        for label in range(label_num):
            positive = 0
            for or_pred in original_predictions.values():
                positive += or_pred[doc,label]
            if ensemble_type == "mv":
                if positive >= (len(dirs)/2):
                    doc_new_prediction.append(1)
                else:
                    doc_new_prediction.append(0)
            # Add other types of voting here
        new_predictions.append(doc_new_prediction)

    # golden data are the same for all runs
    golden_file = dirs[0] + os.path.sep + 'golden.pkl'
    if model_ep is not None:
        subfolder = model_ep
        if model_ep == "both":
            subfolder = "best"
        golden_file = dirs[0] + os.path.sep + subfolder + os.path.sep + 'golden.pkl'
    with open(golden_file, 'rb') as f:
        golden = pickle.load(f)

    new_predictions_array = np.array(new_predictions)

    blance_n_experiment_folder = str(Path(dir).parents[0])
    # No filtering needed, original predictions are already filtered!

    mv_predictions_file = blance_n_experiment_folder + os.path.sep + experiments + "_preditions_mv.pkl"
    mv_predictions_report_csv_file = blance_n_experiment_folder + os.path.sep + experiments + "_report_mv.csv"
    mv_predictions_report_json_file = blance_n_experiment_folder + os.path.sep + experiments + "_report_mv.json"
    if model_ep is not None:
        mv_predictions_file = blance_n_experiment_folder + os.path.sep + model_ep + "_" + experiments + "_preditions_mv.pkl"
        mv_predictions_report_csv_file = blance_n_experiment_folder + os.path.sep + model_ep + "_" + experiments + "_report_mv.csv"
        mv_predictions_report_json_file = blance_n_experiment_folder + os.path.sep + model_ep + "_" + experiments + "_report_mv.json"

    with open(mv_predictions_file, 'wb') as f:
        pickle.dump(new_predictions_array, f)
    report_json = classification_report(golden, new_predictions_array, output_dict=True)
    matrix = multilabel_confusion_matrix(golden, new_predictions_array)

    save_evaluation_report(report_json, fgNL, None, mv_predictions_report_csv_file, matrix)

    report_df = pd.read_csv(mv_predictions_report_csv_file)

    if label_num > 1 :
        report_json["var"] = {"precision" : report_df[report_df["label"] == "var"]['precision'].values[0],
                              "recall" : report_df[report_df["label"] == "var"]['recall'].values[0],
                              "f1-score" : report_df[report_df["label"] == "var"]['f1-score'].values[0]}
        report_json["std"] = {"precision" : report_df[report_df["label"] == "std"]['precision'].values[0],
                              "recall" : report_df[report_df["label"] == "std"]['recall'].values[0],
                              "f1-score" : report_df[report_df["label"] == "std"]['f1-score'].values[0]}
    with open(mv_predictions_report_json_file, 'w') as outfile:
        json.dump(report_json, outfile)

    return report_json

with open(dataset_folder + os.path.sep + 'log.txt', 'w') as f:
    with redirect_stdout(f):
        for me in model_ep_list :
            model_ep = me
            print("Run for ",model_ep," model epoch")

            if multi_seed_experiments:
                # Aggregate multi-seed experiments into a single prediction per dataset/year
                sumup_report_scv_file = dataset_folder + os.path.sep + "sumup_report.csv"
                if model_ep is not None:
                    sumup_report_scv_file = dataset_folder + os.path.sep + model_ep + "_sumup_report.csv"

                with open(sumup_report_scv_file, 'w', newline='', encoding='utf-8') as f:
                    # create the csv writer
                    writer = csv.writer(f)
                    header_row = ["Dir", "LR", "Epochs", "runs", "Model type", "balancing", "loss", "model",
                                  # Weak Label baseline
                                  "WL ma-p", "WL ma-r", "WL ma-f1", "WL mi-p", "WL mi-r", "WL mi-f1",
                                  # Dict baseline
                                  "Dict ma-p", "Dict ma-r", "Dict ma-f1", "Dict mi-p", "Dict mi-r", "Dict mi-f1",
                                  # max of each column across experiments
                                  "max ma-p", "max ma-r", "max ma-f1", "max mi-p", "max mi-r", "max mi-f1",
                                  # Majority vote experiments
                                  "mv all ma-p", "mv all ma-r", "mv all ma-f1", "mv all ma-f1 std", "mv all ma-f1 var", "mv all mi-p", "mv all mi-r", "mv all mi-f1"
                                  ]
                    writer.writerow(header_row)

                    agg_results_dict_df = None
                    agg_results_ws_df = None
                    agg_all_results_df = None

                    for experiment_folder in multi_seed_experiments:
                        ws_df, dict_df, fgNL = aggregate_multi_seed_results(experiment_folder, writer)
                        all_report_mv_csv_file = experiment_folder + os.path.sep + "all_report_mv.csv"
                        if model_ep is not None:
                            all_report_mv_csv_file = experiment_folder + os.path.sep + model_ep + "_all_report_mv.csv"
                        agg_all_df = pd.read_csv(all_report_mv_csv_file)

                        # for the case of datasets with just one label: binary classification
                        if len(fgNL) == 1:
                            agg_all_df = agg_all_df.loc[agg_all_df['label'] == '1.0']
                            ws_df_tmp = ws_df.loc[ws_df['label'] == '1']
                            if ws_df_tmp.empty:
                                ws_df = ws_df.loc[ws_df['label'] == '1.0']
                            else:
                                ws_df = ws_df_tmp
                            dict_df = dict_df.loc[dict_df['label'] == '1']
                            if dict_df.empty:
                                dict_df = dict_df.loc[dict_df['label'] == '1.0']
                            agg_all_df.at[1,'label'] = fgNL[0]
                            ws_df.at[1,'label'] = fgNL[0]
                            dict_df.at[1,'label'] = fgNL[0]
                        if agg_all_results_df is None:
                            agg_all_results_df = agg_all_df
                            agg_results_ws_df = ws_df
                            agg_results_dict_df = dict_df
                        else:
                            agg_all_results_df = pd.concat([agg_all_results_df, agg_all_df], axis=0)
                            agg_results_ws_df = pd.concat([agg_results_ws_df, ws_df], axis=0)
                            agg_results_dict_df = pd.concat([agg_results_dict_df, dict_df], axis=0)

                    agg_all_results_df = agg_all_results_df[(agg_all_results_df.label != "micro avg") & (agg_all_results_df.label != "macro avg") &
                                                            (agg_all_results_df.label != "weighted avg") & (agg_all_results_df.label != "samples avg")
                                                            & (agg_all_results_df.label != "accuracy") & (agg_all_results_df.label != "accuracy")
                                                            & (agg_all_results_df.label != "std") & (agg_all_results_df.label != "var")]
                    agg_results_ws_df = agg_results_ws_df[(agg_results_ws_df.label != "micro avg") & (agg_results_ws_df.label != "macro avg") &
                                                          (agg_results_ws_df.label != "weighted avg") & (agg_results_ws_df.label != "samples avg")
                                                          & (agg_results_ws_df.label != "accuracy") & (agg_results_ws_df.label != "accuracy")
                                                          & (agg_results_ws_df.label != "std") & (agg_results_ws_df.label != "var")]
                    agg_results_dict_df = agg_results_dict_df[(agg_results_dict_df.label != "micro avg") & (agg_results_dict_df.label != "macro avg") &
                                                              (agg_results_dict_df.label != "weighted avg") & (agg_results_dict_df.label != "samples avg")
                                                              & (agg_results_dict_df.label != "accuracy") & (agg_results_dict_df.label != "accuracy")
                                                              & (agg_results_dict_df.label != "std") & (agg_results_dict_df.label != "var")]

                    agg_all_results_csv = dataset_folder + os.path.sep + "aggregated_report_mv_all.csv"
                    if model_ep is not None:
                        agg_all_results_csv = dataset_folder + os.path.sep + model_ep + "_aggregated_report_mv_all.csv"

                    agg_results_ws_csv = dataset_folder + os.path.sep + "aggregated_report_WSLabels.csv"
                    agg_results_dict_csv = dataset_folder + os.path.sep + "aggregated_report_DictLabels.csv"

                    agg_all_results_df.to_csv(agg_all_results_csv, index=False)
                    agg_results_ws_df.to_csv(agg_results_ws_csv, index=False)
                    agg_results_dict_df.to_csv(agg_results_dict_csv, index=False)

                    # wilcoxon stat
                    from scipy.stats import wilcoxon
                    print("~~ - ~~ ")
                    print("f1-score ","diff between WSlabels and ", "mv all")
                    d =  agg_all_results_df["f1-score"] - agg_results_ws_df["f1-score"]
                    w, p = wilcoxon(d)
                    print("Two-sided Wilcoxon test: H0 \"there is no difference in the two groups\"")
                    print("\t W statistic:", w)
                    print("\t p-value:", p)
                    print("\t p-value (rounded):", round(p,6))
                    w, p = wilcoxon(d, alternative='greater')
                    print("Single-sided Wilcoxon test: H0 \"the median of the difference in the two groups is negative\"")
                    print("\t W statistic:", w)
                    print("\t p-value:", p)
                    print("\t p-value (rounded):", round(p,6))
