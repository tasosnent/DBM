import random
import os
import time
import yaml
import json
import codecs
import csv
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import classification_report
from slugify import slugify
from datetime import datetime
from shutil import copyfile

# Convert a RetroBM dataset for Fine-Grained Semantic Indexing into the format required for DBM
# The RetroBM method for developing datasets is available here: https://github.com/ThomasChatzopoulos/MeSH_retrospective_dataset

# Read the settings
settings_file = open("../data/settings.yaml")
settings = yaml.load(settings_file, Loader=yaml.FullLoader)

def handleLine(line):
    '''
    The JSON datasets are JSON Objects with an Array of documents available in the field names "documents"
    However, we follow a convention that "each line is a JSON object representing a single document" to allow reading the files line by line.
    To do so, we need to ignore the comma separating each line to the next one and the closing of the array in the last line "]}".
    :param line:    A line from a JSON dataset
    :return:        The line without the final coma "," or "]}" for the last line, ready to be parsed as JSON.
    '''
    stripped_line = line.strip()
    if stripped_line.endswith(','):
        # Remove the last character
        stripped_line = stripped_line[:len(stripped_line) - 1]
    elif stripped_line.endswith(']}'):
        # Remove the last two characters
        stripped_line = stripped_line[:len(stripped_line) - 2]
    return stripped_line;

def convertSIDataset(inputFile, dataset_folder, weak = False, skip_unlabelled = 0.0):
    '''
    Read a JSON-based dataset and save it as (1) FLAIR-compatible fast text format (2) CSV file with one column per label
    :param inputFile: String path to the original dataset in JSON format
    :param inputFile: Boolean value indicating what label value to add in the files: i) Weak, based on CUI values
                      or ii) Strong, based on manual annotations
    :param dataset_folder:  The folder path to write the specific dataset
    '''
    trueLabelField = "Descriptor_UIs" # String name of the field to get the true labels from to check validity.
    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> Start data conversion based on field: " + trueLabelField + ". Input file: " + inputFile)
    if weak:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "\t type of labels: Weak")
    else:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "\t type of labels: Manual")

  # Open a file to write in FastText format
    skip_suffix = ""
    if skip_unlabelled > 0.0:
        skip_suffix = "_skip"
        if skip_unlabelled < 1.0:
            skip_suffix += str(skip_unlabelled)
    fileFastText = codecs.open(dataset_folder + os.path.sep + inputFile[:inputFile.rfind(".")] + skip_suffix + ".txt", "w+", "utf-8")

    # Open CSV file to write with one column per label
    detailsfgNL = pd.read_csv(settings["workingPath"] + os.path.sep + settings["detailsfgNL"])
    fgNL_UIs = list(detailsfgNL["Descr. UI"]) # The IDs of the labels of interest
    fgNL_names = list(detailsfgNL['Descr. Name']) # The labels of the labels of interest
    fgNL_CUIs = list(detailsfgNL["CUI"]) # The corresponding CUIs of the labels of interest
    if len(fgNL_UIs) != len(fgNL_CUIs):
        print("Error reading file:" + detailsfgNL)
        print('Columns "Descr. UI" and "CUI" seems to have diferent number of rows! '
              '(i.e.' + len(fgNL_UIs) + " and " + len(fgNL_CUIs) + "respectively)")
    fgNL_CUI2UI = dict(zip(fgNL_CUIs, fgNL_UIs))
    fgNL_PHex_UIs_raw = list(detailsfgNL["PHex UIs"]) # The corresponding "extended" set of Previous Hosts separated by '~'
    if len(fgNL_UIs) != len(fgNL_PHex_UIs_raw):
        print("Error reading file:" + detailsfgNL)
        print('Columns "Descr. UI" and "PHex UIs" seems to have diferent number of rows! '
              '(i.e.' + len(fgNL_UIs) + " and " + len(fgNL_PHex_UIs_raw) + "respectively)")
    fgNL_PHex_UIs = []  # The corresponding "extended" set of Previous Hosts as a list
    for phl in fgNL_PHex_UIs_raw:
        fgNL_PHex_UIs.append(phl.split("~"))

    fileCSV = codecs.open(dataset_folder + os.path.sep + inputFile[:inputFile.rfind(".")] + skip_suffix + ".csv", "w+", "utf-8")
    # create the csv writer
    writer = csv.writer(fileCSV)
    # write a row to the csv file
    csvHeader = ["pmid", "text", "valid_labels", "CUIs"] + fgNL_UIs
    writer.writerow(csvHeader)
    no_cui_warnig = False # If the CUI field is missing from the JSON file, print a warning message
    no_cui_docs = [] # If the CUI field is empty in the JSON file, print a warning message
    no_cui_docs_positive = [] # If the CUI field is empty in the JSON file for a positive instance
    # Read the dataset file line by line
    with open(settings["workingPath"] + os.path.sep + inputFile, "r", encoding="utf8") as file:
        for line in file:
            # read each line
            stripped_line = handleLine(line)
            if not stripped_line.startswith('{"documents":['): # Skipp the first line
                document = json.loads(stripped_line)
                # print(document["pmid"],' - ', document["newFGDescriptors"] )
                # read and check CUIs
                occuring_cuis = []
                if "CUIs" in document.keys():
                    occuring_cuis = document["CUIs"]
                    if len(occuring_cuis) == 0:
                        no_cui_docs.append(str(document["pmid"]))
                        if len(fgNL_labels) != 0:
                            # Not a fully negative instance
                            no_cui_docs_positive.append(str(document["pmid"]))
                else:
                    no_cui_warnig = True

                # read labels to write
                # The labels to write are the the true labels
                labels = document["newFGDescriptors"]
                if weak:
                    labels = []
                    # If we want a weakly labeled dataset, get the wak labels based on the occurring CUIs
                    occuring_fgNL_CUIs = set.intersection(set(fgNL_CUIs), set(occuring_cuis))
                    for cui in occuring_fgNL_CUIs:
                        labels.append(fgNL_CUI2UI[cui])
                fgNL_labels = set.intersection(set(fgNL_UIs), set(labels))
                fast_text_labels = []
                fast_text_labels_serealized = " "
                true_labels = document[trueLabelField]
                valid_fgNL_labels = []
                # Iterate all fgNLs. We rely in the assumption that fgNL_UIs and fgNL_PHex_UIs are parallel arrays
                # (i.e. fgNL_UIs[i] is a fgNL label that corresponds to fgNL_PHex_UIs[i] extended list of previous hosts
                for i in range(len(fgNL_PHex_UIs)):
                    true_PHex_labels = set.intersection(set(fgNL_PHex_UIs[i]), set(true_labels))
                    if len(true_PHex_labels) > 0:
                        valid_fgNL_labels.append(fgNL_UIs[i])

                text = document["title"]+ " " + document["abstractText"]

                csvRow = [ document["pmid"], text," ".join(valid_fgNL_labels)," ".join(occuring_cuis)]

                for label in fgNL_UIs:
                    if label in labels and label in valid_fgNL_labels:
                        csvRow.append(1)
                        fast_text_labels.append("__label__" + label)
                    else:
                        csvRow.append(0)

                write = True
                if skip_unlabelled > 0:
                    if len(fast_text_labels) == 0:
                        # perform under-sampling in fully negative instances based on the given ratio:
                        #   for skip_unlabelled == 1.0, all unlabelled are skipped
                        #   for skip_unlabelled == 0.6, 60% of unlabelled are skipped
                        #   By "unlabelled" we mean all documents that are negative for all fgNL labels
                        if random.uniform(0, 1) <= skip_unlabelled:
                            write = False
                if write:
                    #  write this document in the output dataset only if at least one fgNL is valid for it
                    if(len(valid_fgNL_labels)):
                        writer.writerow(csvRow)
                        #  write this document in the fastText output datset
                        fileFastText.write(fast_text_labels_serealized.join(fast_text_labels) + text.replace('\n', ' ') + "\n")
    # Close the files
    fileFastText.close()
    fileCSV.close()
    if no_cui_warnig:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "---- Wanring - No CUI field in input file: " + inputFile)
    if len(no_cui_docs) > 0:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "---- Wanring - No CUI value for the following ", str(len(no_cui_docs)), " pmids : ", " ".join(no_cui_docs))
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "---- In the above instances ", str(len(no_cui_docs_positive)), " positive ones are included : ",
              " ".join(no_cui_docs_positive))

    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> End data convertion." )

def createSIBaselines(inputFile, dataset_folder):
    '''
    Create baselines
    Read a JSON-based dataset and create baseline predictions
    :param inputFile: String path to the original dataset in JSON format
    :param dataset_folder:  The folder path to write the specific dataset
    :return:
    '''
    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> Start baseline development for input file: " + inputFile )
    outputFile = dataset_folder + os.path.sep + inputFile[:inputFile.rfind(".")] + "_baselines.txt"
    labelField = "Descriptor_UIs" # String name of the field to get the true labels from to check validity.
    baselines = { }
    docLabels = []
    docText = []
    docWS = []
    docValidLabels = []

    # Read details for validity filtering and non-trivial baselines
    detailsfgNL = pd.read_csv(settings["workingPath"] + os.path.sep + settings["detailsfgNL"])
    fgNL = list(detailsfgNL["Descr. UI"])
    fgNL_CUIs = list(detailsfgNL["CUI"])  # The corresponding CUIs of the labels of interest
    fgNL_CUI2UI = dict(zip(fgNL_CUIs, fgNL))
    fgNL_PHex_UIs_raw = list(detailsfgNL["PHex UIs"])
    fgNL_PHex_UIs = []
    if len(fgNL) != len(fgNL_PHex_UIs_raw):
        print("Error reading file:" + detailsfgNL)
        print('Columns "Descr. UI" and "PHex UIs" seems to have diferent number of rows! '
              '(i.e.' + len(fgNL) + " and " + len(fgNL_PHex_UIs_raw) + "respectively)")
    for phl in fgNL_PHex_UIs_raw:
        fgNL_PHex_UIs.append(phl.split("~"))

    # create dictionaries for non-trivial baselines
    labelNames = {}
    labelPHs = {}
    for index, row in detailsfgNL.iterrows():
        labelNames[row['Descr. UI']] = row['Descr. Name']
        labelPHs[row['Descr. UI']] = row['PHs UIs']

    # Read the dataset file line by line
    with open(settings["workingPath"] + os.path.sep + inputFile, "r", encoding="utf8") as file:
        for line in file:
            # read each line
            stripped_line = handleLine(line)
            if not stripped_line.startswith('{"documents":['):  # Skipp the first line
                # print(len(docWS))
                document = json.loads(stripped_line)

                valid_fgNL_labels = []
                # Itterate all fgNLs. We rely in the assumption that fgNL_UIs and fgNL_PHex_UIs are parallel arrays
                # (i.e. fgNL_UIs[i] is a fgNL label that corresponds to fgNL_PHex_UIs[i] extended list of previous hosts
                for i in range(len(fgNL_PHex_UIs)):
                    true_PHex_labels= set.intersection(set(fgNL_PHex_UIs[i]), set(document[labelField]))
                    # A label is added in to the valid ones if any of its PHex_UIs exists in the true labels of this article
                    if len(true_PHex_labels) > 0:
                        valid_fgNL_labels.append(fgNL[i])

                # Consider documents that have at least one valid fgNL
                if(len(valid_fgNL_labels) > 0):
                    docLabels.append(document[labelField])
                    occuring_cuis = document["CUIs"]
                    occuring_fgNL_CUIs = set.intersection(set(fgNL_CUIs), set(occuring_cuis))
                    docWS_list = []
                    for cui in occuring_fgNL_CUIs:
                        docWS_list.append(fgNL_CUI2UI[cui])
                    docWS.append(docWS_list)
                    docText.append(document["title"]+ " " + document["abstractText"])
                    docValidLabels.append(valid_fgNL_labels)

    print(len(docLabels))
    # AllAll baseline
    baselines['allAll'] = np.zeros(shape=(len(docLabels),len(fgNL)))
    # baselines['allAll'][:,:] = 1
    for i in range(len(docLabels)):
        for j in range(len(fgNL)):
            # All valid labels are predicted
            if fgNL[j] in docValidLabels[i]:
                baselines['allAll'][i, j] = 1

    # AllRandom baseline
    np.random.seed(0)  # use a seed for reproducible random baseline results
    baselines['allRandom'] = np.zeros(shape=(len(docLabels),len(fgNL)))
    for i in range(len(docLabels)):
        for j in range(len(fgNL)):
            if fgNL[j] in docValidLabels[i]: # predict for valid labels only
                if np.random.random() > 0.5:
                    baselines['allRandom'][i, j] = 1

    # DictLabel baseline
    baselines['dictLabel'] = np.zeros(shape=(len(docLabels),len(fgNL)))
    for i in range(len(docLabels)):
        for j in range(len(fgNL)):
            if fgNL[j] in docValidLabels[i]:  # predict for valid labels only
                if docText[i].find(labelNames[fgNL[j]]) > -1:
                    baselines['dictLabel'][i, j] = 1

    # WSLabel baseline
    baselines['WSLabel'] = np.zeros(shape=(len(docLabels),len(fgNL)))
    for i in range(len(docWS)):
        for j in range(len(fgNL)):
            if fgNL[j] in docValidLabels[i]:  # predict for valid labels only
                if fgNL[j] in docWS[i]:
                    baselines['WSLabel'][i, j] = 1

    filehandler = open(outputFile, 'wb')
    pickle.dump(baselines, filehandler)

    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "-->> End baseline development, save in file: " + outputFile)

    fileCSV = dataset_folder + os.path.sep + inputFile[:inputFile.rfind(".")] + ".csv"
    test = pd.read_csv(fileCSV)
    test = test[fgNL]
    # print(test)
    from modeling.other_functions import save_evaluation_report

    for baseline in baselines.keys():
        print(baseline)
        print(baselines[baseline])
        csv_report_file = dataset_folder + os.path.sep + "report_"+baseline+".csv"
        print(classification_report(test, baselines[baseline]))
        report = classification_report(test, baselines[baseline], output_dict=True)
        # save_evaluation_report(report, fgNL, dataset_folder + os.path.sep + "report_"+baseline+".csv", None)
        save_evaluation_report(report, fgNL, None, csv_report_file)

def create_dataset_folder(taskName):
    # The folder path to write the specific dataset
    dataset_folder = os.path.join(settings["workingPath"]+os.path.sep, taskName + "_" + slugify(str(datetime.now())) + os.path.sep)
    os.mkdir(dataset_folder)
    settings["datasetPath"] = dataset_folder
    # Keep a copy of the settings for reference
    with open(dataset_folder + 'settings.yaml', 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)
    return dataset_folder

def data_stats(inputFile, dataset_folder):
    fileCSV = dataset_folder + os.path.sep + inputFile[:inputFile.rfind(".")] + ".csv"
    fileStats = dataset_folder + os.path.sep + inputFile[:inputFile.rfind(".")] + "_stats.txt"
    # Save data stats in a log file
    with open(fileStats, 'w') as f:
        # print('Filename:', file=f)  # Python 3.x
        dataset = pd.read_csv(fileCSV)
        detailsfgNL = pd.read_csv(settings["workingPath"] + os.path.sep + settings["detailsfgNL"])
        fgNL_UIs = list(detailsfgNL["Descr. UI"])
        # Total size
        print("All documents:",len(dataset), file=f)
        # Label frequency
        print("\nLabel frequency:", file=f)
        for label in fgNL_UIs:
            count = dataset[label].sum()
            print(label, count, file=f)
        # Document multi-labelness
        print("\nLabels per document:", file=f)
        total = dataset.loc[:, fgNL_UIs].sum(axis=1)
        labels_per_doc = total.value_counts()
        print(labels_per_doc, file=f)
        # Label combinations
        print("\nLabel combinations:", file=f)
        combinations = []
        for i in range(len(dataset)):
            s = dataset.iloc[i]
            a = s.index.values[(s == 1)]
            if len(a) > 1:
                combinations.append(" ".join(a))
                # print(a)
        # print(combinations)
        print([[l, combinations.count(l)] for l in set(combinations)], file=f)

# Conver the datasets into the required formats for model development
df = create_dataset_folder(settings["project_name"])

# Keep a copy of "detailsfgNL" CSV file for reference
copyfile(settings["workingPath"] + os.path.sep + settings["detailsfgNL"], df + os.path.sep + settings["detailsfgNL"])

#inputFile, dataset_folder, weak = False, skip_unlabelled = 0.0
# convertSIDataset(settings["trainFileRaw"], df, True)
convertSIDataset(settings["trainFileRaw"], df, True, 1)
# convertSIDataset(settings["trainFileRaw"], "newFGDescriptors", "Descriptor_UIs", df, 0.7)
# convertSIDataset(settings["trainFileRaw"], df, True, 1)
convertSIDataset(settings["testFileRaw"], df, False)

createSIBaselines(settings["testFileRaw"], df)
# createSIBaselines(settings["trainFileRaw"], df)

# data_stats(settings["trainFileRaw"], df )
# data_stats(settings["testFileRaw"], df )


