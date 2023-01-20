import json, csv
import pandas as pd

# Auxiliary functions for DBM model development and evaluation

def read_data(training_csv, target_labels, use_cuis = False):
    '''
    Read data from a CSV file and return prepare a dataset for the target_labels as a dataframe.
    :param training_csv:
    :param target_labels:
    :return:
    '''
    prompts = pd.read_csv(training_csv)
    prompts.head()

    # input_data = prompts[['pmid', target_label, 'text']]
    fields = ['pmid', 'text', 'valid_labels']
    for target_label in target_labels:
        fields.append(target_label)
    input_data = prompts[fields]

    # This is used to add semantic features to logistic regression baseline models
    if use_cuis:
        input_data.loc[:, 'text'] = input_data['text'] + " " + prompts['CUIs'].astype("str")

    # make sure all values are string
    input_data.loc[:, 'pmid'] = input_data['pmid'].astype("str")
    input_data.loc[:, 'text'] = input_data['text'].astype("str")

    input_data.loc[:, 'valid_labels'] = input_data['valid_labels'].astype("str")
    # make sure there are no missing values
    input_data = input_data[~input_data['pmid'].isin([pd.NA])]
    input_data = input_data[~input_data['text'].isin([pd.NA])]
    # remove data for out-of-focus labels
    input_data = input_data[input_data['valid_labels'].str.contains('|'.join(target_labels))]
    if use_cuis:
        print("Sample text with CUIs for article ", input_data.iloc[0]["pmid"] ,": ", input_data.iloc[0]["text"])
    return input_data

def save_evaluation_report(report_json, target_labels, report_json_file, report_csv_file, matrix = None):
    # open the file in the write mode
    if report_json_file is not None:
        with open(report_json_file, 'w') as outfile:
            json.dump(report_json, outfile)

    new_json = {}

    if report_csv_file is not None:
        with open(report_csv_file, 'w', newline='', encoding='utf-8') as f:
            # create the csv writer
            writer = csv.writer(f)
            header_row = ["label"] + list(report_json['macro avg'].keys())
            if matrix is not None:
                header_row = header_row + ["tn","fp","fn","tp"]
            writer.writerow(header_row)
            for row in report_json:
                new_row = report_json[row]
                if "avg" in row:
                    # This is a row like "micro avg", "macro avg" etc
                    content_row = [row] + list(report_json[row].values())
                    if matrix is not None:
                        content_row = content_row + [" "," "," "," "]
                elif "accuracy" in row:
                    # This is an "accuracy" row shown in binary classification
                    content_row = [row, " "," "] + [report_json[row]] + [" "]
                    if matrix is not None:
                        content_row = content_row + [" "," "," "," "]
                else:
                    # This is a label row
                    label_name = row # for binary cases "0.0" and "0.1" for negative and positive instances
                    if len(target_labels) > 1:
                        label_name = target_labels[int(row)] # for multilabel cases the index of the label (e.g. "0")
                    content_row = [label_name] + list(report_json[row].values())
                    if matrix is not None:
                        index = 1
                        if len(target_labels) > 1:
                            index = target_labels.index(label_name)
                        content_row = content_row + [matrix[index][0][0], matrix[index][0][1],matrix[index][1][0],matrix[index][1][1]]
                        new_row['tn'] = int(matrix[index][0][0])
                        new_row['fp'] = int(matrix[index][0][1])
                        new_row['fn'] = int(matrix[index][1][0])
                        new_row['tp'] = int(matrix[index][1][1])
                        new_json[row] = new_row
                # write a row to the csv file
                writer.writerow(content_row)
        if len(target_labels) > 1:
            # read the file just saved again to calculate and add stdv and var
            report_df = pd.read_csv(report_csv_file)
            # keep only rows for labels
            # print(target_labels)
            report_df_labels = report_df.loc[report_df['label'].isin(target_labels)]
            # calculate stdv and var
            report_df_var = report_df_labels.var()
            report_df_var.loc['label'] = "var"
            report_df_std = report_df_labels.std()
            report_df_std.loc['label'] = "std"
            # report_df = report_df.append()
            report_df = report_df.append(report_df_var, ignore_index=True)
            report_df = report_df.append(report_df_std, ignore_index=True)
            # report_df.loc['var'] = report_df_var
            # report_df.loc['std'] = report_df_std
            report_df.to_csv(report_csv_file, index=False)
        if report_json_file is not None:
            with open(report_json_file, 'w') as outfile:
                json.dump(new_json, outfile)

def stats_on_instace_validity(input_data, target_labels):
    positive_instances = input_data[target_labels].sum()
    valid_instances = {}
    valid_negative_instances = {}
    for label in target_labels:
        valid_instances[label] = input_data['valid_labels'].str.contains(label).sum()
        valid_negative_instances[label] = valid_instances[label] - positive_instances[label]
    print("\tValid_instances per label :")
    print(valid_instances)
    print("\tValid_negative_instances per label :")
    print(valid_negative_instances)
    return valid_instances, positive_instances