import pickle, os
import pandas as pd, json
import yaml

# Convert a C2F-generated dataset into the format required for DBM

# Read the settings
settings_file = open("../data/settings_c2f.yaml")
settings = yaml.load(settings_file, Loader=yaml.FullLoader)
datasets = settings["datasets"]

for original_datset_path in datasets:
    year = original_datset_path.split("Dataset_SI_")[1].split("_")[0]
    print(year)
    detailsfgNL = pd.read_csv(original_datset_path + os.path.sep + "UseCasesSelected_" + str(year) + ".csv")
    c2f_dataset_path = original_datset_path + os.path.sep + "fgsi" + str(year) + "u"
    fgNL = list(detailsfgNL["Descr. UI"])
    cgL = list(detailsfgNL["PHs UIs"])
    fgNL_labels = list(detailsfgNL["Descr. Name"])  # The corresponding CUIs of the labels of interest
    cgL_labels = list(detailsfgNL["PHs"])  # The corresponding CUIs of the labels of interest
    # Add parent_rest labels
    parent_uis = set(list(cgL))
    parent_labels = set(list(cgL_labels))
    for p, pui in zip(parent_labels, parent_uis):
        fgNL_labels.append(p + "_rest")
        cgL_labels.append(p)
        fgNL.append(pui + "_rest")
        cgL.append(pui)
        # print(p, pui)
    # clean labels (the two list are parallel ie of the same size)
    for i in range(len(fgNL_labels)):
        fgNL_labels[i] = fgNL_labels[i].replace(" ", "_").replace(",", "")
        cgL_labels[i] = cgL_labels[i].replace(" ", "_").replace(",", "")
    UI2Label = dict(zip(fgNL + cgL, fgNL_labels + cgL_labels))
    Label2UI = dict(zip(fgNL_labels + cgL_labels, fgNL + cgL))
    fgNL_UI2parentLabel = dict(zip(fgNL, cgL_labels))
    print(fgNL_UI2parentLabel)
    parent_child = {}
    with open(c2f_dataset_path + os.path.sep + "parent_to_child.json", "r+") as f:
        parent_child = json.load(f)

    df_train  = pd.read_csv(original_datset_path + os.path.sep + "train_" + str(year) + ".csv")

    print(df_train.info())
    for parent in parent_child.keys():
        # print("parent", parent)
        parent_UI = Label2UI[parent]
        with open(c2f_dataset_path + os.path.sep + "df_gen_" + parent + ".pkl", 'rb') as f:
            print(f)
            df = pickle.load(f)
        children_labels =  parent_child[parent]
        children_UIs = []
        for child_label in children_labels:
            if not child_label.endswith('_rest'):
                children_UIs.append(Label2UI[child_label])
        # print("children", children_labels)
        for child_label in children_labels:
            # print("child_label", child_label)
            if not child_label.endswith('_rest'):
                child_UI = Label2UI[child_label]
                df_fgl = df.loc[df['label'] == child_label]
                true_values = [1] * len(df_fgl)
                df_fgl[child_UI] = true_values

                valid_labels = [' '.join(children_UIs)] * len(df_fgl)
                df_fgl["valid_labels"] = valid_labels
                pmids = ["c2f_gen"] * len(df_fgl)
                df_fgl["pmid"] = pmids
                false_values = [0] * len(df_fgl)
                for other_label in fgNL_labels:
                    if not other_label.endswith('_rest') and not other_label == child_label:
                        other_UI = Label2UI[other_label]
                        df_fgl[other_UI] = false_values
                df_train = df_train.append(df_fgl)

    print(df_train.info())
    df_train.to_csv(original_datset_path  + os.path.sep + "train_gen" + str(year) + ".csv",index=False)