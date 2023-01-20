import pickle, os
import pandas as pd, json
import yaml

# Convert a DBM dataset into the format required for C2F

# Read the settings
settings_file = open("../data/settings_c2f.yaml")
settings = yaml.load(settings_file, Loader=yaml.FullLoader)
datasets = settings["datasets"]

for original_datset_path in datasets:
    year = original_datset_path.split("Dataset_SI_")[1].split("_")[0]
    print(year)
    detailsfgNL = pd.read_csv(original_datset_path + os.path.sep + "UseCasesSelected_" + str(year) + ".csv")
    # new_dataset_path = original_datset_path + os.path.sep + "fgsi" + str(year) + "u"
    new_dataset_path = original_datset_path + os.path.sep + "fgsi" + str(year) + "uef"
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)
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
    # clean labels (the two list are parallel ie of the same size)
    for i in range(len(fgNL_labels)):
        fgNL_labels[i] = fgNL_labels[i].replace(" ", "_").replace(",", "")
        cgL_labels[i] = cgL_labels[i].replace(" ", "_").replace(",", "")
    UI2Label = dict(zip(fgNL + cgL,fgNL_labels + cgL_labels))
    Label2UI = dict(zip(fgNL_labels + cgL_labels,fgNL + cgL))
    fgNL_UI2parentLabel = dict(zip(fgNL, cgL_labels))
    print(fgNL_UI2parentLabel)

    # create parent_to_child
    # {"Dolphins": ["Bottle-Nosed_Dolphin", "Whale_Killer"], "Whales": ["Humpback_Whale", "Bowhead_Whale", "Beluga_Whale", "Sperm_Whale"]}
    parent_to_child = {}
    for l in fgNL:
      parent = fgNL_UI2parentLabel[l]
      if parent in parent_to_child.keys():
        parent_to_child[parent].append(UI2Label[l])
      else:
        parent_to_child[parent] = [UI2Label[l]]
    print(parent_to_child)

    with open(new_dataset_path + os.path.sep + "parent_to_child.json", "w+") as f:
        json.dump(parent_to_child, f)

    # create df_coarse.pkl
    #                                                     text     label
    # 0      (reuters) - carlos tevez sealed his move to ju...    sports
    # df_train  = pd.read_csv(original_datset_path + os.path.sep + "train_" + str(year) + ".csv")
    # df_train  = pd.read_csv(original_datset_path + os.path.sep + "train_" + str(year) + "_minority.csv")
    df_train  = pd.read_csv(original_datset_path + os.path.sep + "train_" + str(year) + "_minority_filtered.csv")
    print(df_train.head())

    # Just on valid instance is accepted for each article, so keep just the first one
    labels_per_instance = []
    for index, row in df_train.iterrows():
        # print(fgNL_UI2parentLabel[row['valid_labels'].split(' ')[0]])
        ui = row['valid_labels'].split(' ')[0]
        parent = ""
        if ui in fgNL_UI2parentLabel.keys():
          parent = fgNL_UI2parentLabel[ui]
        labels_per_instance.append(parent)

    df_train_coarse = df_train.assign(label = labels_per_instance)

    print(df_train_coarse.head())

    with open(new_dataset_path + os.path.sep + "df_coarse.pkl", 'wb') as f:
        pickle.dump(df_train_coarse, f)

    # create exclusive
    #                                                   text   label
    # 0    berlin — germany's storied bayern munich socce...  sports

    for fgNLui in fgNL:
        # check and handle additional _rest labels (represent unlabelled documents)
        label = UI2Label[fgNLui]
        if label.endswith('_rest'):
            paretn_label = fgNL_UI2parentLabel[fgNLui]
            parent_ui = Label2UI[paretn_label]
            # Keep all valid documents
            df_train_fine = df_train_coarse.loc[df_train_coarse['valid_labels'].str.contains(parent_ui)]
            # get siblings
            siblings = list(parent_to_child[paretn_label])
            siblings.remove(label)
            # keep unlabelled articles articles
            for sibling in siblings:
                sibling_ui = Label2UI[sibling]
                df_train_fine = df_train_fine.loc[df_train_fine[str(sibling_ui)] != 1]
        else:
            df_train_fine = df_train_coarse.loc[df_train_coarse[str(fgNLui)] == 1]
        # print(labels)
        # print(df_train_fine.head())

        path = new_dataset_path + os.path.sep + "exclusive" + os.path.sep + "1it"
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
        with open(path + "/" + label + ".pkl", 'wb') as f:
            pickle.dump(df_train_fine, f)

    # df fine
    #                                                   text   label
    # 0    berlin — germany's storied bayern munich socce...  soccer
    df_test  = pd.read_csv(original_datset_path + os.path.sep + "test_" + str(year) + ".csv")
    print(df_test.head())

    labels_per_instance = []
    for index, row in df_test.iterrows():
        # get valid labels
        ui = row['valid_labels'].split(' ')[0]
        parent = ""
        if ui in fgNL_UI2parentLabel.keys():
          parent = fgNL_UI2parentLabel[ui]

        row_label = parent + "_rest"
        for fgNLui in fgNL:
          if fgNLui in row.keys() and row[fgNLui] == 1:
            row_label = UI2Label[fgNLui]
        labels_per_instance.append(row_label)

    df_test_fine = df_test.assign(label = labels_per_instance)

    print(df_test_fine.head())

    with open(new_dataset_path + os.path.sep + "df_fine.pkl", 'wb') as f:
        pickle.dump(df_test_fine, f)