import os
import pandas as pd

# Combine "ALO3" predictions provided by RetroBM (i.e. label_matrix_train_minority_voter_X.csv) with corresponding DBM dataset (i.e. train_X.csv) to use the enhanced supervision.
# The RetroBM method for developing datasets is available here: https://github.com/ThomasChatzopoulos/MeSH_retrospective_dataset

# main_path where the different DBM dataset folders (as created by Datasets.py) are stored.
main_path = "/home/RetroData"
years = {
    # The file names for the DBM-dataset folders to be processed fro merging "ALO3" predictions into the
    2006: "Dataset_SI_2006_2022...",
    2007: "Dataset_SI_2007_2022...",
    2008: "Dataset_SI_2008_2022...",
#     ...
}

for year in years:
    folder = years[year]

    path = main_path + os.path.sep + folder + os.path.sep
    year = str(year)

    train = pd.read_csv(path + os.path.sep + "train_"+year+".csv")
    minority = pd.read_csv(path + os.path.sep + "label_matrix_train_minority_voter_"+year+"_filtered.csv")

    train = train[["text","valid_labels","CUIs"]]
    print(len(train))
    print(train.iloc[1456])
    # print(minority.iloc[1456])

    result = pd.concat([train, minority], axis=1)
    print("----")
    print(len(result))
    print(result.iloc[1456])

    result.to_csv(path + os.path.sep + "train_"+year+"_minority_filtered.csv")
