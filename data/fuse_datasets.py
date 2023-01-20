import os
import pandas as pd

# Fuse original DBM datasets (i.e. train_X.csv etc from different years) into a single dataset

# main_path where the different DBM dataset folders (as created by Datasets.py) are stored.
main_path = "/home/RetroData"
paths = {
     # The specific paths for each dataset to be considered
     "2007": main_path + os.path.sep + "Dataset_SI_2007_2022..." ,
     "2008": main_path + os.path.sep + "Dataset_SI_2008_2022..." ,
     "2009": main_path + os.path.sep + "Dataset_SI_2009_2022..."
     # ...
     }
train_fused = None
train_minority_fused = None
test_fused = None
details_fused = None
print("year\t train\t test\t descr")
for year in paths.keys():
    path = paths[year]

    train = pd.read_csv(path + os.path.sep + "train_"+year+".csv")
    train_minority = pd.read_csv(path + os.path.sep + "train_"+year+"_minority.csv")
    test = pd.read_csv(path + os.path.sep + "test_"+year+".csv")
    details = pd.read_csv(path + os.path.sep + "UseCasesSelected_"+year+".csv")
    if train_fused is None:    # this is the first dataset for fusion. Nothing to fuse with
        print(year,"\t", len(train),"\t",len(test),"\t",len(details) )
        train_fused = train
        train_minority_fused = train_minority
        test_fused = test
        details_fused = details
    else:    # some data are already available. Fuse new data with them.
        print(year,"\t", len(train),"\t",len(test),"\t",len(details) )
        train_fused = train_fused.join(train.set_index(['pmid','text','CUIs']), on = ['pmid','text','CUIs'], how= "outer", rsuffix= year)
        # merge validity values for different labels and remove relict year-specific fields fields
        train_fused['valid_labels'] = train_fused['valid_labels'].astype(str) + train_fused['valid_labels'+year].astype(str)
        train_fused['valid_labels'] = train_fused['valid_labels'].str.replace('nan', '')
        train_fused = train_fused.drop(columns=['valid_labels'+year])
        train_fused = train_fused.fillna(0)
        # print(train_fused['valid_labels'])

        train_minority_fused = train_minority_fused.join(train.set_index(['pmid','text','CUIs']), on = ['pmid','text','CUIs'], how= "outer", rsuffix= year)
        # merge validity values for different labels and remove relict year-specific fields fields
        train_minority_fused['valid_labels'] = train_minority_fused['valid_labels'].astype(str) + train_minority_fused['valid_labels'+year].astype(str)
        train_minority_fused['valid_labels'] = train_minority_fused['valid_labels'].str.replace('nan', '')
        train_minority_fused = train_minority_fused.drop(columns=['valid_labels'+year])
        train_minority_fused = train_minority_fused.fillna(0)

        test_fused = test_fused.join(test.set_index(['pmid','text','CUIs']), on = ['pmid','text','CUIs'], how= "outer", rsuffix= year)
        # merge validity values for different labels and remove relict year-specific fields fields
        test_fused['valid_labels'] = test_fused['valid_labels'].astype(str) + test_fused['valid_labels'+year].astype(str)
        test_fused['valid_labels'] = test_fused['valid_labels'].str.replace('nan', '')
        test_fused = test_fused.drop(columns=['valid_labels'+year])
        test_fused = test_fused.fillna(0)

        # print(details.columns)
        details_fused_keys = ['Descr. UI', 'Descr. Name', 'PH count', '#Ct', '#Parent Ds', 'Parent Ds', 'PHs', 'PHs UIs', '#PHex', 'PHex UIs']
        details_fused = details_fused.join(details.set_index(details_fused_keys), on = details_fused_keys, how= "outer", rsuffix= year)
        details_fused = details_fused[details_fused_keys]

print("total", "\t", len(train_fused), "\t", len(test_fused), "\t", len(details_fused))

# Check for train-test overlap
overlap = pd.merge(train_fused, test_fused, how='inner', on=['pmid'])
print("overlap", len(overlap['pmid']))

# Remove from training data articles that are overlapping with the test
train_fused = train_fused[~train_fused['pmid'].isin(overlap['pmid'].values)]

print("total2", "\t", len(train_fused), "\t", len(test_fused), "\t", len(details_fused))

path_fused = main_path + os.path.sep + "Dataset_SI_fused_v2"
if not os.path.exists(path_fused):
    os.makedirs(path_fused)
years = '_'.join(paths.keys())
overlap.to_csv(path_fused + os.path.sep + "overlap_"+years+".csv", index=False)
train_fused.to_csv(path_fused + os.path.sep + "train_"+years+".csv", index=False)
train_minority_fused.to_csv(path_fused + os.path.sep + "train_"+years+"_minority.csv", index=False)
print("train_minority_fused.sum() :\n", str(train_fused[list(details_fused["Descr. UI"])].sum()))
test_fused.to_csv(path_fused + os.path.sep + "test_"+years+".csv", index=False)
# train positives
positives = train_fused[list(details_fused["Descr. UI"])].sum()
positives.to_csv(path_fused + os.path.sep + "train_positives_"+years+".csv")
# test positives
positives = test_fused[list(details_fused["Descr. UI"])].sum()
positives.to_csv(path_fused + os.path.sep + "test_positives_"+years+".csv")
# labels per doc
total = train_fused.loc[:, list(details_fused["Descr. UI"])].sum(axis=1)
labels_per_doc = total.value_counts()
print("train_data labels_per_doc :\n", str(labels_per_doc))
total = test_fused.loc[:, list(details_fused["Descr. UI"])].sum(axis=1)
labels_per_doc = total.value_counts()
print("test_data labels_per_doc :\n", str(labels_per_doc))
details_fused.to_csv(path_fused + os.path.sep + "UseCasesSelected_"+years+".csv", index=False)
