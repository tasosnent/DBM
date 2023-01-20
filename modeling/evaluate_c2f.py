from Transformer_models import *
from BertModeling import *
from other_functions import *

# Evaluate C2F model predictions

if __name__ == '__main__':
    freeze_support()
    if len(sys.argv) == 2:
        settings_file = sys.argv[1]
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "Run with settings.py file at: " + settings_file)
    else:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "No settings.py file found as argument, try to find the file in the project folder.")
        settings_file = "settings_c2f.yaml"

    # Read the settings
    settings = {}
    # settings = yaml.load(settings_file, Loader=yaml.FullLoader)
    with open(settings_file) as parameters:
      settings = yaml.safe_load(parameters)

    print('settings:',settings)

    gpu_number = None
    # If there's a GPU available...
    if torch.cuda.is_available():
        gpu_number = int(settings['gpu_number'])

        # Tell PyTorch to use the GPU.
        gpu = "cuda:" + str(gpu_number)
        print(gpu)
        device = torch.device(gpu)

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU {' + str(gpu_number) + '} :', torch.cuda.get_device_name(gpu_number))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    import Transformer_models
    Transformer_models.device = device
    test_csv = settings["datasetPath"] + os.path.sep + settings["testcsv"]
    model_ep = "best"

    parent_child = {}
    experiment_folder = settings["experiment_folder"]
    with open(experiment_folder + os.path.sep + "parent_to_child.json", "r+") as f:
        parent_child = json.load(f)
    detailsfgNL = pd.read_csv(settings["datasetPath"] + os.path.sep + settings["detailsfgNL"])
    target_label_UIs = list(detailsfgNL["Descr. UI"])
    target_label_names = list(detailsfgNL["Descr. Name"])
    for i in range(len(target_label_names)):
        target_label_names[i] = target_label_names[i].replace(" ", "_").replace(",", "")
    Label2UI = dict(zip(target_label_names, target_label_UIs))

    report_df = pd.DataFrame(columns=["label","precision","recall", "f1-score", "support", "tn","fp","fn","tp"])
    for parent in parent_child.keys():

        label_to_index = {}
        label_to_index_file = experiment_folder + os.path.sep + "dict_" + parent + ".json"
        with open(label_to_index_file, "r+") as f:
            label_to_index = json.load(f)
        labels = label_to_index.keys()
        indexes = label_to_index.values()
        index_to_label = dict(zip(labels, indexes))

        # find the correct order of labels and remove _rest (if added)
        focus_labels = []
        label_to_skip = None
        for lab in label_to_index.keys():
            if lab in Label2UI.keys():
                focus_labels.append(lab)
            else:
                # set label_to_skip to be the index of the _rest label for unlabelled docs
                label_to_skip = label_to_index[lab]
        # reorder the focus labels based on the C2F index considering the removal of _rest label
        focus_labels_reordered = list(range(len(focus_labels)))
        for lab in focus_labels:
            index = label_to_index[lab]
            if label_to_skip is not None:
                if label_to_skip < index:
                    index = index - 1
            focus_labels_reordered[index] = lab
        # Convert labels (names) to UIs
        focus_labels = []
        for lab in focus_labels_reordered:
            focus_labels.append(Label2UI[lab])
        # In case of a single label of interest, without an additional _rest label.
        # the model still outputs two predictions, we are only interested in positives (column 1)
        if len(focus_labels) == 1 and label_to_skip is None:
            label_to_skip = 0
        model_path = experiment_folder + os.path.sep + "model_" + parent + ".pt"
        # model = torch.load(model_path, map_location = device)
        model = None
        focus_labels_suffix = "-".join(focus_labels)
        # The settings file should include:
        # "datasetPath" for reading and writing the test data
        # "experiment_folder" for storing results
        settings["experiment_folder"] = experiment_folder + os.path.sep + "evaluation" + os.path.sep + focus_labels_suffix
        if not os.path.exists(settings["experiment_folder"]):
            os.makedirs(settings["experiment_folder"])
        if not os.path.exists(settings["experiment_folder"] + os.path.sep + model_ep):
            os.makedirs(settings["experiment_folder"] + os.path.sep + model_ep)

        evaluate_on_dataset(settings, test_csv, focus_labels, model, model_ep, focus_labels_suffix, label_to_skip)

        model_folder = settings["experiment_folder"] + os.path.sep + model_ep
        report_json_file = model_folder + os.path.sep + "report.json"
        report = {}
        with open(report_json_file, "r+") as f:
            report = json.load(f)
        if len(focus_labels) == 1:
            # The case of binary prediction, keep positive predictions only
            result = report["1.0"]
            d = {
                # "label","precision","recall", "f1-score", "support"
                "label": focus_labels[0],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1-score": result["f1-score"],
                "support": result["support"],
                "tn": result["tn"],
                "fp": result["fp"],
                "fn": result["fn"],
                "tp": result["tp"]
            }
            report_df = report_df.append(d, ignore_index=True)
        if len(focus_labels) > 1:
            # The case of more than one prediction labels, keep all of them
            for i in range(len(focus_labels)):
                label = focus_labels[i]
                result = report[str(i)]
                d = {
                    # "label","precision","recall", "f1-score", "support"
                    "label": focus_labels[i],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1-score": result["f1-score"],
                    "support": result["support"],
                    "tn": result["tn"],
                    "fp": result["fp"],
                    "fn": result["fn"],
                    "tp": result["tp"]
                }
                report_df = report_df.append(d, ignore_index=True)

    report_df.to_csv(experiment_folder + os.path.sep + "test_report.csv")