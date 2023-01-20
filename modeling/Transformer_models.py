import yaml
from torch.utils.data import TensorDataset
from slugify import slugify
import pickle
import os
import random
from sklearn.metrics import multilabel_confusion_matrix
from multiprocessing import freeze_support
from Transformer_functions import *
from other_functions import *

# DBM model development and evaluation

def run_experiment(settings, seed_val, balance_n = None, focus_labels = None):
    '''
    Run experiment described by the settings object for developing a model
    :param settings:        Configurations for the experiment as read from the a settings.yaml file
    :param seed_val:        A seed value for reproducibility of "random" choices
    :param balance_n:       A number indicating the desired negative-to-positive-instances ratio for each label.
                                e.g. "balance_n=1" indicates negative instances as many as positive the ones.
                                This is not strict, and different reatio may hold actualy as:
                                i) We use under-sampling only. Hence, for initial_ratio<balance_n no change is applied.
                                ii) We remove only "fully-negative articles". Hence, more negative instances may be kept
                                if they are positive for any other label.
    :param focus_labels:    A list of labels to consider. Only these labels and their "valid articles" are kept in datasets.
    :return:                None
    '''

    # Create folder(s) required for the experiment
    workingPath = settings["datasetPath"]
    experiment_suffix = ''
    focus_labels_suffix = ''
    if focus_labels is not None:
        focus_labels_suffix = "-".join(focus_labels)
    # Suffixes are used for different versions of a dataset
    if "experiment_suffix" in settings.keys():
        experiment_suffix = settings["experiment_suffix"]
    if "experiment_folder" in settings.keys():
        experiment_folder = settings["experiment_folder"]
        experiment_folder = experiment_folder + os.path.sep + str(balance_n)
    else:
        experiment_folder = workingPath + os.path.sep + experiment_suffix + "_" + focus_labels_suffix + "-" + slugify(str(datetime.now())) + os.path.sep + str(balance_n)

    if focus_labels is not None:
        experiment_folder = experiment_folder + os.path.sep + focus_labels_suffix

    experiment_folder = experiment_folder + os.path.sep + str(seed_val)

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    settings["experiment_folder"] = experiment_folder
    with open(workingPath + os.path.sep + "settings.yaml", 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)
    testFileRaw = settings["testFileRaw"]
    test_csv = workingPath + os.path.sep + testFileRaw[:testFileRaw.rfind(".")] + ".csv"
    trainFileRaw = settings["trainFileRaw"]
    training_csv = workingPath + os.path.sep + trainFileRaw[:trainFileRaw.rfind(".")] + experiment_suffix + ".csv"

    # Configurations for the experiment
    detailsfgNL = pd.read_csv(settings["datasetPath"] + os.path.sep + settings["detailsfgNL"])
    target_labels = list(detailsfgNL["Descr. UI"])
    modelName = settings["modelName"]     # modelName = 'bert-base-uncased'
    batch_size = int(settings["batch_size"])    # batch_size = 8
    epochs = int(settings["epochs"])    # epochs = 3
    learning_rate = float(settings["learning_rate"])
    prediction_threshold = float(settings["prediction_threshold"])
    # Set the seed value all over the place to make this reproducible.
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    project_name = settings["project_name"] + "_" + focus_labels_suffix
    if len(project_name) > 100:
        project_name = project_name[0:100]
    pos_weight_flag = settings["pos_weight"]
    loss_func_name = None
    if "loss_func_name" in settings.keys():
        loss_func_name = settings["loss_func_name"]

    # Compare labels of focus to the ones available in the CSV
    if focus_labels is not None:
        focus_target_labels = set.intersection(set(focus_labels), set(target_labels))
        if len(focus_target_labels) == len(focus_labels):
            # All focus labels were found int target labels
            target_labels = list(focus_target_labels)
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  "\t All focus labels found available!")
        else:
            if len(focus_target_labels) == 0:
                # No focus labels were found int target labels at all, continue with all available labels
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                      "\t None of the focus labels (", ",".join(focus_labels),
                      ") found available!")
            else:
                # Only some focus labels were found int target labels at all, continue with these labels
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                      "\t Only some of the focus labels (", ",".join(focus_labels),
                      ") found available!")
                target_labels = list(focus_target_labels)
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "\t Continue with target labels: ", ",".join(target_labels))

    wandb.init(project=project_name, entity="tasosnent")
    wandb.config.update({
        "experiment_folder": experiment_folder,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed_val": seed_val,
        "modelName": modelName,
        "prediction_threshold": prediction_threshold,
        "trainFileRaw": trainFileRaw,
        "pos_weight": pos_weight_flag,
        "loss_func_name": loss_func_name,
        "target_labels": " ".join(target_labels),
        "GPU_number": str(gpu_number),
        "balance_n": str(balance_n)
    })

    # Initialize the desired models to consider based on the experiment settings
    models = {}
    if settings["epoch_to_use"] == 'both':
        models['best'] = None
        models['prev'] = None
    elif settings["epoch_to_use"] == 'more':
        models['best'] = None
        models['prev'] = None
        models['current'] = None
    elif settings["epoch_to_use"] == 'previous':
        models['prev'] = None
    elif settings["epoch_to_use"] == 'best':
        models['best'] = None

    # Check for existing predictions to evaluate
    prediction_missing = False
    for model_ep in models.keys():
        model_folder = experiment_folder + os.path.sep + model_ep
        prediction_file = experiment_folder + os.path.sep + model_ep + os.path.sep + "preditions.pkl"
        golden_file = experiment_folder + os.path.sep + model_ep + os.path.sep + "golden.pkl"
        if os.path.isfile(prediction_file) and os.path.isfile(golden_file):
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  '\t Saved predictions are already avaiable for ', model_ep)
        else:
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  '\t Saved predictions missing for ', model_ep, ": ",
                 "\n\t", prediction_file, "\n\t and/or", golden_file)
            prediction_missing = True
    if not prediction_missing:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\t Saved predictions are already available for desired model(s)',
              '\n\t Skip training and prediction steps. Move directly to evaluation ')
    else:
        # Check for existing models to evaluate
        model_missing = False
        for model_ep in models.keys():
            model_folder = experiment_folder + os.path.sep + model_ep
            model_file = experiment_folder + os.path.sep + model_ep + os.path.sep + "model.pt"
            if os.path.isdir(model_folder) and os.path.isfile(model_file):
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                      '\t Saved model found for ', model_ep)
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                      '\tLoad a fine-tunned model for ', model_ep, " from ", model_file)
                models[model_ep] = torch.load(model_file, map_location=device)
            else:
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                      '\t Saved model missing for ', model_ep)
                model_missing = True
        if not model_missing:
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  '\t Saved model found for desired model(s) ',
                  '\n\tSkip steps 2 & 3')
        else:
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  '\t No saved model found for all desired model(s), proceed with fine-tunning')
            # Fine-tune new model
            models = fine_tune_on_dataset(settings, training_csv, target_labels, balance_n, loss_func_name, focus_labels_suffix)

    if settings["skip_test_evaluation"]:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\t Evaluation on test data is skiped due to skip_test_evaluation setting.')
    else:
        for model_ep in models.keys():
            model = models[model_ep]
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  "Step 4 - Evaluate the " + model_ep + " model: ")
            evaluate_on_dataset(settings, test_csv, target_labels, model, model_ep, focus_labels_suffix)

    wandb.finish()

def fine_tune_on_dataset(settings, training_csv, target_labels, balance_n, loss_func_name, focus_labels_suffix = ''):
    '''
    Fine-tune a pre-trained BERT model on the dataset provided.
    :param settings:        Configurations for the experiment as read from the a settings.yaml file
    :param training_csv:    The path to a CSV file with the training dataset.
                            Each row represents an article and the columns should include:
                                i) 'pmid':              The pmid of each article
                                ii) 'text':             The text of the article (i.e. title and abstract)
                                iii) 'valid_labels':    The new labels that are "valid" for this article as a
                                space-separated list of descriptor UIs (e.g. "D050681 D050685")
                                iv) One column named after the descriptor UI of each new label (e.g. 'D049920').
                                With value 1 if the article is annotated with this descriptor, value 0 otherwise.
    :param target_labels:   A list of labels to consider. (e.g. ['D049920'] )
    :param balance_n:       A number indicating the desired negative-to-positive-instances ratio for each label.
                                e.g. "balance_n=1" indicates negative instances as many as positive the ones.
                                This is not strict, and different reatio may hold actualy as:
                                i) We use under-sampling only. Hence, for initial_ratio<balance_n no change is applied.
                                ii) We remove only "fully-negative articles". Hence, more negative instances may be kept
                                if they are positive for any other label.
    :param loss_func_name:  A string indicating the loss function to be used. I.e one of 'BCE', 'FL', 'CBloss',
                            'R-BCE-Focal', 'NTR-Focal', 'DBloss-noFocal', 'CBloss-ntr', 'DBloss'.
                             If None, the standard BCEWithLogitsLoss is used.
    :return:    The fine-tuned BERT model.
    '''

    experiment_folder = settings["experiment_folder"]
    dataset_folder = settings["datasetPath"]
    modelName = settings["modelName"]  # modelName = 'bert-base-uncased'
    batch_size = int(settings["batch_size"])  # batch_size = 8
    pos_weight_flag = settings["pos_weight"]

    # Tokenized datasets and mask files depend on the tokenizer (i.e. the model)
    # but also depend on the balance_n value, the random seed, and focus labels
    train_data_dir = dataset_folder  + os.path.sep + str(balance_n)
    train_data_tokenized_dir = train_data_dir + os.path.sep + slugify(modelName)
    if not focus_labels_suffix == '':
        train_data_dir = train_data_dir + os.path.sep + focus_labels_suffix
        train_data_tokenized_dir = train_data_tokenized_dir + os.path.sep + focus_labels_suffix
    train_data_dir = train_data_dir + os.path.sep + str(seed_val)
    train_data_tokenized_dir = train_data_tokenized_dir + os.path.sep + str(seed_val)
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    if not os.path.exists(train_data_tokenized_dir):
        os.makedirs(train_data_tokenized_dir)
    train_tokenized_pickle_file = train_data_tokenized_dir + os.path.sep + "train_tokenized.pkl"
    train_masks_pickle_file = train_data_tokenized_dir + os.path.sep + "train_masks.pkl"
    val_tokenized_pickle_file = train_data_tokenized_dir + os.path.sep + "val_tokenized.pkl"
    val_masks_pickle_file = train_data_tokenized_dir + os.path.sep + "val_masks.pkl"

    # Golden/Weak labels are independent of tokenization but depend on balance_n and seed as well
    train_data_pickle_file = train_data_dir + os.path.sep + "train_data.pkl"
    val_data_pickle_file = train_data_dir + os.path.sep + "val_data.pkl"
    train_y_pickle_file = train_data_dir + os.path.sep + "train_y.pkl"
    val_y_pickle_file = train_data_dir + os.path.sep + "val_y.pkl"

    # Load pretrained model
    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "Step 1 - Load pretrained model: ", modelName)

    if os.path.isfile(train_tokenized_pickle_file) and os.path.isfile(train_masks_pickle_file) and \
            os.path.isfile(train_y_pickle_file) and os.path.isfile(val_tokenized_pickle_file) and \
            os.path.isfile(val_masks_pickle_file) and os.path.isfile(val_y_pickle_file) and \
            os.path.isfile(val_data_pickle_file):
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "\t Load already formatted data...",
              '\n\tSkip step 2')
        with open(train_tokenized_pickle_file, 'rb') as f:
            train_inputs = pickle.load(f)
        with open(train_masks_pickle_file, 'rb') as f:
            train_masks = pickle.load(f)
        with open(train_y_pickle_file, 'rb') as f:
            train_y = pickle.load(f)
        with open(val_tokenized_pickle_file, 'rb') as f:
            val_inputs = pickle.load(f)
        with open(val_masks_pickle_file, 'rb') as f:
            val_masks = pickle.load(f)
        with open(val_y_pickle_file, 'rb') as f:
            val_y = pickle.load(f)
        with open(val_data_pickle_file, 'rb') as f:
            val_data = pickle.load(f)
    else:
        # No tokenized training data already available, load and tokenize data
        if os.path.isfile(train_data_pickle_file) and os.path.isfile(val_data_pickle_file) :
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  "Step 2 - Load already saved training data... ", training_csv)
            # We already have the sentences saved
            with open(train_data_pickle_file, 'rb') as f:
                train_data = pickle.load(f)
            with open(val_data_pickle_file, 'rb') as f:
                val_data = pickle.load(f)
        else:
            # No saved sentences available, load them from the CSV
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  "Step 2 - Load and format training data: ", training_csv)
            # 3. Prepare and format the input data (tokenization, encoding etc)
            # Text must be split into tokens, and then these tokens must be mapped to their index in the tokenizer vocabulary.
            train_data, val_data = read_training_data(training_csv, target_labels, balance_n, float(settings["validation_ratio"]), seed_val)

            # Store the tokenized sentences
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  "\tSave Tokenized and formatted train sentences in pickle files")
            with open(train_data_pickle_file, 'wb') as f:
                pickle.dump(train_data, f)
            with open(val_data_pickle_file, 'wb') as f:
                pickle.dump(val_data, f)

        train_sentences, train_y = get_sentences(train_data, target_labels)
        val_sentences, val_y = get_sentences(val_data, target_labels)

        # Load the BERT tokenizer.
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\tLoading tokenizer...')
        tokenizer = BertTokenizer.from_pretrained(modelName, do_lower_case=True)
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\tDone!')

        if settings["tokenization_example"]:
            # print a tokenization example for the first sentence
            print_tokenization_example(train_data,tokenizer)

        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\tConvert data into sentence and label lists...')
        # We are required to:
        # Add special tokens to the start and end of each sentence.
        # Pad & truncate all sentences to a single constant length (512 tokens).
        # Explicitly differentiate real tokens from padding tokens with the "attention mask".
        print('\tSentences length:')
        print('\tTraining: ', len(train_sentences))
        print('\tValidation: ', len(val_sentences))

        # find the maximum and average sentence length
        print_token_stats(train_sentences, tokenizer)

        # format all input data
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\tTokenize and format sentences...')

        train_inputs, train_masks = format_for_bert(train_sentences, tokenizer)
        val_inputs, val_masks = format_for_bert(val_sentences, tokenizer)

        # Store the tokenized sentences
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "\tSave Tokenized and fromated train sentences in pickle files")
        with open(train_tokenized_pickle_file, 'wb') as f:
            pickle.dump(train_inputs, f)
        with open(train_masks_pickle_file, 'wb') as f:
            pickle.dump(train_masks, f)
        with open(train_y_pickle_file, 'wb') as f:
            pickle.dump(train_y, f)
        with open(val_tokenized_pickle_file, 'wb') as f:
            pickle.dump(val_inputs, f)
        with open(val_masks_pickle_file, 'wb') as f:
            pickle.dump(val_masks, f)
        with open(val_y_pickle_file, 'wb') as f:
            pickle.dump(val_y, f)

    if settings["skip_model_training"]:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\t Fine-tunning on training data is skiped due to skip_model_training setting.')
        models = {}
    else:
        # Combine into TensorDataset.
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\tCreate TensorDatasets...')
        train_y[train_y == -1] = 2
        val_y[val_y == -1] = 2
        train_dataset = TensorDataset(train_inputs, train_masks, train_y)
        val_dataset = TensorDataset(val_inputs, val_masks, val_y)

        # Calculate metrics for imbalance handling
        pos_weight = None
        class_freq = None
        train_num = None
        if pos_weight_flag or loss_func_name is not None:
            pos_weight, class_freq, train_num = imbalance_counter(train_y, target_labels)

        # Create dataloaders
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\tCreate DataLoaders...')
        # The DataLoader needs to know our batch size for training, so we specify it
        # here. For fine-tuning BERT on a specific task, the authors recommend a batch
        # size of 16 or 32.
        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = create_dataloader(train_dataset, batch_size, True)
        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = create_dataloader(val_dataset, batch_size)

        wandb.config.update({
            "train_data_len": len(train_dataloader),
            "validation_data_len": len(validation_dataloader)
        })

        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "Step 3 - Fine-tune the model: ")
        # Fine Tune BioBERT model
        print("\tpos_weight:", pos_weight)
        print("\tclass_freq:", class_freq)
        print("\ttrain_num:", train_num)
        print("\tloss_func_name:", loss_func_name)
        models = fine_tune(settings, target_labels, train_dataloader, device, validation_dataloader, pos_weight=pos_weight, class_freq=class_freq, train_num=train_num, loss_func_name=loss_func_name)

        for model_name in models.keys():
            model = models[model_name]
            # [Optional] Save the fine-tuned model
            model_path = experiment_folder + os.path.sep + model_name
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if settings["save_model"]:
                print('\tSave the fine-tuned ' + model_name + '_model in ', model_path)
                torch.save(model, model_path + os.path.sep + "model.pt")

            # Evaluate model on validation
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  "Step 3+ - Evaluate " + model_name + "_model on validation dataset: ")
            predictions, true_labels = predict(validation_dataloader, model, device, float(settings["prediction_threshold"]))
            predictions = validity_filtering(val_data, predictions, target_labels)
            report_json = evaluation_report(predictions, true_labels, modelName, target_labels,"validation")
            save_evaluation_report(report_json, target_labels,
                                   model_path + os.path.sep + "val_report.json",
                                   model_path + os.path.sep + "val_report.csv")
            wandb.log({"val_p": str(round(report_json["macro avg"]["precision"],3)),
                       "val_r": str(round(report_json["macro avg"]["recall"],3)),
                       "val_f1": str(round(report_json["macro avg"]["f1-score"],3)),
                       "val_support": str(round(report_json["macro avg"]["support"],3))})

    return models

def evaluate_on_dataset(settings, test_csv, target_labels, model, model_ep, focus_labels_suffix = '', label_to_skip = None ):
    '''
    :param settings:        Configurations for the experiment as read from the a settings.yaml file
    :param test_csv:        The path to a CSV file with the test dataset.
                            Each row represents an article and the columns should include:
                                i) 'pmid':              The pmid of each article
                                ii) 'text':             The text of the article (i.e. title and abstract)
                                iii) 'valid_labels':    The new labels that are "valid" for this article as a
                                space-separated list of descriptor UIs (e.g. "D050681 D050685")
                                iv) One column named after the descriptor UI of each new label (e.g. 'D049920').
                                With value 1 if the article is annotated with this descriptor, value 0 otherwise.
    :param target_labels:   A list of labels to consider. (e.g. ['D049920'] )
    :param model:           The fine-tuned BERT model to be evaluated on this dataset.
    :param model_ep:        A string identifying the epoch of the fine-tuned BERT model (e.g. "best", of "prev").
    :return:    None. The evaluation report is saved on a CSV and a JSON file.
    '''
    experiment_folder = settings["experiment_folder"]
    model_folder = experiment_folder + os.path.sep + model_ep
    dataset_folder = settings["datasetPath"]
    modelName = settings["modelName"]  # modelName = 'bert-base-uncased'
    batch_size = int(settings["batch_size"])  # batch_size = 8
    prediction_threshold = float(settings["prediction_threshold"])

    # Tokenized datasets and mask files depend on the tokenizer (i.e. modelname)
    test_data_dir = dataset_folder + os.path.sep + slugify(modelName)
    if not focus_labels_suffix == '':
        test_data_dir = test_data_dir  + os.path.sep + focus_labels_suffix
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    test_tokenized_pickle_file = test_data_dir + os.path.sep + "test_tokenized.pkl"
    test_masks_pickle_file = test_data_dir + os.path.sep + "test_masks.pkl"
    # Golden/Weak labels are independent of tokenization
    test_sentences_pickle_file = dataset_folder + os.path.sep + "test_sentences.pkl"
    test_y_pickle_file =  dataset_folder + os.path.sep + "test_y.pkl"
    # However, test_y_pickle_file depends on focus_labels_suffix
    if not focus_labels_suffix == '':
        test_sentences_pickle_file = dataset_folder + os.path.sep  + focus_labels_suffix + "_test_sentences.pkl"
        test_y_pickle_file =  dataset_folder + os.path.sep + focus_labels_suffix + "_test_y.pkl"
    # Predictions done by the trained model
    prediction_pickle_file = model_folder + os.path.sep + "preditions.pkl"
    predictionF_pickle_file = model_folder + os.path.sep + "preditions_filtered.pkl"
    golden_pickle_file = model_folder + os.path.sep + "golden.pkl"
    # Metrics on the predictions
    report_json_file = model_folder + os.path.sep + "report.json"
    report_csv_file = model_folder + os.path.sep + "report.csv"


    # Evaluate the model on test data
    if os.path.isfile(predictionF_pickle_file) and os.path.isfile(golden_pickle_file):
        # Load stored predictions and true labels
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\tLoad filtered predictions ')
        with open(predictionF_pickle_file, 'rb') as f:
            predictions = pickle.load(f)
        with open(golden_pickle_file, 'rb') as f:
            true_labels = pickle.load(f)
    else:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              '\tLoad test data from ', test_csv )
        test_data = read_data(test_csv, target_labels)
        print('\ttotal test:', test_data.shape)
        # print(test_data.head())
        if os.path.isfile(prediction_pickle_file) and os.path.isfile(golden_pickle_file):
            # Load stored predictions and true labels
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  '\tLoad unfiltered predictions ')
            with open(prediction_pickle_file, 'rb') as f:
                predictions = pickle.load(f)
                print(predictions)
            with open(golden_pickle_file, 'rb') as f:
                true_labels = pickle.load(f)
        else:
            # We already have the tokenized test data, proceed with data loaders etc
            if os.path.isfile(test_tokenized_pickle_file) and os.path.isfile(test_masks_pickle_file) and os.path.isfile(test_y_pickle_file):
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                      "\t Load already formated test data...")
                with open(test_tokenized_pickle_file, 'rb') as f:
                    test_inputs = pickle.load(f)
                    # print("test_inputs",test_inputs.size())
                with open(test_masks_pickle_file, 'rb') as f:
                    test_masks = pickle.load(f)
                    # print("test_masks",test_masks.size())
                with open(test_y_pickle_file, 'rb') as f:
                    test_y = pickle.load(f)
                    # print("test_y",test_y.size())
            else:
                # We don't have tokenized test data, proceed with loading data for tokenization
                if os.path.isfile(test_sentences_pickle_file) and os.path.isfile(test_y_pickle_file):
                    # We already have the test sentences, proceed with tokenization
                    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                          "\t Load already saved test sentences...")
                    with open(test_sentences_pickle_file, 'rb') as f:
                        test_sentences = pickle.load(f)
                    with open(test_y_pickle_file, 'rb') as f:
                        test_y = pickle.load(f)
                else:
                    # We don't have the sentences, get them from the CSV
                    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                          '\tConvert data into sentence and label lists...')
                    test_sentences, test_y = get_sentences(test_data, target_labels)
                    print('\tSentences length:',len(test_sentences))
                    test_y[test_y == -1] = 2

                    # Store the test sentences
                    with open(test_sentences_pickle_file, 'wb') as f:
                        pickle.dump(test_sentences, f)
                    with open(test_y_pickle_file, 'wb') as f:
                        pickle.dump(test_y, f)

                # Load the BERT tokenizer.
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                      '\tLoading tokenizer...')
                tokenizer = BertTokenizer.from_pretrained(modelName, do_lower_case=True)
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                      '\tDone!')

                # find the maximum and average sentence length
                print_token_stats(test_sentences, tokenizer)

                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                      '\tTokenize and format sentences...')
                test_inputs, test_masks = format_for_bert(test_sentences, tokenizer)

                # Store the tokenized test
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                      "\tSave Tokenized and fromated sentences in:" + test_tokenized_pickle_file + " & " +test_masks_pickle_file)
                with open(test_tokenized_pickle_file, 'wb') as f:
                    pickle.dump(test_inputs, f)
                with open(test_masks_pickle_file, 'wb') as f:
                    pickle.dump(test_masks, f)

            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  '\tCreate TensorDataset...')
            # Create the test DataLoader.
            prediction_data = TensorDataset(test_inputs, test_masks, test_y)
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  '\tCreate DataLoader...')
            prediction_dataloader = create_dataloader(prediction_data, batch_size)

            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  '\tPredict on ', test_csv)
            # Prediction on test set
            predictions, true_labels = predict(prediction_dataloader,model,device,prediction_threshold)

            # Save the predictions and golden labels
            # Store the predictions
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  "\tSave predictions in:" + prediction_pickle_file)
            with open(prediction_pickle_file, 'wb') as f:
                pickle.dump(predictions, f)
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  "\tSave golden in:" + golden_pickle_file)
            with open(golden_pickle_file, 'wb') as f:
                pickle.dump(true_labels, f)

        # Filtering "invalid" predictions (if needed)
        print("size test_data:", len(test_data), " ", test_data.shape)
        print("size predictions:", len(predictions), " ", predictions.shape)

        predictions = validity_filtering(test_data, predictions, target_labels)

        # Store the predictions
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "\tSave filtered predictions in:" + predictionF_pickle_file)
        with open(predictionF_pickle_file, 'wb') as f:
            pickle.dump(predictions, f)

    # handle evaluation of models trained by the C2F method
    pred_num_rows, pred_num_cols = predictions.shape
    true_num_rows, true_num_cols = true_labels.shape
    if label_to_skip is not None:
        predictions = np.delete(predictions, label_to_skip, 1)
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "\tpredictions for column",str(label_to_skip)," are removed for evaluation.")

    # Evaluate and print scores
    print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
          "\tPrint evaluation scores:")
    report_json = evaluation_report(predictions, true_labels, modelName, target_labels, "test")
    print(report_json)
    matrix = multilabel_confusion_matrix(true_labels, predictions)

    save_evaluation_report(report_json, target_labels, report_json_file, report_csv_file, matrix)

    # Keep an additional copy of the settings in the experiment_folder
    with open(experiment_folder + os.path.sep + "settings.yaml", 'w') as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)

def run_experiments(settings, seed_val, balance_n = None):
    workingPath = settings["datasetPath"]
    experiment_suffix = ''
    if "experiment_suffix" in settings.keys():
        experiment_suffix = settings["experiment_suffix"]
    if "experiment_folder" in settings.keys():
        experiment_folder = settings["experiment_folder"]
    else:
        experiment_folder = workingPath + os.path.sep + experiment_suffix + slugify(str(datetime.now()))
        os.makedirs(experiment_folder)
        settings["experiment_folder"] = experiment_folder
        with open(experiment_folder + os.path.sep + "settings.yaml", 'w') as outfile:
            yaml.dump(settings, outfile, default_flow_style=False)

    # Handle focus labels
    focus_labels = None
    # Run for each focus-label group independently
    if "focus_labels" in settings.keys():
        focus_labels = settings["focus_labels"]
        print("focus_labels : ", focus_labels)
        for i in range(len(focus_labels)):
            focus_label_group = focus_labels[i]
            print("focus_label group ", i, " : ", focus_label_group)
            run_experiment(copy.deepcopy(settings), seed_val, balance_n, focus_label_group)
    else:
        # Run for all available labels to create a single multi-label model.
        run_experiment(copy.deepcopy(settings), seed_val, balance_n, None)

# Start the main here

print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
      "Step 0 - Begin the process")

if __name__ == '__main__':
    freeze_support()
    if len(sys.argv) == 2:
        settings_file = sys.argv[1]
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "Run with settings.py file at: " + settings_file)
    else:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "No settings.py file found as argument, try to find the file in the project folder.")
        settings_file = "../modeling/settings.yaml"

    # Read the settings
    settings = {}
    # settings = yaml.load(settings_file, Loader=yaml.FullLoader)
    with open(settings_file) as parameters:
      settings = yaml.safe_load(parameters)

    print('settings:',settings)

    import wandb
    print('wandb key:', settings['wandb_key'])
    wandb.login()
    # To login in wandb a key is also needed.
    gpu_number = None
    # If there's a GPU available...
    if torch.cuda.is_available():
        gpu_number = int(settings['gpu_number'])

        # Tell PyTorch to use the GPU.
        gpu = "cuda:"+str(gpu_number)
        print(gpu)
        device = torch.device(gpu)

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU {' + str(gpu_number) + '} :', torch.cuda.get_device_name(gpu_number))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Handle experiment folders
    seed_vals = settings["seed_vals"]
    for seed_val in seed_vals:
        seed_val = int(seed_val)
        if "balance_ns" in settings.keys():
            balance_ns = settings["balance_ns"]
            for balance_n in balance_ns:
                balance_n = int(balance_n)
                run_experiments(settings, seed_val, balance_n)
        else:
            run_experiments(settings, seed_val)


