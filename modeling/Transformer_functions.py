import torch
from datetime import datetime
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
import time
from sklearn.metrics import f1_score, classification_report
import wandb
import copy
from BertModeling import BertForMultiLabelSequenceClassification
from other_functions import *

# Basic functions for model development and evaluation

def format_for_bert(sentences_list, tokenizer, verbose = True):
    '''
    Tokenize all of the sentences in sentences_list and map the tokens to their word IDs.
    :param sentences_list:
    :param tokenizer:
    :param verbose:
    :return:
    '''
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences_list:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # max_length = 256,           # Pad & truncate all sentences.
            max_length=512,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        if verbose and len(input_ids)%10000 == 0:
            ratio = round((len(input_ids)/len(sentences_list))*100,2)
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  len(input_ids), " sentences processed (",ratio,"%)")
    if verbose and len(input_ids)%10000 == 0:
        ratio = round((len(input_ids)/len(sentences_list))*100,2)
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              len(input_ids), " sentences processed (",ratio,"%)")
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

def get_sentences(dataset, target_labels):
    '''
    Read a dataframe dataset and return a list of texts (stentences) and a tensor of "predictions" (y values)
    :param dataset:
    :param target_labels:
    :return:
    '''
    sentences_list = []
    yy = []
    for i, row in dataset.iterrows():
      # print(row)
      dx = row['text']
      y = []
      for target_label in target_labels:
        y.append(row[target_label])
      # print(y)
      yy.append(y)

      sentences_list.append(dx)
    yy = torch.Tensor(yy)
    yy = yy.double()
    return sentences_list, yy

def read_training_data(training_csv, target_labels,balance_n, validation_ratio, seed):
    '''
    Read data rom a CSV file and prepare a training datasets, splitting into train and validation parts.
    :param training_csv:
    :param target_labels:
    :param balance_n:
    :param validation_ratio:
    :param seed:
    :return:
    '''
    input_data = read_data(training_csv, target_labels)

    # print(input_data.head())

    print('\ttotal inputdata:', input_data.shape)

    print("\tpositive_instances per label :")
    positive_instances = input_data[target_labels].sum()
    print(positive_instances)

    print('\ttotal data:', input_data.shape)
    input_data = balance_dataset(input_data, target_labels, balance_n, seed)
    # Divide the dataset by randomly selecting samples 90/10.
    input_data = input_data.sample(frac=1, random_state=seed)
    train_data, val_data = np.split(input_data, [int(validation_ratio * len(input_data))])
    print('\ttotal train:', train_data.shape)
    stats_on_instace_validity(input_data, target_labels)
    print('\ttotal balanced train:', train_data.shape)

    print('\ttotal val:', val_data.shape)
    stats_on_instace_validity(input_data, target_labels)

    return train_data, val_data

def balance_dataset(input_data, target_labels, balance_n, seed):
    # remove unwanted samples (e.g. for undersampling)
    positive_instances = input_data[target_labels].sum()

    remove_pmids = []
    if balance_n is not None :
        # balance the dataset removing fully negative articles
        input_data["label_count"] = input_data.loc[:, target_labels].sum(axis=1)
        all_negative_instances = input_data.loc[input_data['label_count'] == 0].copy()
        print("\tlen(input_data) :", str(len(input_data)))
        print("\tlen(All negative_instances) :", str(len(all_negative_instances)))

        all_positive_instances = len(input_data) - len(all_negative_instances)
        print("\tlen(All positive_instances) :", str(all_positive_instances))

        valid_instances = {}
        valid_negative_instances = {}
        for label in target_labels:
            valid_instances[label] = input_data['valid_labels'].str.contains(label).sum()
            valid_negative_instances[label] = valid_instances[label] - positive_instances[label]
        print("\tvalid_instances per label :")
        print(valid_instances)
        print("\tValid_negative_instances per label :")
        print(valid_negative_instances)

        # Condition to meet
        # For each label, valid negative instances should count at most balance_n times the positive ones
        condition_met = False

        initial_negative = len(all_negative_instances)
        counter = 0
        while not condition_met and not all_negative_instances.empty:
        # print progress every 100 steps
            counter = counter + 1
            if counter % 10000 == 0:
                print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), '\t',
                      str(round(counter / initial_negative, 3) * 100), " % of " , str(initial_negative),
                      " negative instances considered. ",
                      str(round( 1.0 - (len(all_negative_instances) / initial_negative), 3)),
                      " % negative reduction so far.")
        # Run until the condition is met, or no more unecesary instances are available to remove for meetting it
            random_index = all_negative_instances.sample(random_state=seed).index
            # Each instance is randomly selected and checked only once
            random_negative_instance = all_negative_instances.loc[random_index]
            # Check if necessary for at least on label
            necessary = False
            valid_labels = random_negative_instance['valid_labels'].values[0].split(" ") # All valid labels for this instance
            # keep only valid labels of focus. Valid labels for which we don't need to learn a model are not considered.
            valid_labels = set.intersection(set(valid_labels), set(target_labels))
            # print('valid_labels ', valid_labels)

            # valid_labels = random_negative_instance['valid_labels'].split(" ")
            for valid_label in valid_labels:
                if valid_negative_instances[valid_label] <= positive_instances[valid_label] * balance_n:
                # This instance is necessary for this label. Do nothing.
                    necessary = True
            if not necessary:
            # This instance is not necessary for any label, it can be removed from the training dataset
                # remove a random negative
                remove_pmids.append(random_negative_instance['pmid'].values[0])
                # Update valid negative values to reflect new situation after the removal
                for valid_label in valid_labels:
                    valid_negative_instances[valid_label] = valid_negative_instances[valid_label] - 1
            # Check all labels to update the condition
            condition_met = True
            for label in target_labels:
                if valid_negative_instances[label] > positive_instances[label] * balance_n:
                    condition_met = False
            #   remove this instance from the dataframe as each instance is only checked once
            all_negative_instances.drop(random_index, inplace = True)

        print('\tremove_pmids (final) :', len(remove_pmids))

    data = input_data[~input_data['pmid'].isin(remove_pmids)]
    stats_on_instace_validity(data, target_labels)
    return data

def create_dataloader(dataset, batch_size, suffle = False):
    '''
    Create a torch DataLoader for the given dataset
    :param dataset:
    :param batch_size:
    :param suffle:
    :return:
    '''
    if suffle:
        # We'll take training samples in random order.
        sampler = RandomSampler(dataset)
    else:
        # For validation the order doesn't matter, so we'll just read them sequentially.
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,  # The training samples.
        sampler= sampler,  # Select batches accordingly
        batch_size=batch_size  # Trains with this batch size.
    )
    return dataloader

def fine_tune(settings, target_labels, train_dataloader, device, validation_dataloader, pos_weight = None, class_freq = None, train_num = None, loss_func_name = None):
    '''
    Fine Tune a model.
    As long as our input data are sentences along with their corresponding labels, this task is similar to sentence classification task. For this reason, we will use the BertForSequenceClassification model.
    :param settings:
    :param target_labels:
    :param train_dataloader:
    :param device:
    :param validation_dataloader:
    :param pos_weight:
    :param class_freq:
    :param train_num:
    :param loss_func_name:
    :return:
    '''

    if not settings["pos_weight"]:
        pos_weight = None
    elif pos_weight is not None:
        pos_weight = pos_weight.to(device)

    modelName = settings["modelName"]
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    epochs = int(settings["epochs"])
    learning_rate = float(settings["learning_rate"])

    model = BertForMultiLabelSequenceClassification.from_pretrained(modelName, num_labels=len(target_labels),
                                                                    output_attentions=True)
    # Tell pytorch to run this model on the GPU.
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Now the training loop

    # Store our loss and accuracy for plotting
    train_loss_set = []
    torch.cuda.empty_cache()

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.

    # Measure the total training time for the whole run.
    start = time.time()
    best_model = None
    previous_model = None # The model trained one epoch prior to the currebt one.
    previous_to_best_model = None # The model trained one epoch prior to the best one.
    best_model_vaL_loss = None
    best_model_epoch = None
    early_stopping = False

    # For each epoch...
    for epoch_i in range(0, epochs):
        if not early_stopping:
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print('\t======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

            # Set our model to training mode (as opposed to evaluation mode)
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()
            wandb.watch(model)

            # Reset the total loss for this epoch.
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Train the data for one epoch
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 500 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = round((time.time() - t0)/60, 2)

                    # Report progress.
                    print('\t  Batch {:>5,}  of  {:>5,}.    Elapsed: {:} m.'.format(step, len(train_dataloader), elapsed))

                # Add batch to GPU
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Clear out the gradients (by default they accumulate)
                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                optimizer.zero_grad()
                # Forward pass
                # Perform a forward pass (evaluate the model on this training batch).
                # In PyTorch, calling `model` will in turn call the model's `forward`
                # function and pass down the arguments. The `forward` function is
                # documented here:
                # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
                # The results are returned in a results object, documented here:
                # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
                # Specifically, we'll get the loss (because we provided labels) and the
                # "logits"--the model outputs prior to activation.
                # Check that all tesors are in the same device:
                # print("b_input_ids.is_cuda", b_input_ids.is_cuda)
                # print("b_input_mask.is_cuda", b_input_mask.is_cuda)
                # print("b_labels.is_cuda", b_labels.is_cuda)
                # print("pos_weight.is_cuda", pos_weight.is_cuda)
                loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, pos_weight = pos_weight, class_freq = class_freq, train_num = train_num, loss_func_name = loss_func_name, device=device)

                train_loss_set.append(loss[0].item())
                # Backward pass
                # Perform a backward pass to calculate the gradients.
                loss[0].backward()
                # Update parameters and take a step using the computed gradient
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update tracking variables
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                tr_loss += loss[0].item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                if step % 10000 == 0:
                    t = time.time()
                    print("\tTrain loss: {}".format(round(tr_loss / nb_tr_steps,2)))
                    print("\tTime: {}".format(round((t - start)/60, 2)))
                    wandb.log({"intermediate tr_loss per instance": round(tr_loss / nb_tr_examples, 2), 'epoch': (epoch_i+ 1)})
            print("\tEpoch train loss : {}".format(round(tr_loss / nb_tr_steps,2)))

            wandb.log({"total tr_loss per instance": round(tr_loss / len(train_dataloader),2), 'epoch': (epoch_i+ 1)})

            # Validation
            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels = b_labels)
                    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels = b_labels, pos_weight = pos_weight, class_freq = class_freq, train_num = train_num, loss_func_name = loss_func_name, device=device)

                # Update tracking variables
                eval_loss += loss[0].item()
                nb_eval_examples += b_input_ids.size(0)
                nb_eval_steps += 1
                if step % 1000 == 0:
                    t = time.time()
                    print("\tValidation loss: {}".format(round(eval_loss / nb_eval_steps,2)))
                    print("\tTime: {}".format(round((t - start)/60, 2)))

            curent_val_loss = eval_loss / len(validation_dataloader)
            print("\tEpoch validation loss: {}".format(round(curent_val_loss,7)))
            wandb.log({"val_loss per instance": round(curent_val_loss,7), 'epoch': (epoch_i + 1)} )
            if best_model is None:
                best_model = copy.deepcopy(model)
                # If the best models is the one train on the first epoch, use this as previous as well.
                previous_to_best_model = copy.deepcopy(model)
                best_model_vaL_loss = curent_val_loss
                best_model_epoch = epoch_i + 1
            elif best_model_vaL_loss > curent_val_loss:
                print("\tModel of epoch ", best_model_epoch, " replaced by ", epoch_i + 1, " as best (", best_model_vaL_loss, " > ",curent_val_loss,")" )
                best_model = copy.deepcopy(model)
                previous_to_best_model = copy.deepcopy(previous_model)
                best_model_vaL_loss = curent_val_loss
                best_model_epoch = epoch_i + 1
            else:
                # this epoch is worse than the previous, ignore further epochs
                early_stopping = True

            previous_model = copy.deepcopy(model)

    end = time.time()
    t = end - start
    print("\tElapsed time: ", round(t/60, 7), "m")

    wandb.log({"best_model_epoch": best_model_epoch})

    models ={}

    if settings["epoch_to_use"] == 'both':
        models['best'] = best_model
        models['prev'] = previous_to_best_model
    elif settings["epoch_to_use"] == 'more':
        models['best'] = best_model
        models['prev'] = previous_to_best_model
        models['current'] = model
    elif settings["epoch_to_use"] == 'previous':
        models['prev'] = previous_to_best_model
    elif settings["epoch_to_use"] == 'best':
        models['best'] = best_model

    return models

def print_token_stats(train_sentences, tokenizer):
    # find the maximum sentence length
    max_len = 0
    total_len = 0

    # For every sentence...
    for sent in train_sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
        total_len += len(input_ids)

    print('\tMax sentence length: ', max_len)
    print('\tAvg sentence length: ', total_len / len(train_sentences))

def print_tokenization_example(train_data,tokenizer):
    # a test sentece
    print('\tThis is an example of tokenization: ')
    t = train_data['text'].iloc[0]
    # Print the original sentence.
    print('\tOriginal: ', t)
    # Print the sentence split into tokens.
    print('\tTokenized: ', tokenizer.tokenize(t))
    # Print the sentence mapped to token ids.
    print('\tToken IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t)))

def predict(prediction_dataloader,model,device, prediction_threshold, verbose = True):
    # print('Predicting labels for {:,} test sentences...'.format(len(test_inputs)))
    print('\tPredicting labels for test sentences...')

    # Tracking variables
    predictions = []
    true_labels = []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # result = model(b_input_ids,
            #               token_type_ids=None,
            #               attention_mask=b_input_mask,
            #               return_dict=True)
            logits = model(batch[0], token_type_ids=None, attention_mask=batch[1])[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            # print(logits)
            preds = []
            for l in logits:
                preds.append(l)
            # print(preds)
            sigmoid = torch.nn.Sigmoid()
            preds = sigmoid(torch.tensor(preds))
            # print(preds)
            preds = np.asarray(preds)
            # print(preds)
            # print("-")

            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            preds2 = np.zeros(preds.shape)
            preds2[preds >= prediction_threshold] = 1
            predictions.append(preds2)
            true_labels.append(label_ids)

        if verbose and len(predictions)%10000 == 0:
            ratio = round((len(predictions) / len(prediction_dataloader)) * 100, 2)
            print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                  len(predictions), " sentences processed (", ratio, "%)")


    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    print('\tDONE.')
    return flat_predictions, flat_true_labels

def evaluation_report(flat_predictions, flat_true_labels, modelName, target_labels, dataset = ""):
    # Calculate the MCC
    mcc = f1_score(flat_true_labels, flat_predictions, average='macro')

    print('\n \t Total f1-score of ' + modelName + ' model for ' + str(
        target_labels) + ': %.3f \n' % mcc)

    print(classification_report(flat_true_labels, flat_predictions))
    return classification_report(flat_true_labels, flat_predictions, output_dict=True)

def validity_filtering(test_data, predictions, target_labels):
    if len(target_labels) > 1:
        # Filtering is only meaningful when more than one labels are handled together.
        # Even with more than one labels, filtering will still be unnecessary if all labels have the same PH.
        # However, applying the filtering will just confirm that no change was needed, without affecting the results.
        # Filter "invalid" predictions
        for initial_index, doc in test_data.iterrows():
            # print("initial_index >", initial_index)
            # We need the position_index because the initial_index available in test_data is not from 0 increasing by one
            # In fact in skips some numbers as it comes from the original dataset, prior to filtering.
            # We need this position_index for finding for which article each prediction is.
            position_index = test_data.index.get_loc(initial_index)
            valid_doc_labels = doc['valid_labels'].split(" ")
            for li, l in enumerate(target_labels):
                if l not in valid_doc_labels:
                    if not predictions[position_index][li] == 0:
                        predictions[position_index][li] = 0
    else:
        print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "\t No prediction filtering needed for just one label (All test articles are from its PH anyway).")
    return predictions

def imbalance_counter(train_y, target_labels):
    # Calculate values about the imbalance in the training datasets useful for modifying the loss during training.
    # 1)    pos_weight tensor with weights to increase the impact of positive examples (used by BCEWithLogitsLoss)
    # 2)    class_freq array with the frequency of each label in the training datasets (used by ResampleLoss)
    # 3)    train_num the size of the training dataset (used by ResampleLoss)
    pos_weights = np.ones(len(target_labels))
    class_freq = torch.sum(train_y, 0) # i.e. positive count
    print("class_freq",class_freq)
    # total_count = len(train_y)
    train_num = train_y.shape[0] # i.e. total size of training dataset
    print("train_num",train_num)
    for i in range(len(target_labels)):
        pos_weights[i] = (train_num - class_freq[i])/ class_freq[i]
    print("pos_weights", pos_weights)
    return torch.as_tensor(pos_weights, dtype=torch.float), class_freq, train_num
