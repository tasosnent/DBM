from transformers import BertForSequenceClassification, BertTokenizerFast
import datetime
from sklearn.metrics import classification_report
import copy
import yaml
from datetime import datetime as dt
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW
from torch.nn import CrossEntropyLoss, MarginRankingLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import numpy as np
import random
import time
import pickle
import sys
import torch
import pandas as pd
import os
import json
from shutil import copyfile

# Run C2F experiments on Fine-Grained Semantic Indexing
# This script is based on:
#  - https://github.com/dheeraj7596/C2F

#  From bert_train.py
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def bert_tokenize(tokenizer, df, label_to_index):
    input_ids = []
    attention_masks = []
    # For every sentence...
    sentences = df.text.values
    labels = copy.deepcopy(df.label.values)
    for i, l in enumerate(list(labels)):
        labels[i] = label_to_index[l]
    labels = np.array(labels, dtype='int32')
    for sent in sentences:
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
            max_length=512,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    # print("input_ids ", len(input_ids), " > ", input_ids)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.LongTensor(labels)
    # Print sentence 0, now as a list of IDs.
    return input_ids, attention_masks, labels

def create_data_loaders(dataset, batch_size = 32 ):
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader

def train(train_dataloader, validation_dataloader, device, num_labels):
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epcoh took: {:}".format(training_time), flush=True)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("", flush=True)
        print("Running Validation...", flush=True)

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy), flush=True)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss), flush=True)
        print("  Validation took: {:}".format(validation_time), flush=True)

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("", flush=True)
    print("Training complete!", flush=True)

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)), flush=True)
    return model

def evaluate(model, prediction_dataloader, device, input_ids):
    # Prediction on test set
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)), flush=True)

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs.logits

        # Move logits and labels to CPU
        logits = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    return predictions, true_labels

def test(df_test_original, label_to_index, index_to_label, tokenizer, model, device):
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, df_test_original, label_to_index)
    # Set the batch size.
    batch_size = 32
    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    predictions, true_labels = evaluate(model, prediction_dataloader, device,input_ids )
    preds = []
    for i, pred in enumerate(predictions):
        if i == 0:
            pred_probs = pred
        else:
            pred_probs = np.concatenate((pred_probs, pred))

        preds = preds + list(pred.argmax(axis=-1))
    true = []
    for t in true_labels:
        true = true + list(t)

    for i, t in enumerate(true):
        true[i] = index_to_label[t]
        preds[i] = index_to_label[preds[i]]

    print(classification_report(true, preds), flush=True)
    return true, preds, pred_probs

def get_high_quality_inds(true, preds, pred_probs, label_to_index, num):
    pred_inds = []
    for p in preds:
        pred_inds.append(label_to_index[p])

    pred_label_to_inds = {}
    for i, p in enumerate(pred_inds):
        try:
            pred_label_to_inds[p].append(i)
        except:
            pred_label_to_inds[p] = [i]

    # print("pred_label_to_inds",pred_label_to_inds)

    label_to_probs = {}
    for p in pred_label_to_inds:
        label_to_probs[p] = []
        for ind in pred_label_to_inds[p]:
            label_to_probs[p].append(pred_probs[ind][p])

    min_ct = num
    print("Collecting", min_ct, "samples as high quality")
    final_inds = {}
    for p in label_to_probs:
        probs = label_to_probs[p]
        inds = np.array(probs).argsort()[-min_ct:][::-1]
        final_inds[p] = []
        for i in inds:
            final_inds[p].append(pred_label_to_inds[p][i])

    # print("final_inds",final_inds)

    temp_true = []
    temp_preds = []
    for p in final_inds:
        for ind in final_inds[p]:
            temp_true.append(true[ind])
            temp_preds.append(preds[ind])

    print("Classification Report of High Quality data")
    print(classification_report(temp_true, temp_preds), flush=True)

    # print("final_inds",final_inds)

    return final_inds

def bert_train(data_dir, model_dir, iteration, parent_label, num_high_quality, device):
    tok_path = os.path.join(model_dir, "bert/" + parent_label + "/tokenizer")
    model_path = os.path.join(model_dir, "bert/" + parent_label + "/model")
    os.makedirs(tok_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # with open(os.path.join(data_dir, "num_dic.json")) as f:
    #     num_dic = json.load(f)

    df_train = pickle.load(open(os.path.join(data_dir, "df_gen_" + parent_label + ".pkl"), "rb"))
    df_fine = pickle.load(open(os.path.join(data_dir, "df_fine.pkl"), "rb"))
    # keep only the instances labelled with some of the fine-grained labels
    #     df_test = df_fine[df_fine["valid_labels"].str.contains(parent_label)].reset_index(drop=True)
    df_test = df_fine[df_fine["label"].isin(list(set(df_train.label.values)))].reset_index(drop=True)

    with open(os.path.join(data_dir, "parent_to_child.json")) as f:
        parent_to_child = json.load(f)

    for ch in parent_to_child[parent_label]:
        for i in range(1, iteration + 1):
            temp_child_df = pickle.load(open(os.path.join(data_dir, "exclusive/" + str(i) + "it/" + ch + ".pkl"), "rb"))
            if i == 1:
                child_df = temp_child_df
            else:
                child_df = pd.concat([child_df, temp_child_df])
        child_df["label"] = [ch] * len(child_df)
        df_train = pd.concat([df_train, child_df])

    print(df_train.label.value_counts())

    # Tokenize all of the sentences and map the tokens to their word IDs.
    print('Loading BERT tokenizer...', flush=True)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

    label_set = set(df_train.label.values)
    label_to_index = {}
    index_to_label = {}
    for i, l in enumerate(list(label_set)):
        label_to_index[l] = i
        index_to_label[i] = l

    input_ids, attention_masks, labels = bert_tokenize(tokenizer, df_train, label_to_index)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    train_dataloader, validation_dataloader = create_data_loaders(dataset)

    # Tell pytorch to run this model on the GPU.
    num_labels = len(label_to_index)
    if num_labels == 1:
        num_labels = 2
    model = train(train_dataloader, validation_dataloader, device, num_labels=num_labels)
    true, preds, pred_probs = test(df_test, label_to_index, index_to_label,tokenizer, model, device)
    high_quality_inds = get_high_quality_inds(true, preds, pred_probs, label_to_index, num_high_quality)

    for p in high_quality_inds:
        print("save generated for ", index_to_label[p], " (size: ", len(high_quality_inds[p]),")")
        inds = high_quality_inds[p]
        temp_df = df_test.loc[inds].reset_index(drop=True)
        os.makedirs(os.path.join(data_dir, "exclusive/" + str(iteration + 1) + "it"), exist_ok=True)
        pickle.dump(temp_df, open(
            os.path.join(data_dir, "exclusive/" + str(iteration + 1) + "it/" + index_to_label[p] + ".pkl"), "wb"))
    # Add this check to hadnle cases where some label is not predicted at all by the model.
    # In this case, no high quality data are available for this label, therefore use again the ones from the previous epovh.
    for i in index_to_label:
        if i not in high_quality_inds:
            print("Keep old generated for ", index_to_label[i], " as no predictions are available.")
            os.makedirs(os.path.join(data_dir, "exclusive/" + str(iteration + 1) + "it"), exist_ok=True)
            copyfile(os.path.join(data_dir, "exclusive/" + str(iteration) + "it/" + index_to_label[i] + ".pkl"),
                     os.path.join(data_dir, "exclusive/" + str(iteration + 1) + "it/" + index_to_label[i] + ".pkl"))
    df_test["pred"] = preds
    pickle.dump(df_test, open(os.path.join(data_dir, "preds_" + parent_label + ".pkl"), "wb"))
    torch.save(model, os.path.join(data_dir, "model_" + parent_label + ".pt"))
    with open(os.path.join(data_dir,"dict_" + parent_label + ".json"), 'w') as fp:
        json.dump(label_to_index, fp)

#  From gpt2_ce_hinge.py

def test_generate(model, tokenizer, label_set, pad_token_dict, device):
    model.eval()
    for l in label_set:
        print("Generating sentence for label", l)
        temp_list = ["<|labelpad|>"] * pad_token_dict[l]
        if len(temp_list) > 0:
            label_str = " ".join(l.split("_")) + "".join(temp_list)
        else:
            label_str = " ".join(l.split("_"))
        text = tokenizer.bos_token + label_str + "<|labelsep|>"
        sample_outputs = model.generate(
            input_ids=tokenizer.encode(text, return_tensors='pt').to(device),
            do_sample=True,
            top_k=50,
            max_length=200,
            top_p=0.95,
            num_return_sequences=1
        )
        for i, sample_output in enumerate(sample_outputs):
            print("{}: {}".format(i, tokenizer.decode(sample_output)))

def basic_gpt2_tokenize(tokenizer, sentences, labels, pad_token_dict, max_length=768):
    input_ids = []
    attention_masks = []
    # For every sentence...
    for i, sent in enumerate(sentences):
        label = labels[i]
        temp_list = ["<|labelpad|>"] * pad_token_dict[label]
        if len(temp_list) > 0:
            label_str = " ".join(label.split("_")) + "".join(temp_list)
        else:
            label_str = " ".join(label.split("_"))
        encoded_dict = tokenizer.encode_plus(
            label_str + "<|labelsep|>" + sent,  # Sentence to encode.
            truncation=True,
            max_length=max_length - 1,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        encoded_dict['input_ids'] = torch.tensor(
            [[tokenizer.bos_token_id] + encoded_dict['input_ids'].data.tolist()[0]]
        )
        encoded_dict['attention_mask'] = torch.tensor(
            [[1] + encoded_dict['attention_mask'].data.tolist()[0]]
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

def gpt2_hinge_tokenize(tokenizer, sentences, labels, pad_token_dict, child_to_parent, max_length=768):
    input_ids = []
    attention_masks = []
    # For every sentence...
    num_sentences = len(sentences)

    for i, sent in enumerate(sentences):
        hinge_input_ids = []
        hinge_attn_masks = []
        for label in [labels[i], child_to_parent[labels[i]]]:
            processed_label_str = " ".join(label.split("_"))
            temp_list = ["<|labelpad|>"] * pad_token_dict[label]
            if len(temp_list) > 0:
                label_str = processed_label_str + "".join(temp_list)
            else:
                label_str = processed_label_str
            encoded_dict = tokenizer.encode_plus(
                label_str + "<|labelsep|>" + sent,  # Sentence to encode.
                truncation=True,
                max_length=max_length - 1,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            encoded_dict['input_ids'] = torch.tensor(
                [[tokenizer.bos_token_id] + encoded_dict['input_ids'].data.tolist()[0]]
            )
            encoded_dict['attention_mask'] = torch.tensor(
                [[1] + encoded_dict['attention_mask'].data.tolist()[0]]
            )
            hinge_input_ids.append(encoded_dict['input_ids'])
            hinge_attn_masks.append(encoded_dict['attention_mask'])

        # Add the encoded sentence to the list.
        input_ids.append(torch.cat(hinge_input_ids, dim=0))

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(torch.cat(hinge_attn_masks, dim=0))
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).view(num_sentences, -1, max_length)
    attention_masks = torch.cat(attention_masks, dim=0).view(num_sentences, -1, max_length)
    return input_ids, attention_masks

def compute_doc_prob(logits, b_fine_input_mask, b_fine_labels, doc_start_ind):
    mask = b_fine_input_mask > 0
    maski = mask.unsqueeze(-1).expand_as(logits)
    logits_pad_removed = torch.masked_select(logits, maski).view(-1, logits.size(-1)).unsqueeze(0)
    logits_pad_removed = logits_pad_removed[:, doc_start_ind - 1:-1, :]

    b_fine_labels_pad_removed = torch.masked_select(b_fine_labels, mask).unsqueeze(0)
    b_fine_labels_pad_removed = b_fine_labels_pad_removed[:, doc_start_ind:]
    log_probs = logits_pad_removed.gather(2, b_fine_labels_pad_removed.unsqueeze(dim=-1)).squeeze(dim=-1).squeeze(
        dim=0)
    return log_probs.sum()

def train_gtp2(model, tokenizer, coarse_train_dataloader, coarse_validation_dataloader, fine_train_dataloader,
          fine_validation_dataloader, doc_start_ind, parent_labels, child_labels, pad_token_dict, device):
    def calculate_ce_loss(lm_logits, b_labels, b_input_mask, doc_start_ind):
        loss_fct = CrossEntropyLoss()
        batch_size = lm_logits.shape[0]
        logits_collected = []
        labels_collected = []
        for b in range(batch_size):
            logits_ind = lm_logits[b, :, :]  # seq_len x |V|
            labels_ind = b_labels[b, :]  # seq_len
            mask = b_input_mask[b, :] > 0
            maski = mask.unsqueeze(-1).expand_as(logits_ind)
            # unpad_seq_len x |V|
            logits_pad_removed = torch.masked_select(logits_ind, maski).view(-1, logits_ind.size(-1))
            labels_pad_removed = torch.masked_select(labels_ind, mask)  # unpad_seq_len

            shift_logits = logits_pad_removed[doc_start_ind - 1:-1, :].contiguous()
            shift_labels = labels_pad_removed[doc_start_ind:].contiguous()
            # Flatten the tokens
            logits_collected.append(shift_logits.view(-1, shift_logits.size(-1)))
            labels_collected.append(shift_labels.view(-1))

        logits_collected = torch.cat(logits_collected, dim=0)
        labels_collected = torch.cat(labels_collected, dim=0)
        loss = loss_fct(logits_collected, labels_collected)
        return loss

    def calculate_hinge_loss(fine_log_probs, other_log_probs):
        loss_fct = MarginRankingLoss(margin=1.609)
        length = len(other_log_probs)
        temp_tensor = []
        for i in range(length):
            temp_tensor.append(fine_log_probs)
        temp_tensor = torch.cat(temp_tensor, dim=0)
        other_log_probs = torch.cat(other_log_probs, dim=0)
        y_vec = torch.ones(length).to(device)
        loss = loss_fct(temp_tensor, other_log_probs, y_vec)
        return loss

    def calculate_loss(lm_logits, b_labels, b_input_mask, doc_start_ind, fine_log_probs, other_log_probs,
                       lambda_1=0.01, is_fine=True):
        ce_loss = calculate_ce_loss(lm_logits, b_labels, b_input_mask, doc_start_ind)
        if is_fine:
            hinge_loss = calculate_hinge_loss(fine_log_probs, other_log_probs)
        else:
            hinge_loss = 0
        return ce_loss + lambda_1 * hinge_loss

    optimizer = AdamW(model.parameters(),
                      lr=5e-4,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    sample_every = 100
    warmup_steps = 1e2
    epochs = 5
    total_steps = (len(coarse_train_dataloader) + len(fine_train_dataloader)) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    seed_val = 81
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)
        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(coarse_train_dataloader):
            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(
                    '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(coarse_train_dataloader), elapsed),
                    flush=True)
                model.eval()
                lbl = random.choice(parent_labels)
                temp_list = ["<|labelpad|>"] * pad_token_dict[lbl]
                if len(temp_list) > 0:
                    label_str = " ".join(lbl.split("_")) + "".join(temp_list)
                else:
                    label_str = " ".join(lbl.split("_"))
                text = tokenizer.bos_token + label_str + "<|labelsep|>"
                sample_outputs = model.generate(
                    input_ids=tokenizer.encode(text, return_tensors='pt').to(device),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output)), flush=True)
                model.train()

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = calculate_loss(outputs[1], b_labels, b_input_mask, doc_start_ind, None, None, is_fine=False)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        for step, batch in enumerate(fine_train_dataloader):
            # batch contains -> fine_input_ids mini batch, fine_attention_masks mini batch
            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(fine_train_dataloader), elapsed),
                      flush=True)
                model.eval()
                lbl = random.choice(child_labels)
                temp_list = ["<|labelpad|>"] * pad_token_dict[lbl]
                if len(temp_list) > 0:
                    label_str = " ".join(lbl.split("_")) + "".join(temp_list)
                else:
                    label_str = " ".join(lbl.split("_"))
                text = tokenizer.bos_token + label_str + "<|labelsep|>"
                sample_outputs = model.generate(
                    input_ids=tokenizer.encode(text, return_tensors='pt').to(device),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output)), flush=True)
                model.train()

            b_fine_input_ids_minibatch = batch[0].to(device)
            b_fine_input_mask_minibatch = batch[1].to(device)

            b_size = b_fine_input_ids_minibatch.shape[0]
            assert b_size == 1
            mini_batch_size = b_fine_input_ids_minibatch.shape[1]

            model.zero_grad()

            batch_other_log_probs = []
            prev_mask = None

            for b_ind in range(b_size):
                for mini_batch_ind in range(mini_batch_size):
                    b_fine_input_ids = b_fine_input_ids_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(device)
                    b_fine_labels = b_fine_input_ids_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(device)
                    b_fine_input_mask = b_fine_input_mask_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(device)
                    outputs = model(b_fine_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_fine_input_mask,
                                    labels=b_fine_labels)
                    log_probs = torch.log_softmax(outputs[1], dim=-1)
                    doc_prob = compute_doc_prob(log_probs, b_fine_input_mask, b_fine_labels, doc_start_ind).unsqueeze(0)
                    if mini_batch_ind == 0:
                        batch_fine_log_probs = doc_prob
                        orig_output = outputs
                        orig_labels = b_fine_labels
                        orig_mask = b_fine_input_mask
                    else:
                        batch_other_log_probs.append(doc_prob)
                    if prev_mask is not None:
                        assert torch.all(b_fine_input_mask.eq(prev_mask))
                    prev_mask = b_fine_input_mask

            loss = calculate_loss(orig_output[1], orig_labels, orig_mask, doc_start_ind, batch_fine_log_probs,
                                  batch_other_log_probs, is_fine=True)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        # **********************************

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / (len(coarse_train_dataloader) + len(fine_train_dataloader))

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epcoh took: {:}".format(training_time), flush=True)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("", flush=True)
        print("Running Validation...", flush=True)

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in coarse_validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

            # Accumulate the validation loss.
            loss = calculate_loss(outputs[1], b_labels, b_input_mask, doc_start_ind, None, None, is_fine=False)
            # loss = outputs[0]
            total_eval_loss += loss.item()

        for batch in fine_validation_dataloader:
            # batch contains -> fine_input_ids mini batch, fine_attention_masks mini batch
            b_fine_input_ids_minibatch = batch[0].to(device)
            b_fine_input_mask_minibatch = batch[1].to(device)

            b_size = b_fine_input_ids_minibatch.shape[0]
            assert b_size == 1
            mini_batch_size = b_fine_input_ids_minibatch.shape[1]

            with torch.no_grad():
                batch_other_log_probs = []
                prev_mask = None

                for b_ind in range(b_size):
                    for mini_batch_ind in range(mini_batch_size):
                        b_fine_input_ids = b_fine_input_ids_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(device)
                        b_fine_labels = b_fine_input_ids_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(device)
                        b_fine_input_mask = b_fine_input_mask_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(
                            device)
                        outputs = model(b_fine_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_fine_input_mask,
                                        labels=b_fine_labels)
                        log_probs = torch.log_softmax(outputs[1], dim=-1)
                        doc_prob = compute_doc_prob(log_probs, b_fine_input_mask, b_fine_labels,
                                                    doc_start_ind).unsqueeze(0)
                        if mini_batch_ind == 0:
                            batch_fine_log_probs = doc_prob
                            orig_output = outputs
                            orig_labels = b_fine_labels
                            orig_mask = b_fine_input_mask
                        else:
                            batch_other_log_probs.append(doc_prob)
                        if prev_mask is not None:
                            assert torch.all(b_fine_input_mask.eq(prev_mask))
                        prev_mask = b_fine_input_mask

            loss = calculate_loss(orig_output[1], orig_labels, orig_mask, doc_start_ind, batch_fine_log_probs,
                                  batch_other_log_probs, is_fine=True)
            total_eval_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / (len(coarse_validation_dataloader) + len(fine_validation_dataloader))

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss), flush=True)
        print("  Validation took: {:}".format(validation_time), flush=True)

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("", flush=True)
    print("Training complete!", flush=True)

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)), flush=True)
    return model

def gpt2_ce_hinge(model_dir, parent_to_child, data_dir, iteration, device):
    seed_val = 81
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tok_path = os.path.join(model_dir, "gpt2/coarse_fine/tokenizer")
    model_path = os.path.join(model_dir, "gpt2/coarse_fine/model/")
    model_name = "coarse_fine.pt"

    os.makedirs(tok_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', bos_token='<|startoftext|>', pad_token='<|pad|>',
                                                  additional_special_tokens=['<|labelsep|>', '<|labelpad|>'])

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    child_to_parent = {}
    for p in parent_to_child:
        for ch in parent_to_child[p]:
            child_to_parent[ch] = p

    parent_labels = []
    child_labels = []
    for p in parent_to_child:
        parent_labels.append(p)
        child_labels += parent_to_child[p]

    all_labels = parent_labels + child_labels

    pad_token_dict = {}
    max_num = -float("inf")
    for l in all_labels:
        tokens = tokenizer.tokenize(" ".join(l.split("_")))
        max_num = max(max_num, len(tokens))

    doc_start_ind = 1 + max_num + 1  # this gives the token from which the document starts in the inputids, 1 for the starttoken, max_num for label info, 1 for label_sep

    for l in all_labels:
        tokens = tokenizer.tokenize(" ".join(l.split("_")))
        pad_token_dict[l] = max_num - len(tokens)

    df_weaksup = None
    for p in parent_to_child:
        for ch in parent_to_child[p]:
            temp_df = pickle.load(
                open(os.path.join(data_dir, "exclusive/" + str(iteration) + "it/" + ch + ".pkl"), "rb"))
            temp_df["label"] = [ch] * len(temp_df)
            if df_weaksup is None:
                df_weaksup = temp_df
            else:
                df_weaksup = pd.concat([df_weaksup, temp_df])

    coarse_input_ids, coarse_attention_masks = basic_gpt2_tokenize(tokenizer, df.text.values, df.label.values,
                                                                   pad_token_dict)
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(coarse_input_ids, coarse_attention_masks)

    # Create a 90-10 train-validation split.
    coarse_train_dataloader, coarse_validation_dataloader = create_data_loaders(dataset, batch_size=4)

    fine_input_ids, fine_attention_masks = gpt2_hinge_tokenize(tokenizer, df_weaksup.text.values,
                                                               df_weaksup.label.values, pad_token_dict, child_to_parent)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(fine_input_ids, fine_attention_masks)

    # Create a 90-10 train-validation split.
    fine_train_dataloader, fine_validation_dataloader = create_data_loaders(dataset, batch_size=1)

    model = train_gtp2(model,
                  tokenizer,
                  coarse_train_dataloader,
                  coarse_validation_dataloader,
                  fine_train_dataloader,
                  fine_validation_dataloader,
                  doc_start_ind,
                  parent_labels,
                  child_labels,
                  pad_token_dict,
                  device)
    test_generate(model, tokenizer, all_labels, pad_token_dict, device)

    tokenizer.save_pretrained(tok_path)
    torch.save(model, model_path + model_name)
    pickle.dump(pad_token_dict, open(os.path.join(data_dir, "pad_token_dict.pkl"), "wb"))

#  From generate_data_ce_hinge.py
def generate(l, tokenizer, model, pad_token_dict, num_samples=1000):
    model.eval()
    temp_list = ["<|labelpad|>"] * pad_token_dict[l]
    if len(temp_list) > 0:
        label_str = " ".join(l.split("_")) + " " + " ".join(temp_list)
    else:
        label_str = " ".join(l.split("_"))
    text = label_str + " <|labelsep|> "
    encoded_dict = tokenizer.encode_plus(text, return_tensors='pt')
    ids = torch.tensor([[tokenizer.bos_token_id] + encoded_dict['input_ids'].data.tolist()[0]]).to(device)

    sents = []
    its = num_samples / 250
    if its < 1:
        sample_outputs = model.generate(
            input_ids=ids,
            do_sample=True,
            top_k=50,
            max_length=200,
            top_p=0.95,
            num_return_sequences=num_samples
        )
        for i, sample_output in enumerate(sample_outputs):
            # print("{}: {}".format(i, tokenizer.decode(sample_output)))
            sents.append(tokenizer.decode(sample_output))
    else:
        for it in range(int(its)):
            sample_outputs = model.generate(
                input_ids=ids,
                do_sample=True,
                top_k=50,
                max_length=200,
                top_p=0.95,
                num_return_sequences=250
            )
            for i, sample_output in enumerate(sample_outputs):
                # print("{}: {}".format(i, tokenizer.decode(sample_output)))
                sents.append(tokenizer.decode(sample_output))
    return sents

def post_process(sentences):
    proc_sents = []
    label_sep_token = '<|labelsep|>'
    label_pad_token = '<|labelpad|>'
    pad_token = '<|pad|>'
    bos_token = '<|startoftext|>'
    remove_list = [label_sep_token, label_pad_token, pad_token, bos_token]

    for sent in sentences:
        ind = sent.find(label_sep_token)
        temp_sent = sent[ind + len(label_sep_token):].strip()
        temp_sent = ' '.join([i for i in temp_sent.strip().split() if i not in remove_list])
        proc_sents.append(temp_sent)
    return proc_sents

def generate_data_ce_hinge(model_dir, data_dir, parent_to_child, parent_label, num, device):
    fine_tok_path = os.path.join(model_dir, "gpt2/coarse_fine/tokenizer")
    fine_model_path = os.path.join(model_dir, "gpt2/coarse_fine/model/")

    pad_token_dict = pickle.load(open(os.path.join(data_dir, "pad_token_dict.pkl"), "rb"))

    fine_tokenizer = GPT2TokenizerFast.from_pretrained(fine_tok_path, do_lower_case=True)
    fine_model = torch.load(fine_model_path + "coarse_fine.pt", map_location=device)

    all_sents = []
    all_labels = []
    children = parent_to_child[parent_label]
    for ch in children:
        sentences = generate(ch, fine_tokenizer, fine_model, pad_token_dict, num_samples=num)
        sentences = post_process(sentences)
        labels = [ch] * num
        all_sents += sentences
        all_labels += labels

    df = pd.DataFrame.from_dict({"text": all_sents, "label": all_labels})
    pickle.dump(df, open(os.path.join(data_dir, "df_gen_" + parent_label + ".pkl"), "wb"))

# Train models
if __name__ == '__main__':
    # freeze_support()

    if len(sys.argv) == 2:
        settings_file = sys.argv[1]
        print(dt.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "Run with settings.py file at: " + settings_file)
    else:
        print(dt.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
              "No settings.py file found as argument, try to find the file in the project folder.")
        settings_file = "settings_c2f.yaml"

    # Read the settings
    settings = {}
    # settings = yaml.load(settings_file, Loader=yaml.FullLoader)
    with open(settings_file) as parameters:
        settings = yaml.safe_load(parameters)
    experiment_folder = settings["experiment_folder"]

    device = torch.device('cuda:0')

    parent_child = {}
    with open(experiment_folder + os.path.sep + "parent_to_child.json", "r+") as f:
        parent_child = json.load(f)
    df = pickle.load(open(os.path.join(experiment_folder, "df_coarse.pkl"), "rb"))

    itterations = [1, 2]

    for it in itterations:
        # Train gtp2 generative model
        print("Traing gtp2, Iteration :", it)
        gpt2_ce_hinge(experiment_folder, parent_child, experiment_folder, it, device)

        for parent in parent_child.keys():

            # Generate instances
            print("Generate instances for:", parent)
            print("Iteration :", it)
            generate_data_ce_hinge(experiment_folder, experiment_folder, parent_child, parent, 500, device)

            # Train BERT classifiers
            print("Train bert model for:", parent)
            print("Iteration :", it)
            bert_train(experiment_folder, experiment_folder, it, parent, 500, device)