# -*- coding: utf-8 -*-

# Load Essential Libraries
import os
import re
import tqdm
import time
import nltk
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from _bert_class import BertClassifier
import torch.nn.functional as torch_nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, roc_curve, auc, ConfusionMatrixDisplay, classification_report

# Uncomment to avoid "stopwords"
nltk.download("stopwords")
from nltk.corpus import stopwords


def train_set(train_data):
    global df_train
    path = os.getcwd()
    if train_data == "rashkin":
        _path = path.replace('src\BERT_model', 'data\data_used_for_bert\RASHKIN_train.txt')
        df_rashkin_train = pd.read_fwf(_path, header=None)
        df_rashkin_train.drop(df_rashkin_train.iloc[:, 2:], inplace=True, axis=1)
        df_rashkin_train = df_rashkin_train.add_prefix('col_')
        # Remove double quotes from start and end
        for i in range(len(df_rashkin_train)):
            df_rashkin_train["col_1"][i] = df_rashkin_train["col_1"][i].strip('"')

        df_rashkin_train["col_0"].replace(1, 0, inplace=True)
        df_rashkin_train["col_0"].replace(2, 0, inplace=True)
        df_rashkin_train["col_0"].replace(3, 0, inplace=True)
        df_rashkin_train["col_0"].replace(4, 1, inplace=True)

        # Set the fraction value in a way to randomly sample and keep only 4000 rows of each False and True news
        d1 = df_rashkin_train[df_rashkin_train['col_0'] == 0].sample(frac=0.10295, random_state=100).reset_index(
            drop=True)
        d2 = df_rashkin_train[df_rashkin_train['col_0'] == 1].sample(frac=0.4002, random_state=100).reset_index(
            drop=True)
        # Shuffle dataset
        df_rashkin_train_updated = pd.concat([d1, d2]).sample(frac=1, random_state=100).reset_index(drop=True)
        df_rashkin_train_updated.columns = ['label', 'text']

        # Final
        df_train = df_rashkin_train_updated
        labels = df_train["label"]
        text = df_train["text"].tolist()

    elif train_data == "mix":
        _path = path.replace('src\BERT_model', 'data\data_used_for_bert\\trainRaw_mix')
        __path = path.replace('src\BERT_model', 'data\data_used_for_bert\\trainlRaw_mix')
        texts_train = np.load(_path, allow_pickle=True)
        labels_train = np.load(__path, allow_pickle=True)

        train_df = pd.DataFrame(
            {'label': labels_train.astype(str),
             'text': texts_train},
        )
        df = train_df.sample(frac=1, random_state=100).reset_index(drop=True)

        # Final
        df_train = df
        labels = df_train["label"]
        text = df_train["text"].tolist()

    elif train_data == "small":
        _path = path.replace('src\BERT_model', 'data\data_used_for_bert\\trainRaw_small')
        __path = path.replace('src\BERT_model', 'data\data_used_for_bert\\trainlRaw_small')
        texts_train = np.load(_path, allow_pickle=True)
        labels_train = np.load(__path, allow_pickle=True)

        train_df = pd.DataFrame(
            {'label': labels_train,
             'text': texts_train},
        )
        df = train_df.sample(frac=1, random_state=100).reset_index(drop=True)

        # Final
        df_train = df
        labels = df_train["label"]
        text = df_train["text"].tolist()

    return df_train


def test_set(test_data):
    global df_test
    path = os.getcwd()
    if test_data == "top":
        _path = path.replace('src\BERT_model', 'data\data_used_for_bert\\buzzfeed-top.csv')
        df_test_top = pd.read_csv(_path)
        # Buzzfeed-top dataset's labels are all false
        df_test_top["label"] = 0

        # Final
        df_test = df_test_top
        labels_test = df_test["label"]
        df_test["text"] = df_test["original_article_text_phase2"]
        text_test = df_test["original_article_text_phase2"].tolist()

    elif test_data == "snope_f":
        _path = path.replace('src\BERT_model', 'data\data_used_for_bert\\snopes.csv')
        df_test_snope = pd.read_csv(_path)

        # Snope_checked dataset mixture labels are removed
        df_test_snope = df_test_snope[df_test_snope["label"] != "mixture"]
        df_test_snope["label"].replace("mfalse", 0, inplace=True)
        df_test_snope["label"].replace("ffalse", 0, inplace=True)
        df_test_snope["label"].replace("mtrue", 1, inplace=True)
        df_test_snope["label"].replace("ftrue", 1, inplace=True)
        # Separate fake news
        df_test_snope_fake = df_test_snope[df_test_snope["label"] == 0].reset_index(drop=True)

        # Final
        df_test = df_test_snope_fake
        labels_test = df_test["label"]
        text_test = df_test["text"].tolist()

    elif test_data == "snope":
        _path = path.replace('src\BERT_model', 'data\data_used_for_bert\\snopes.csv')
        df_test_snope = pd.read_csv(_path)

        # Snope_checked dataset mixture labels are removed
        df_test_snope = df_test_snope[df_test_snope["label"] != "mixture"]
        df_test_snope["label"].replace("mfalse", 0, inplace=True)
        df_test_snope["label"].replace("ffalse", 0, inplace=True)
        df_test_snope["label"].replace("mtrue", 1, inplace=True)
        df_test_snope["label"].replace("ftrue", 1, inplace=True)

        # Final
        df_test = df_test_snope
        labels_test = df_test["label"]
        text_test = df_test["text"].tolist()

    elif test_data == "celeb_f":
        _path = path.replace('src\BERT_model', 'data\data_used_for_bert\\celeb.csv')
        df_test_celeb = pd.read_csv(_path)

        # Celeb dataset
        df_test_celeb["label"].drop(columns=["Unnamed: 0"], inplace=True)
        df_test_celeb["label"].replace("legit", 1, inplace=True)
        df_test_celeb["label"].replace("fake", 0, inplace=True)

        # Separate fake news
        df_test_celeb_fake = df_test_celeb[df_test_celeb["label"] == 0].reset_index(drop=True)

        # Final
        df_test = df_test_celeb_fake
        labels_test = df_test["label"]
        text_test = df_test["text"].tolist()

    elif test_data == "celeb":
        _path = path.replace('src\BERT_model', 'data\data_used_for_bert\\celeb.csv')
        df_test_celeb = pd.read_csv(_path)

        # Celeb dataset
        df_test_celeb["label"].drop(columns=["Unnamed: 0"], inplace=True)
        df_test_celeb["label"].replace("legit", 1, inplace=True)
        df_test_celeb["label"].replace("fake", 0, inplace=True)

        # Final
        df_test = df_test_celeb
        labels_test = df_test["label"]
        text_test = df_test["text"].tolist()

    return df_test


def _train_test_split(train_data, test_data):
    # Randomly split the entire training data into two sets: a train set with 80% of the data and a
    # validation set with 20% of the data.
    _train_data = train_set(train_data)
    X = _train_data.text.values
    y = _train_data.label.values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2022)

    # Test data
    _test_data = test_set(test_data)
    X_test = _test_data.text.values
    y_test = _test_data.label.values

    return X_train, X_val, y_train, y_val, X_test, y_test


def text_preprocessing(s):
    """
    - Remove trailing whitespace
    - Replace '&amp;' with '&'
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Replace '&amp;' with '&'
    s = re.sub(r'&amp;', '&', s)
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\".()!?\\/,])', r' \1 ', s)
    s = re.sub(r'[^\w\s?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([;:|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def evaluate_roc(probs, y_true):
    """
    To evaluate the performance of our model, we will calculate the accuracy rate and the AUC score of our model on the
    validation set.

    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]  # this will give you the probability of getting the output as 1
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')

    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision == 0:
        print("Zero precision!")
        _f1_score = f1_score(y_true, y_pred, zero_division=1)
    else:
        _f1_score = f1_score(y_true, y_pred, average='weighted')

    print(classification_report(y_true, y_pred, labels=[0, 1]))
    print("**********************")
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()
    print("**********************")

    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def preprocessing_for_bert(data, tokenizer=None, MAX_LEN=384):
    """
    Perform required preprocessing steps for pretrained BERT (tokenizing a set of texts)
    @param data: Array of texts to be processed.
    @param tokenizer: BERT tokenizer
    @param MAX_LEN: input sequence length of BERT
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which tokens should be attended to by the
    model.
    """

    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,  # Max length to truncate/pad
            pad_to_max_length=True,  # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True  # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def initialize_model(epochs=6, lr=1e-5, train_dataloader=None, device=torch.device("cpu")):
    """
    Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=True)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(
        bert_classifier.parameters(),
        lr=lr,  # Default learning rate 2e-4 was terrible
        eps=1e-8  # Default epsilon value
    )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=10,
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


def set_seed(seed_value=42):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def train(model, train_dataloader, val_dataloader=None, epochs=6, evaluation=False, optimizer=None, scheduler=None,
          device=torch.device("cpu")):
    """
    Train the BertClassifier model.

    Training:
    - Unpack our data from the dataloader and load the data onto the GPU
    - Zero out gradients calculated in the previous pass
    - Perform a forward pass to compute logits and loss
    - Perform a backward pass to compute gradients (`loss.backward()`)
    - Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
    - Update the model's parameters (`optimizer.step()`)
    - Update the learning rate (`scheduler.step()`)

    Evaluation:
    - Unpack our data and load onto the GPU
    - Forward pass
    - Compute loss and accuracy rate over the validation set
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # Specify loss function
        loss_fn = nn.CrossEntropyLoss()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | "
                    f"{time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)

        # =======================================
        #               Evaluation
        # =======================================
        if evaluation:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader, device)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | "
                f"{time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    print("Training complete!")


def evaluate(model, val_dataloader, device=torch.device("cpu")):
    """
    After the completion of each training epoch, measure the model's performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during the test time.
    model.eval()

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def bert_predict(model, test_dataloader, device=torch.device("cpu")):
    """
    Perform a forward pass on the trained BERT model to predict probabilities on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = torch_nn.softmax(all_logits, dim=1).cpu().numpy()

    return probs


def main():
    # Set up Colab GPU for training: Runtime -> Change runtime type -> Hardware accelerator: GPU`
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # ----------------------------------------------------------------------------------------------------- #

    # For fine-tuning BERT, the BERT authors recommend a batch size of 16 or 32.
    batch_size = 8
    max_seq_length = 384
    epochs = 6
    drop_out = 0.25
    lr = 1e-5

    # ----------------------------------------------------------------------------------------------------- #

    # Load the BERT tokenizer using the transformer library of Hugging Face
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # Choose train & test datasets
    train_data = "rashkin"  # choice of ["rashkin", "mix", "small"] datasets
    test_data = "top"  # choice of ["top", "snope", "celeb"] datasets
    X_train, X_val, y_train, y_val, X_test, y_test = _train_test_split(train_data, test_data)

    # Run function `preprocessing_for_bert` on the train set and the validation set
    train_inputs, train_masks = preprocessing_for_bert(X_train, tokenizer=tokenizer, MAX_LEN=max_seq_length)
    val_inputs, val_masks = preprocessing_for_bert(X_val, tokenizer=tokenizer, MAX_LEN=max_seq_length)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    # ----------------------------------------------------------------------------------------------------- #

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # ----------------------------------------------------------------------------------------------------- #

    # Set seed for reproducibility
    set_seed(42)
    bert_classifier, optimizer, scheduler = initialize_model(epochs=epochs, lr=lr, train_dataloader=train_dataloader,
                                                             device=device)
    train(bert_classifier, train_dataloader, val_dataloader, epochs=epochs, evaluation=True, optimizer=optimizer,
          scheduler=scheduler, device=device)

    # Compute predicted probabilities on the test set
    probs = bert_predict(bert_classifier, val_dataloader, device=device)

    # Evaluate the Bert classifier
    evaluate_roc(probs, y_val)

    # ----------------------------------------------------------------------------------------------------- #

    # Train Our Model on the Entire Training Data (both training and validation sets)
    full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
    full_train_sampler = RandomSampler(full_train_data)
    full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=batch_size)

    # Train the Bert Classifier again on the entire training data
    train(bert_classifier, full_train_dataloader, epochs=epochs, optimizer=optimizer, scheduler=scheduler,
          device=device)

    # ----------------------------------------------------------------------------------------------------- #

    # Predictions on test data
    # Run `preprocessing_for_bert` on the test set
    test_inputs, test_masks = preprocessing_for_bert(X_test, tokenizer, MAX_LEN=max_seq_length)

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    # Compute predicted probabilities on the test set
    probs_test = bert_predict(bert_classifier, test_dataloader)

    # Evaluate the Bert classifier
    evaluate_roc(probs_test, y_test)


if __name__ == '__main__':
    main()
