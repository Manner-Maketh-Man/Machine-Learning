import torch
torch.cuda.empty_cache()

import io
import os
import torch
import warnings
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

import eval
import collator
import dataset





if __name__ == "__main__":


    set_seed(123)

    # Number of training epochs (authors on fine-tuning Bert recommend between 2 and 4).
    epochs = 4

    # Number of batches - depending on the max sequence length and GPU memory.
    # For 512 sequence length batch of 10 works without cuda memory issues.
    # For small sequence length can try batch of 32 or higher.
    batch_size = 32

    # Pad or truncate text sequences to a specific length
    # if `None` it will use maximum sequence of word piece tokens allowed by model.
    max_length = 60

    # Look for gpu to use. Will use `cpu` by default if no gpu found.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model_name_or_path='skt/kogpt2-base-v2'

    # Dictionary of labels and their id - this will be used to convert.
    # String labels to number ids.
    labels_ids = {'sad': 0, 'angry': 1, 'disgust': 2, 'happiness': 3, 'fear': 4,
     'neutral': 5, 'surprise': 6}
    # How many labels are we using in training.
    # This is used to decide size of classification head.
    n_labels = len(labels_ids)

    import os

    dir_path = "./dataset"
    data_list = []
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            if '.csv' in file:
                file_path = os.path.join(root, file)
                data_list.append(pd.read_csv(file_path,encoding="cp949"))
            if '.xlsx' in file:
                file_path = os.path.join(root, file)
                data_list.append(pd.read_excel(file_path))

    data = data_list[0].iloc[:,1:3]

    for _ in range(1,3):
        data = pd.concat([data,data_list[_].iloc[:,1:3]])
    buf = data_list[3].iloc[:,1:]
    #buf.columns = ['발화문',"a상황"]
    #data = pd.concat([data,buf])
    data.columns = ["context","label"]

    data[data["label"] == "sadness"] = "sad"
    data[data["label"] == "anger"] = "angry"

    train_x, val_x, train_y, val_y = train_test_split(data["context"],data["label"],test_size = 0.2,stratify=data["label"])

    train_data = pd.DataFrame(columns=["context", "label"])
    train_data["context"] = train_x
    train_data["label"] = train_y

    val_data = pd.DataFrame(columns=["context", "label"])
    val_data["context"] = val_x
    val_data["label"] = val_y



    print('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
      bos_token='</s>', eos_token='</s>', unk_token='<unk>',
      pad_token='<pad>', mask_token='<mask>')
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token

    # Get the actual model.
    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`'%device)

    # Create data collator to encode text and labels into numbers.
    gpt2_classificaiton_collator = collator.Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                              labels_encoder=labels_ids,
                                                              max_sequence_len=max_length)


    print('Dealing with Train...')
    # Create pytorch dataset.
    train_dataset = dataset.NewsDataset(data =train_data,
                                   use_tokenizer=tokenizer)
    print('Created `train_dataset` with %d examples!'%len(train_dataset))

    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

    print()

    print('Dealing with Validation...')
    # Create pytorch dataset.
    valid_dataset = dataset.NewsDataset(data = val_data,
                                   use_tokenizer=tokenizer)
    print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
    print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # default is 1e-8.
                      )

    # Total number of training steps is number of batches * number of epochs.
    # `train_dataloader` contains batched data so `len(train_dataloader)` gives
    # us the number of batches.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))

    # Loop through each epoch.
    #print('Epoch')
    for epoch in (range(epochs)):
        print("Epoch : ",epoch+1)
        print('Training on batches...')
        # Perform one full pass over the training set.
        train_labels, train_predict, train_loss = eval.train(model,train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)

      # Get prediction form model on validation data.
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = eval.validation(model,valid_dataloader, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

      # Print loss and accuracy values to see how training evolves.
        print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
        #print("train_loss: {0:.5f} - train_acc: {1:.5f}".format(train_loss, train_acc))
        print()

      # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)

    # Plot loss curves.
    plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

    # Plot accuracy curves.
    plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

