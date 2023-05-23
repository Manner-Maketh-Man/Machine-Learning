import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import logging
logging.set_verbosity_error() # ingnore transformer warning
from sklearn.metrics import classification_report
import torch

import eval
import collator
import dataset


if __name__ == "__main__":
    #multiprocessing.set_start_method("spawn", True)

    # Read JSON File
    max_length = 60
    batch_size = 60
    test_data = pd.read_csv("./test_data.csv")


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    labels_ids = {'sad': 0, 'fear': 1, 'disgust': 2, 'neutral': 3, 'happiness': 4,
                  'angry': 5, 'surprise': 6}
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    model = torch.load("jm_model.pt", map_location=device)
    test_dataset = dataset.NewsDataset(data=test_data,
                                        use_tokenizer=tokenizer)
    print('Created `valid_dataset` with %d examples!' % len(test_dataset))

    gpt2_classificaiton_collator = collator.Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                                       labels_encoder=labels_ids,
                                                                       max_sequence_len=max_length)
    # Move pytorch dataset into dataloader.
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=gpt2_classificaiton_collator)

    test_labels, test_predict, test_loss = eval.validation(model, test_dataloader, device)

    label_decoder = {0: "sadness", 1: "fear", 2: "disgust", 3: "neutral", 4: "happiness", 5: "angry", 6: "surprise"}

    print(classification_report(test_labels,test_predict, target_names=label_decoder.values()))
    # if predict only last sentence






