import numpy as np
import pandas as pd
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import gluonnlp as nlp
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import BertModel, BertTokenizer
from transformers import logging
import multiprocessing
logging.set_verbosity_error() # ingnore transformer warning 

# kogpt model
class Kogpt:
    # check model path
    MODEL_PATH = "./jm_model.pt"
    TOKENIZER_NAME = "skt/kogpt2-base-v2"

    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_NAME)
        self.device = device
        self.model = torch.load(self.MODEL_PATH, map_location=device)

    def predict(self, sent):
        # set my model max len
        MAX_LEN = 60

        self.model.eval()
        self.tokenizer.padding_side = "right"
        
        # Define PAD Token = EOS Token = 50256
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokenized_sent = self.tokenizer(
            sent,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN
        )

        tokenized_sent.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized_sent["input_ids"],
                attention_mask=tokenized_sent["attention_mask"],
            )

        logits = outputs[0]
        logits = logits.detach().cpu()
        result = logits.argmax(-1)

        return np.array(result)[0]
    
# koelectra model
class Koelectra:
    # check model path
    MODEL_PATH = "C:/vscode project/캡스톤/test/koelectra_imbalanced.pt"
    TOKENIZER_NAME = "monologg/koelectra-base-v3-discriminator"

    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_NAME)
        self.device = device
        self.model = torch.load(self.MODEL_PATH, map_location=device)
    
    def predict(self, sent):
        # set my model max len 
        MAX_LEN = 64
        
        self.model.eval()

        tokenized_sent = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=MAX_LEN
        )

        tokenized_sent.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                    input_ids = tokenized_sent["input_ids"],
                    attention_mask=tokenized_sent["attention_mask"],
                    token_type_ids=tokenized_sent["token_type_ids"]
            )

        logits = outputs[0]
        logits = logits.detach().cpu()
        result = logits.argmax(-1)
        
        return np.array(result)[0]

class Kobert:
    MODEL_PATH = "KoBERT_ver2.pt"
    def __init__(self, device):
        self.tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
        self.model = torch.load(self.MODEL_PATH, map_location=device)
        self.model.eval()
        self.device = device

    def predict(self, sentence):
        max_len = 64
        self.model.eval()
        
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze().to(self.device)
        attention_mask = encoding["attention_mask"].squeeze().to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))[0]
            probabilities = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
            predicted_label = np.argmax(probabilities)
        
        return predicted_label
    
def get_sentence(file):
    target = file["images"][0][0]['fields'][0]['inferText']
    start = False
    sentences = []

    for d in file["images"][0][0]['fields'][2:]:
        x = d['boundingPoly']['vertices'][0]['x']
        content = d['inferText']

        if start == True:
            if x < 900:
                temp += content + " "

            else:
                sentences.append(temp)
                start = False

        else:
            if x < 900 and content == target:
                start = True
                temp = ""

    func = lambda s:s[:list(re.finditer(f"오전|오후",s))[-1].span()[0]-1]
    sentences = [func(i) for i in sentences]
    return sentences


if __name__ == "__main__":
    
    multiprocessing.set_start_method("spawn", True)
    
    # Read JSON File
    JSON_FILE_PATH = "C:/vscode project/캡스톤/test/test_json_file_2.json"
    f = pd.read_json(JSON_FILE_PATH, lines=True)
    sentences = get_sentence(file=f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load KOGPT 
    kogpt = Kogpt()


    # Load KOELECTRA 
    koelectra = Koelectra(device)


    # Load KOBERT
    kobert = Kobert(device)

    # Make Prediction
    label_decoder =  {0:"sadness",
                    1:"fear",
                    2:"disgust",
                    3:"neutral",
                    4:"happiness",
                    5:"angry",
                    6:"surprise"}

    # if predict only last sentence
    sentence = sentences[-1]

    # KOGPT Prediction
    kogpt_prediction = kogpt.predict(sentence)

    # KOELECTRA Prediction
    koelectra_prediction = koelectra.predict(sentence)

    # KOBERT Prediction
    kobert_prediction = kobert.predict(sentence)


    print(f"kogpt_prediction : {kogpt_prediction} : {label_decoder[kogpt_prediction]}")
    print(f"koelectra_prediction : {koelectra_prediction} : {label_decoder[koelectra_prediction]}")
    print(f"kobert_prediction : {kobert_prediction} : {label_decoder[kobert_prediction]}")

    # ensemble
    prediction_list = [kogpt_prediction, koelectra_prediction, kobert_prediction]
    ensemble_result = max(prediction_list, key=prediction_list.count)

    print(f"ensemble_result : {ensemble_result} : {label_decoder[ensemble_result]}")