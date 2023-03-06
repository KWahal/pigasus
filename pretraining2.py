from transformers import pipeline, set_seed
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import PegasusConfig, PegasusModel
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer

class CustomTrainer(Trainer):
    def new_compute_loss (self, model, inputs, return_outputs=False):
        tok = PegasusTokenizer.from_pretrained("pegasus-large")
        model = PegasusForConditionalGeneration(PegasusConfig()).from_pretrained("pegasus-large") # from pretrained??

        #examples
        input_string = ["Pegasus is <mask_2> . <mask_1> it <mask_2> the model ."]
        decoder_input_string = "<s> It is pure white ."
        labels_string = "It is pure white . <eos>"

        input_ids = tok(input_string, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = tok(decoder_input_string, add_special_tokens=False, return_tensors="pt").input_ids
        labels = tok(labels_string, add_special_tokens=False, return_tensors="pt").input_ids

        #loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)[0]
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        loss = outputs[0]

        return (loss, outputs) if return_outputs else loss

def gsg_masking():



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_bills = load_dataset('billsum', split="ca_test")
    dataset_bills = dataset_bills.train_test_split(test_size=0.3)  # CHANGE TEST SPLIT??
    gsg_masking(dataset)


