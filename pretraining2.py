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
        input_string = ["Pegasus is mythical. <mask_1> it names the model ."]
        decoder_input_string = "<s> It is pure white ."
        labels_string = "It is pure white . <eos>"

        input_ids = tok(input_string, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = tok(decoder_input_string, add_special_tokens=False, return_tensors="pt").input_ids
        labels = tok(labels_string, add_special_tokens=False, return_tensors="pt").input_ids

        #loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)[0]
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        loss = outputs[0]

        return (loss, outputs) if return_outputs else loss



if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset_bills = load_dataset('billsum', split="ca_test")
    # dataset_bills = dataset_bills.train_test_split(test_size=0.3)  # CHANGE TEST SPLIT??
    # # gsg_masking(dataset)

    '''
    tok = PegasusTokenizer.from_pretrained("google/pegasus-large")
    model = PegasusForConditionalGeneration(PegasusConfig()).from_pretrained("google/pegasus-large") # from pretrained??

    #examples
    input_string = ["Pegasus is <mask_2>. <mask_1> it <mask_2> the model ."]
    decoder_input_string = "</s> It is pure white ."
    labels_string = "It is pure white . </s>"

    input_ids = tok(input_string, add_special_tokens=False, return_tensors="pt").input_ids
    decoder_input_ids = tok(decoder_input_string, add_special_tokens=False, return_tensors="pt").input_ids
    labels = tok(labels_string, add_special_tokens=False, return_tensors="pt").input_ids

    print(input_ids.shape)
    print(decoder_input_ids.shape)
    print(labels.shape)

    loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)[0]
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)

    print(outputs[0])
    '''

    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
    TXT = "My friends are <mask> but they eat too many carbs."

    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
    input_ids = tokenizer([TXT], return_tensors="np")["input_ids"]
    logits = model(input_ids).logits

    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = jax.nn.softmax(logits[0, masked_index], axis=0)
    values, predictions = jax.lax.top_k(probs)

    print(tokenizer.decode(predictions).split())

