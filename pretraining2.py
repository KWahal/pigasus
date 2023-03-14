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
from transformers.optimization import Adafactor, AdafactorSchedule

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs[0]
        return (loss, outputs) if return_outputs else loss

class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)


def prepare_data(model_name,
                 train_texts, train_labels,
                 val_texts=None, val_labels=None,
                 test_texts=None, test_labels=None):
    """
    Prepare input data for model pre2training
    """
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    prepare_val = False if val_texts is None or val_labels is None else True
    prepare_test = False if test_texts is None or test_labels is None else True

    def tokenize_data(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True)
        decodings = tokenizer(labels, truncation=True, padding=True)
        dataset_tokenized = PegasusDataset(encodings, decodings)
        return dataset_tokenized

    train_dataset = tokenize_data(train_texts, train_labels)
    val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
    test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

    return train_dataset, val_dataset, test_dataset, tokenizer


def prepare_pre2training(model_name, tokenizer, train_dataset, torch_device, 
                         val_dataset=None, freeze_encoder=False, output_dir='./results'):
    """
    Prepare configurations and base model for pretraining round 2
    """
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    if freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    if val_dataset is not None:
        # matters less bc never pass in val_dataset
        training_args = TrainingArguments(
            output_dir=output_dir,  # output directory
            num_train_epochs=10,  # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
            per_device_eval_batch_size=1,  # batch size for evaluation, can increase if memory allows
            save_steps=500,  # number of updates steps before checkpoint saves
            save_total_limit=5,  # limit the total amount of checkpoints and deletes the older checkpoints
            evaluation_strategy='steps',  # evaluation strategy to adopt during training
            eval_steps=100,  # number of update steps before evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=10,
            # gradient_accumulation_steps=15,
            optim="adafactor",
            learning_rate = 1e-4
        )

        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)

        trainer = CustomTrainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            tokenizer=tokenizer,
            optimizers=(optimizer, lr_scheduler),
            eval_dataset=val_dataset,  # evaluation dataset
        )

    else:
        training_args = TrainingArguments(
            output_dir=output_dir,  # output directory
            num_train_epochs=10,  # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
            save_steps=500,  # number of updates steps before checkpoint saves
            save_total_limit=5,  # limit the total amount of checkpoints and deletes the older checkpoints
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=10,
            # label_names=["labels"] => need this? i mean i did do it as labels
            # label_smoothing_factor = 0 DEFAULT
            optim="adafactor",
            learning_rate = 1e-4
        )

        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)

        trainer = CustomTrainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            tokenizer=tokenizer,
            optimizers=(optimizer, lr_scheduler),
        )

    return trainer


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'google/pegasus-large'
    
    model = PegasusForConditionalGeneration(PegasusConfig()).from_pretrained(model_name)
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")

    train_dataset = (load_dataset("json", data_files="./government_documents.json"))['train']
    
    print(len(train_dataset))

    # TODO: make val set and plug into prepare data as well. => DONE
    all_text, all_labels = train_dataset['inputs'], train_dataset['labels']

    train_text = all_text[:256]#[:12261]
    train_labels = all_labels[:256]#[:12261]

    val_text = all_text[256:] #[12261:15764]
    val_labels = all_labels[256:] #[12261:15764]

    train_dataset, val_dataset, _, tokenizer = prepare_data(model_name, train_text, train_labels, val_texts=val_text, val_labels=val_labels, test_texts=None, test_labels=None)

    trainer = prepare_pre2training(model_name, tokenizer, train_dataset, val_dataset=val_dataset, torch_device=device)
    trainer.train()

    trainer.save_model("./MODELS/pretrain_textrank_mdl")
    
    # model = PegasusForConditionalGeneration(PegasusConfig()).from_pretrained("pigasus/MODELS/test10govdoc_mdl")
    # print(model)

