from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers.optimization import Adafactor, AdafactorSchedule

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
    Prepare input data for model fine-tuning
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


def prepare_fine_tuning(model_name, tokenizer, train_dataset, torch_device, val_dataset=None, freeze_encoder=False,
                        output_dir='./results'):
    """
    Prepare configurations and base model for fine-tuning
    """
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model)

    if freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    if val_dataset is not None:
        training_args = TrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=80,  # total number of training epochs
            per_device_train_batch_size=2,  # batch size per device during training, can increase if memory allows
            per_device_eval_batch_size=2,  # batch size for evaluation, can increase if memory allows
            save_steps=200,  # number of updates steps before checkpoint saves
            save_total_limit=5,  # limit the total amount of checkpoints and deletes the older checkpoints
            evaluation_strategy='steps',  # evaluation strategy to adopt during training
            eval_steps=200,  # number of update steps before evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=10,
            optim="adafactor",
            learning_rate=1e-3
            #gradient_accumulation_steps=4
            # 15 to 4
        )
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            optimizers=(optimizer, lr_scheduler)
        )

    else:
        training_args = TrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=70,  # total number of training epochs
            #prev 1
            per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
            save_steps=500,  # number of updates steps before checkpoint saves
            save_total_limit=5,  # limit the total amount of checkpoints and deletes the older checkpoints
            warmup_steps=400,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=10,
            optim="adafactor",
            learning_rate=1e-3
            #gradient_accumulation_steps=15
        )
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
        lr_scheduler = AdafactorSchedule(optimizer)
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            optimizers=(optimizer, lr_scheduler),
        )

    return trainer

def print_val_summaries(dataset_text, dataset_summary, model, tokenizer, device, file_name,
                                batch_size=16):
    text_file = open(file_name, "w") # MAKE SURE TO CHANGE
    articles = list(dataset_text)
    targets = list(dataset_summary)
    for article, target in tqdm(zip(articles, targets), total=len(articles)):
        inputs = tokenizer(article_batch, max_length=1024, truncation=True,
                           padding="max_length", return_tensors="pt")
        translated = model.generate(**inputs)
        decoded_summaries = tokenizer.batch_decode(translated, skip_special_tokens=True)
        text_file.write(" ".join(decoded_summaries))
        text_file.write("==================================")
    text_file.close()


if __name__ == '__main__':
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from datasets import load_dataset


    dataset = load_dataset('billsum', split="ca_test")
    train_texts, train_labels = dataset['text'][:10], dataset['summary'][:10]
    val_texts, val_labels = dataset['text'][865:910], dataset['summary'][865:910]  # 1051

    # use Pegasus Large model as base for fine-tuning
    model_name = 'google/pegasus-large'
    train_dataset, val_dataset, _, tokenizer = prepare_data(model_name, train_texts, train_labels, val_texts=val_texts, val_labels=val_labels)
    trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=val_dataset, freeze_encoder=True, torch_device=torch_device)
    train_results = trainer.train()
    trainer.save_model("./saved_model")
    #model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device) #DEL - for testing
    #tokenizer = PegasusTokenizer.from_pretrained(model_name) #DEL - for testing
    print_val_summaries( val_texts2, val_labels2, model=model3, tokenizer=tokenizer3, batch_size=1, file_name="Test-P-large0-again.txt",
        device=torch_device  # model=trainer.model)
