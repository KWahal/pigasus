from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
from transformers import PegasusConfig, PegasusModel
import torch
from datasets import load_dataset
from tqdm import tqdm
from ray import tune
from ray.air import session


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

def preprocessing(train_text, train_summs, test_text, test_summs, valid_text, valid_summs):
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    
    def tokenize_data(text, summaries):
        tokenized_text = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
        tokenized_summaries = tokenizer(summaries, truncation=True, padding="longest", return_tensors="pt")
        dataset_tokenized = PegasusDataset(tokenized_text, tokenized_summaries)
        return dataset_tokenized
    
    train_dataset = tokenize_data(train_text, train_summs)
    test_dataset = tokenize_data(test_text, test_summs)
    val_dataset = tokenize_data(valid_text, valid_summs)
    
    return train_dataset, val_dataset, test_dataset, tokenizer
'''
def trainable(config: dict):
    intermediate_score = 0
    for x in range(20):
        intermediate_score = objective(x, config["a"], config["b"])
        session.report({"score": intermediate_score})
'''
def finetune(train_data, val_data, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    # hyperparams: FIND max input tokens and max target tokens
    finetune_args = {
        'learning_rate': tune.grid_search([]),
        'label_smoothing_factor': 0.1,
        'num_train_epochs': tune.grid_search([90000, 100000, 115000, 120000]), #Number of steps???
        'per_gpu_train_batch_size': tune.grid_search([256, 512]) 
    }
    #trainer = Trainer(model, finetune_args, train_data, val_data, tokenizer)
    tuner = tune.Tuner(
        trainable = model,
        param_space = finetune_args
    )
    results = tuner.fit()



if __name__ == "__main__":
    model_name = "google/pegasus-large"
    dataset = load_dataset('billsum', split="ca_test")

    doc_data = dataset["text"] 
    val = doc_data[:865] + doc_data[1051:] # combine train + val
    test = doc_data[865:1051]

    sum_data = dataset["summary"]
    val_label = sum_data[:865] + sum_data[1051:] # combine train + val
    test_label = sum_data[865:1051]

    train_dataset, val_dataset, test_dataset, tokenizer = preprocessing(train, train_label, test, test_label, val, val_label)
    finetune(train_dataset, val_dataset, tokenizer)