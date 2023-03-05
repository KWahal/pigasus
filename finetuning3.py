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

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_batch_sized_chunks(list_of_elements, batch_size):
    """split the dataset into smaller batches that we can process simultaneously
    Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i: i + batch_size]


def calculate_metric_on_test_ds(dataset, metric, model, tokenizer,
                                batch_size=16, device=device,
                                column_text="article",
                                column_summary="highlights"):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
        inputs = tokenizer(article_batch, max_length=1024, truncation=True,
                           padding="max_length", return_tensors="pt")

        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                   attention_mask=inputs["attention_mask"].to(device),
                                   length_penalty=0.8, num_beams=8, max_length=128)
        ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''

        # Finally, we decode the generated texts,
        # replace the <n> token, and add the decoded texts with the references to the metric.
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)
                             for s in summaries]

        decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]

        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    #  Finally compute and return the ROUGE scores.
    score = metric.compute()
    return score


def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['text'], max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['summary'], max_length=128, truncation=True)

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ckpt = "google/pegasus-large"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

    dataset_bills = load_dataset('billsum', split="ca_test")
    dataset_bills = dataset_bills.train_test_split(test_size=0.3) # CHANGE TEST SPLIT??

    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_metric = load_metric('rouge')
    score = calculate_metric_on_test_ds(dataset_bills['test'], rouge_metric, model_pegasus, tokenizer,
                                        column_text='text', column_summary='summary', batch_size=8)
    rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

    trainer_args = TrainingArguments(
        output_dir='pegasus-samsum', num_train_epochs=1, warmup_steps=500,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        weight_decay=0.01, logging_steps=10,
        evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
        gradient_accumulation_steps=16
    )
    dataset_bills_pt = dataset_bills.map(convert_examples_to_features, batched=True)
    trainer = Trainer(model=model_pegasus, args=trainer_args,
                      tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                      train_dataset=dataset_bills_pt["train"],
                      eval_dataset=dataset_bills_pt["validation"])

    trainer.train()

    score = calculate_metric_on_test_ds(
        dataset_bills['test'], rouge_metric, trainer.model, tokenizer, batch_size=2, column_text='dialogue',
        column_summary='summary'
    )
    rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
    pd.DataFrame(rouge_dict, index=[f'pegasus'])