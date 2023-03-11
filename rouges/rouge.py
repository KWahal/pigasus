from rouge_score import rouge_scorer
from datasets import load_dataset
from tqdm import tqdm
import csv

SEPARATOR = "=================================="
# load datasets

path_to_summs = "./summaries/val_ca_sum_10_100.txt"

billsum_test = load_dataset('billsum', split="ca_test")
target_summaries = billsum_test['summary'][865:959]
bill_names = billsum_test['title'][865:959]

test_summs = "".join(open(path_to_summs, 'r').readlines())
test_summs = test_summs.split(SEPARATOR)

print(test_summs)

print(len(test_summs), len(bill_names), len(target_summaries))

# calculate rouge scores between extractive summary and billsum summary
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
all_scores = []

for i in tqdm(range(len(bill_names))):
    score = scorer.score(test_summs[i], target_summaries[i])
    # parse + save score data type
    all_scores.append({'title': bill_names[i], 'rouge1': score['rouge1'].fmeasure, 'rouge2': score['rouge2'].fmeasure, 'rougeL': score['rougeL'].fmeasure})

    
# write rouge scores to CSV file
field_names = ["title", "rouge1", "rouge2", "rougeL"]

with open('catest_10d_100e.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = field_names)
    writer.writeheader()
    writer.writerows(all_scores)
