from rouge_score import rouge_scorer
from datasets import load_dataset
from tqdm import tqdm
import csv

SEPARATOR = "=================================="
# load datasets

'''
- catest_10d_100e.csv => uses [865:955] of ca_test
- caval_peglarge.txt => uses [865:1051] of ca_test
- extractivesumms_rouges.csv => extractive summs on federal tests??
- pegasussumms_rouges.csv => pegasus summs on federal test
- caval_100d_24e.csv => finetuned on 100, 25 epochs. summaries on VAL set [865:1051]
'''

path_to_summs = "./summaries/val_ca_sum_100_25_1.txt"

billsum_test = load_dataset('billsum', split="ca_test")
target_summaries = billsum_test['summary'][865:1051]
bill_names = billsum_test['title'][865:1051]
bill_indices = list(range(865, 1051))

test_summs = "".join(open(path_to_summs, 'r').readlines())
test_summs = test_summs.split(SEPARATOR)[:-1]


print(len(test_summs), len(bill_names), len(target_summaries))

# calculate rouge scores between extractive summary and billsum summary
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
all_scores = []

for i in tqdm(range(len(bill_names))):
    score = scorer.score(test_summs[i], target_summaries[i])
    # parse + save score data type
    all_scores.append({'index': bill_indices[i], 'title': bill_names[i], 'rouge1': score['rouge1'].fmeasure, 'rouge2': score['rouge2'].fmeasure, 'rougeL': score['rougeL'].fmeasure})

    
# write rouge scores to CSV file
field_names = ["index", "title", "rouge1", "rouge2", "rougeL"]

with open('./rouges/caval_100d_25e.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = field_names)
    writer.writeheader()
    writer.writerows(all_scores)
