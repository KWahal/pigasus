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
    - ^ long, repetitive summaries
- caval_100d_24e-SHORT.csv => finetuned 100, 25 epochs, val set => SHORTER
- caval_ourpretrain_100_SHORT.csv => our pretraining method (extractive), finetuned on 100 CA bills, shorter version
- caval_theirprrain_100.csv => their pretraining method, finetunde on 100 CA bills
'''

path_to_summs = "./test_summaries/Test-P-large0-again.txt"

billsum_test = load_dataset('billsum', split="ca_test")
target_summaries = billsum_test['summary'][1051:]
bill_names = billsum_test['title'][1051:]
bill_indices = list(range(1051, len(bill_names)+1051))

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

with open('./rouges/catest_PLarge0.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = field_names)
    writer.writeheader()
    writer.writerows(all_scores)
