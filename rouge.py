from rouge_score import rouge_scorer
from datasets import load_dataset
from tqdm import tqdm
import csv

SEPARATOR = "=================================="

# load datasets
billsum_test = load_dataset('billsum', split="ca_test")
target_summaries = billsum_test['summary'][865:1051]
bill_names = billsum_test['title'][865:1051]

#test_extsumms = "".join(open('extractive_summaries.txt', 'r').readlines())
#test_extsumms = test_extsumms.split(SEPARATOR)

# test_pegasussumms = "".join(open('pegasus_summaries.txt', 'r').readlines())
# test_pegasussumms = test_pegasussumms.split(SEPARATOR)

test_casumms = "".join(open('summaries_catest.txt', 'r').readlines())
test_casumms = test_casumms.split(SEPARATOR)[:-1]


print(len(test_casumms), len(bill_names), len(target_summaries))

# calculate rouge scores between extractive summary and billsum summary
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
all_scores = []

for i in tqdm(range(len(bill_names))):
    score = scorer.score(test_casumms[i], target_summaries[i])
    # parse + save score data type
    all_scores.append({'title': bill_names[i], 'rouge1': score['rouge1'].fmeasure, 'rouge2': score['rouge2'].fmeasure, 'rougeL': score['rougeL'].fmeasure})


    
# write rouge scores to CSV file
field_names = ["title", "rouge1", "rouge2", "rougeL"]

with open('catest_rouges.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = field_names)
    writer.writeheader()
    writer.writerows(all_scores)
