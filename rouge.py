from rouge_score import rouge_scorer
from datasets import load_dataset
from tqdm import tqdm
import csv

SEPARATOR = "=================================="

# load datasets
billsum_test = load_dataset('billsum', split="test")
target_summaries = billsum_test['summary']

# create target sums file
# with open(r'target_sums.txt', 'w') as fp:
#         for item in target_summaries:
#             # write each item on a new line
#             fp.write("%s\n" % item)
#             fp.write("==================================\n")

# # Create original bills file
# original_bills = billsum_test['text']
# with open(r'original_bills.txt', 'w') as fp:
#         for item in original_bills:
#             # write each item on a new line
#             fp.write("%s\n" % item)
#             fp.write("==================================\n")


bill_names = billsum_test['title']

#test_extsumms = "".join(open('extractive_summaries.txt', 'r').readlines())
#test_extsumms = test_extsumms.split(SEPARATOR)

test_pegasussumms = "".join(open('pegasus_summaries.txt', 'r').readlines())
test_pegasussumms = test_pegasussumms.split(SEPARATOR)

# calculate rouge scores between extractive summary and billsum summary
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
all_scores = []

for i in tqdm(range(len(bill_names))):
    score = scorer.score(test_pegasussumms[i], target_summaries[i])
    # parse + save score data type
    all_scores.append({'title': bill_names[i], 'rouge1': score['rouge1'].fmeasure, 'rouge2': score['rouge2'].fmeasure, 'rougeL': score['rougeL'].fmeasure})


    
# write rouge scores to CSV file
field_names = ["title", "rouge1", "rouge2", "rougeL"]

with open('pegasussumms_rouges.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = field_names)
    writer.writeheader()
    writer.writerows(all_scores)
