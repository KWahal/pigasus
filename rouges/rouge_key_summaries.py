import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
import csv

SEPARATOR = "=================================="

csv_df = pd.read_csv("./rouges/catest_theirpretrain_100.csv")
csv_df = pd.read_csv("./rouges/catest_ourpretrain_100.csv")
summary_file = open("./test_summaries/Test-Pretrain-squared.txt", "r")
summary_file = open("./test_summaries/Test-Pigasus.txt", "r")

summary_data = summary_file.read()
summary_list = summary_data.replace('\n', ' ').split(SEPARATOR)
summary_file.close()

# r1 = csv_df["index", "rouge1"]
# r2 = csv_df["index", "rouge2"]
# rl = csv_df["index", "rougeL"]


r1_smallest = csv_df.nsmallest(10, 'rougeL')
r1_largest = csv_df.nlargest(10, 'rougeL')

billsum_test = load_dataset('billsum', split="ca_test")

#print(SEPARATOR)
#print(billsum_8)

r1_smallest_list = r1_smallest["index"].tolist()
r1_largest_list = r1_largest["index"].tolist()

print(r1_smallest_list)
print(r1_largest_list)

target_summaries = []
produced_summaries = []
original_bills = []
r1_scores = []
for i in r1_smallest_list:
    target_summary = billsum_test['summary'][i]
    produced_summary = summary_list[i-1051]
    original_bill = billsum_test['text'][i]
    r1_score = csv_df.loc[csv_df['index'] == i, 'rouge1'].iloc[0]

    target_summaries.append(target_summary)
    produced_summaries.append(produced_summary)
    original_bills.append(original_bill)
    r1_scores.append(r1_score)

target_greatest = []
produced_greatest = []
original_greatest = []
r1_greatest = []
for i in r1_largest_list:
    target_summary = billsum_test['summary'][i]
    produced_summary = summary_list[i-1051]
    original_bill = billsum_test['text'][i]
    r1_score = csv_df.loc[csv_df['index'] == i, 'rouge1'].iloc[0]

    target_greatest.append(target_summary)
    produced_greatest.append(produced_summary)
    original_greatest.append(original_bill)
    r1_greatest.append(r1_score)

with open('./key_summs/catest_Pigasus_keysums.txt', 'w') as f:
    f.write('Lowest 3 R1 scores for PEG^2 tested on CA, can be seen below')
    f.write('\n')
    for i in range(len(target_summaries)):
        f.write('target summary is: ')
        f.write('\n')
        f.write(target_summaries[i])
        f.write('\n')
        f.write('produced summary is: ')
        f.write('\n')
        f.write(produced_summaries[i])
        f.write('\n')
        f.write('original bill is: ')
        f.write('\n')
        f.write(original_bills[i])
        f.write('\n')
        f.write('r1 score is: ')
        f.write('\n')
        f.write(str(r1_scores[i]))
        f.write('\n')
        f.write(SEPARATOR)
        f.write('\n')

    f.write('Highest 3 R1 scores for PEG^2 tested on CA can be seen below')
    f.write('\n')

    for i in range(len(target_greatest)):
        f.write('target summary is: ')
        f.write('\n')
        f.write(target_greatest[i])
        f.write('\n')
        f.write('produced summary is: ')
        f.write('\n')
        f.write(produced_greatest[i])
        f.write('\n')
        f.write('original bill is: ')
        f.write('\n')
        f.write(original_greatest[i])
        f.write('\n')
        f.write('r1 score is: ')
        f.write('\n')
        f.write(str(r1_greatest[i]))
        f.write('\n')
        f.write(SEPARATOR)
        f.write('\n')
        



