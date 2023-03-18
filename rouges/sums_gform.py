import pandas as pd
from datasets import load_dataset

'''
GOAL:
- compare PIGASUS finetune 100 vs. PEGASUS^2 finetune 100
- five bills each
    - #1 bill for averaged ROUGE-Ls
    - 3 bills at median (median, med + 1, med-1) for averaged ROUGE-Ls
    - last bill for averaged ROUGE-Ls
'''

SEPARATOR = "=================================="

dataset = load_dataset('billsum', split="ca_test")
summaries = dataset["summary"][1051:]

pig_sums = open('test_summaries/Test-Pigasus.txt', 'r').readlines()[0].split(SEPARATOR)[:-1]
peg2_sums = open('test_summaries/Test-Pretrain-squared.txt', 'r').readlines()[0].split(SEPARATOR)[:-1]

# print(len(pig_sums), len(peg2_sums))

pig_df = pd.read_csv("rouges/catest_ourpretrain_100.csv")
peg2_df = pd.read_csv("rouges/catest_theirpretrain_100.csv")

# print(pig_df.shape, peg2_df.shape)

average = (pig_df["rougeL"] + peg2_df["rougeL"])/2

d = {"PIG RL": pd.Series(pig_df["rougeL"]), "PEG2 RL": pd.Series(peg2_df["rougeL"]), "AVG RL": pd.Series(average)}
rougel_df = pd.DataFrame(data=d)
# print(rougel_df)

max_idx = average.idxmax()
min_idx = average.idxmin()

min1_i, med_i, plu1_i = 0, 0, 0

s_list = list(average)
s_list.sort()

med_idx = len(s_list)//2
min1_i = med_idx - 1
pl1_i = med_idx + 1

for i, x in enumerate(list(s_list)):
    if i == min1_i:
        MEDMIN1 = x
    if i == med_idx:
        MEDIAN = x
    if i == pl1_i:
        MEDPL1 = x

# print(MEDMIN1, MEDIAN, MEDPL1)

for i, x in enumerate(average):
    if x == MEDMIN1:
        print('medmin1', x)
        min1_idx = i
    elif x == MEDIAN:
        print('med', x)
        med_idx = i
    elif x == MEDPL1:
        print('medpl1', x)
        pl1_idx = i

# print(rougel_df.nsmallest(2, 'AVG RL'))
# print(max_idx, min_idx, min1_idx, med_idx, pl1_idx)

print("")
print("number 1 average summaries ===============")
print('TARGET')
print(summaries[max_idx])

print('PEGASUS SQUARED')
print(peg2_sums[max_idx])

print('PIGASUS')
print(pig_sums[max_idx])

print("")
print("med 1 average summaries ===============")
print('TARGET')
print(summaries[min1_idx])

print('PEGASUS SQUARED')
print(peg2_sums[min1_idx])

print('PIGASUS')
print(pig_sums[min1_idx])

print("med 2 average summaries ===============")
print('TARGET')
print(summaries[med_idx])

print('PEGASUS SQUARED')
print(peg2_sums[med_idx])

print("")
print('PIGASUS')
print(pig_sums[med_idx])

print("")
print("med 3 average summaries ===============")
print('TARGET')
print(summaries[pl1_idx])

print('PEGASUS SQUARED')
print(peg2_sums[pl1_idx])

print('PIGASUS')
print(pig_sums[pl1_idx])

print("")
print("worst average summaries ===============")
print('TARGET')
print(summaries[min_idx])

print('PEGASUS SQUARED')
print(peg2_sums[min_idx])

print('PIGASUS')
print(pig_sums[min_idx])
