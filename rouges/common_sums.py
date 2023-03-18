from datasets import load_dataset
from tqdm import tqdm

SEPARATOR = "=================================="

billsum_test = load_dataset('billsum', split="ca_test")

target_summaries = billsum_test['summary']
bill_texts = billsum_test['text']
pegsquared_summaryOG = open("./test_summaries/Test-Pretrain-squared.txt", "r")
pig_summaryOG = open("./test_summaries/Test-Pigasus.txt", "r")

pegsquared_summary = pegsquared_summaryOG.read()
pegsquared_summary = pegsquared_summary.replace('\n', ' ').split(SEPARATOR)
pegsquared_summaryOG.close()

pig_summary = pig_summaryOG.read()
pig_summary = pig_summary.replace('\n', ' ').split(SEPARATOR)
pig_summaryOG.close()

testers = [1118, 1090, 1203, 1151]

with open('./key_summs/COMMON_GOOD_SUMS.txt', 'w') as f:
    f.write('Common good summaries')
    f.write('\n')
    for i in testers:
        f.write(SEPARATOR)
        f.write('target summary: ')
        f.write(target_summaries[i])
        f.write('\n')
        f.write('bill text: ')
        f.write(bill_texts[i])
        f.write('\n')
        f.write('peg squared summary: ')
        f.write(pegsquared_summary[i-1051])
        f.write('\n')
        f.write('pigasus summary: ')
        f.write(pig_summary[i-1051])
        f.write('\n')
