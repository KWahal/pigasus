from rouge_score import rouge_scorer
from datasets import load_dataset
from tqdm import tqdm




billsum_test = load_dataset('billsum', split="test")

test_extsumms = open('extractive_summaries.txt', 'r').readlines()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

target_summaries = billsum_test['summary']

# for i in range(len(target_summaries)):
for i in range(3):
    print(test_extsumms[i])
    print(target_summaries[i])

    # scores = scorer.score(test_extsumms[i], target_summaries[i])


scores = scorer.score('The quick brown fox jumps over the lazy dog',
                      'The quick brown dog jumps on the log.')



print(scores)

''' {'rouge1': Score(precision=0.75, recall=0.6666666666666666, fmeasure=0.7058823529411765), 
    'rougeL': Score(precision=0.625, recall=0.5555555555555556, fmeasure=0.5882352941176471)}'''