# Prepare data for pretraining by following the method used in PEGASUS:
# Mask out the top 30% sentences by rouge1 score; rouge scores calculated independently and with repetition if necessary

import spacy
import pytextrank
from math import sqrt
import argparse
from datasets import load_dataset
from tqdm import tqdm
import json
from rouge_score import rouge_scorer
import csv
from collections import OrderedDict


scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

with open('datasets/train.txt') as f:
    lines = f.readlines()

    lines = lines[0:1000]
    start_loc = 12
    all_outputs = []
    for line in tqdm(lines):
       if len(line) >= 1000000:
           continue

       end_loc = line.find('"summary"') - 3
       report = line[start_loc:end_loc]
       sentences = report.split('.')
       sentences = sentences[:len(sentences)-1]
       num_sentences = 3*len(sentences)//10

       all_scores = {}
       for i in range(len(sentences)):
           new_report = report.replace(sentences[i] + ".", "", 1)
           score = scorer.score(sentences[i], new_report)
           rouge1 = score['rouge1'].fmeasure
           all_scores[i] = rouge1
    
       sorted_sentences = dict(sorted(all_scores.items(), key=lambda x:x[1]))

       filtered_sentences = {}

       # Find top 30% of sentences
       i = 0
       for key, value in sorted_sentences.items():
           if i == num_sentences:
               break
           filtered_sentences[key] = value
           i += 1
       
       # Resort dictionary to be in order
       filtered_sentences = OrderedDict(sorted(filtered_sentences.items()))
       # mask out sentences
       final_summary = ""
       for key in filtered_sentences.keys():
           sentence = sentences[key] + "."
           #print(sentence)
           report = report.replace(sentence, "<mask_1>.", 1)
           final_summary += sentence + " "

       counts = report.count("<mask_1>.")
     #  print("counts is " + str(counts))
       output = {}
       output["inputs"] = report
       output["labels"] = final_summary
       all_outputs.append(output)


    json_object = json.dumps(all_outputs)
    with open("pegasus_masked_gov_docs.json", "w") as outfile:
        outfile.write(json_object)
