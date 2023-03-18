# find most important 30% of sentences using textrank
# mask out those sentences with <mask_1>
# return the resulting text (sentences concatenated together) in separate file

# OLD BILL DATA - the entire bill
# OLD SUMMARY LABEL - the original summary

# NEW BILL DATA - entire bill, w 30% most imp sentences masked out.
# NEW SUMMARY LABEL - the 30% sentences concatenated together.

import spacy
import pytextrank
from math import sqrt
import argparse
from datasets import load_dataset
from tqdm import tqdm
import json

# Textrank code
def generate_extractive_summary(text, limit_phrases, limit_sentences):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("textrank")
    doc = nlp(text)

    # print out the top ranked phrases
    # for p in doc._.phrases:
    #     print(p.text)
    #     print("{:.4f} {:5d} {}".format(p.rank, p.count, p.text))
    #     print(p.chunks)

    # generate the sentence bounds, with empty phrase vector for each sentence.
        # this phrase vector will be used to measure each sentence's euclidean distance from 
        # the unit vector, which is a vector with the p.rank for each phrase, sorted
    # Each phrase vector includes the phrase_id of each phrase occurring in the sentence. 
    sent_bounds = [ [s.start, s.end, set([])] for s in doc.sents ]

    # generate each sentence's phrase vector + the unit vector
        # limited to top X phrases, param
    phrase_id = 0
    unit_vector = []
    for p in doc._.phrases:
        # print("{:5d} {:.4f} {}".format(phrase_id, p.rank, p.text))
        
        unit_vector.append(p.rank)
        
        for chunk in p.chunks:
            # print("{:5d} {:5d}".format(chunk.start, chunk.end))

            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.end <= sent_end:
                    # print("Chunk: {:5d} {:5d} ; Sentence: {:5d} {:5d}".format(chunk.start, chunk.end, sent_start, sent_end))
                    sent_vector.add(phrase_id)
                    break

        phrase_id += 1

        if phrase_id == limit_phrases:
            break

    # normalize the unit vector
    sum_ranks = sum(unit_vector)
    unit_vector = [ rank/sum_ranks for rank in unit_vector ]

    # Calculate the euclidean distance between each sentence's phrase vector + the unit vector 
    sent_rank = {}
    sent_id = 0

    for sent_start, sent_end, sent_vector in sent_bounds:
        # print(sent_vector)
        sum_sq = 0.0
        for phrase_id in range(len(unit_vector)):
            # print("{:5d} {:.4f}".format(phrase_id, unit_vector[phrase_id]))

            if phrase_id not in sent_vector:
                sum_sq += unit_vector[phrase_id]**2.0

        sent_rank[sent_id] = sqrt(sum_sq)
        sent_id += 1

    # Sort sent_rank in order of ascending euclidean distance from the unit vector
    sent_rank = sorted(sent_rank.items(), key=lambda x: x[1], reverse=False)

    # Get original sentences from text
    sent_text = {}
    sent_id = 0

    for sent in doc.sents:
        sent_text[sent_id] = sent.text
        sent_id += 1

    # Generate extractive summary, with limit_sentences (param) number of sentences
    num_sent = 0
    summary = "" 
    for sent_id, rank in sent_rank:
        # print(str(sent_id) + ": " + sent_text[sent_id])
        summary += sent_text[sent_id]
        num_sent += 1

        if num_sent == limit_sentences:
            break
    
    return summary

with open('datasets/train.txt') as f:
    lines = f.readlines()

    start_loc = 12
    all_outputs = []

    lines = lines[:5]
    for line in tqdm(lines):
       if len(line) >= 1000000:
           continue

       end_loc = line.find('"summary"') - 3
       report = line[start_loc:end_loc]
       sentences = report.split('.')
       num_sentences = 3*len(sentences)//10
      # print("num sentences is " + str(num_sentences))
       summary = generate_extractive_summary(report, 15, num_sentences)
       #print(summary)
       summary_sentences = summary.split('.')
       print(summary_sentences)

       summary_sentences = set(summary.split('.'))
       summary_sentences = list(summary_sentences)
      # print("summary sentence number is " + str(len(summary_sentences)))
       ordered_summaries = {}

        # Resort summaries in order, not by rank
       for i in range(len(summary_sentences)):
           location = report.find(summary_sentences[i])
           ordered_summaries[i] = location
        
     #  print("first dict is " + str(ordered_summaries))
       sorted_summaries = dict(sorted(ordered_summaries.items(), key=lambda x:x[1]))
    #   print("sorted dict is " + str(sorted_summaries))
       final_summaries = []
       for key, value in sorted_summaries.items():
           this_summary = summary_sentences[key]
           this_summary += ". "
           final_summaries.append(this_summary)

       
       final_summary = ''.join(final_summaries)
       for sentence in summary_sentences:
          # print(sentence)
           #print(report.count(sentence))
           report = report.replace(sentence, "<mask_1>", 1)
    

      # print(report)
       counts = report.count("<mask_1>")
     #  print("counts = " + str(counts))

       output = {}
       output["inputs"] = report
       output["labels"] = final_summary
       all_outputs.append(output)

    json_object = json.dumps(all_outputs)
    #with open("government_documents.json", "w") as outfile:
        #outfile.write(json_object)
      # print(summary)






# import jsonlines

# with jsonlines.open('train.uscode.jsonl') as f:
#     for line in f.iter():
#         print(len(line['text']))

      #  print(line['text'][:1000])
    

# please install HuggingFace datasets by pip install datasets 

# multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20220616")
# # Download multi_lexsum locally and load it as a Dataset object 

# train_dataset = multi_lexsum["train"]
# print(len(train_dataset))
# example = multi_lexsum["train"][0] # The first instance of the train set 
# # print(example["sources"]) # A list of source document text for the case
# # print("_____________")
# # print(len(example["sources"]))
# # print("_____________")
# # print(example["sources"][0])
# # print(example["summary/short"])
# # for sum_len in ["long", "short", "tiny"]:
# #    print(example["summary/" + sum_len]) # Summaries of three lengths

# iterations = len(train_dataset)
# # iterations = 1
# for i in range(iterations):
#     example = multi_lexsum["train"][i]
#     source = ''.join(example["sources"])
#     if len(source) < 20000:
#         print(source)
#         sentences = source.split('.')
#         num_sentences = 3*len(sentences)//10
#         summary = generate_extractive_summary(source, 15, num_sentences)
#         print("________________________")
#         print("________________________")
#         print(summary)
#         print("good iteration " + str(i))
#         break
#     print("iteration " + str(i))

