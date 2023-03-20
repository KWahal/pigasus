# Our baseline extractive summarization. 
# Modified from: https://github.com/DerwenAI/pytextrank/blob/main/examples/explain_summ.ipynb
# To load the spaCy model, first RUN:
#   python -m spacy download en_core_web_sm
#   python -m spacy download en

import spacy
import pytextrank
from math import sqrt
import argparse
from datasets import load_dataset
from tqdm import tqdm

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

# this function evaluates our extractive summary generator on the billsum CA test split
def evaluate():
    billsum_test = load_dataset('billsum', split="ca_test")
    billsum_test = billsum_test[1051:]
    test_texts = billsum_test['text']
    print('loaded dataset. starting evaluation')

    limit_phrases = 10
    limit_sentences = 3
    test_summaries = []

    for t in tqdm(test_texts):
        s = generate_extractive_summary(t, limit_phrases, limit_sentences)
        test_summaries.append(s)

    print('evaluation on billsum CA test done. writing summaries.')
    with open(r'test_summaries/extractive_summaries_catest.txt', 'w') as fp:
        for item in test_summaries:
            # write each item on a new line
            fp.write("%s\n" % item)
            fp.write("==================================\n")


if __name__ == "__main__":
    ''' To test from terminal
    parser = argparse.ArgumentParser(
                        prog = 'ExtractiveSummary',
                        description = 'Our baseline: generate an X-sentence summary of given text.')

    parser.add_argument('text', metavar='t', type=str, nargs=1,
                        help='path to text to summarize')
    parser.add_argument('limit_phrases', metavar='p', type=str, nargs=1,
                        help='number of important phrases to consider')
    parser.add_argument('limit_sentences', metavar='s', type=str, nargs=1,
                        help='length of summary (in sentences)')


    args = parser.parse_args()
    text = open(str(args.text[0]), 'r').readlines()[0]

    limit_phrases = int(args.limit_phrases[0])
    limit_sentences = int(args.limit_sentences[0])

    summary = generate_extractive_summary(text, limit_phrases, limit_sentences)
    print(summary)'''

    # run on billsum test set
    evaluate()