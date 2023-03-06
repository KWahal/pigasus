import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

SEPARATOR = "=================================="
num_beams = 1
num_return_sequences = 1
#input_text = "This bill authorizes the Contra Costa Transportation Authority to conduct a pilot project for the testing of autonomous vehicles that do not have a driver seated in the driver seat and are not equipped with a steering wheel, a brake pedal, or an accelerator. The testing shall be conducted only at a privately owned business park designated by the authority, inclusive of public roads within the designated business park, and at GoMentum Station located within the boundaries of the former Concord Naval Weapons Station. The autonomous vehicle shall operate at speeds of less than 35 miles per hour. Before the testing of an autonomous vehicle that does not have a driver seated in the driver seat, the operator, and a combination of the two, shall: obtain an instrument of insurance, surety bond, or proof of self-insurance in an amount of five million dollars $5 million; and provide evidence of such insurance to the California Department of Motor Vehicles."
# # output:
# ['The test of your knowledge is your ability to convey it.',
#  'The ability to convey your knowledge is the ultimate test of your knowledge.',
#  'The ability to convey your knowledge is the most important test of your knowledge.',
#  'Your capacity to convey your knowledge is the ultimate test of it.',
#  'The test of your knowledge is your ability to communicate it.',
#  'Your capacity to convey your knowledge is the ultimate test of your knowledge.',
#  'Your capacity to convey your knowledge to another is the ultimate test of your knowledge.',
#  'Your capacity to convey your knowledge is the most important test of your knowledge.',
#  'The test of your knowledge is how well you can convey it.',
#  'Your capacity to convey your knowledge is the ultimate test.']


def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

original_texts = "".join(open('summaries/summaries_catest.txt', 'r').readlines())
original_texts = original_texts.split(SEPARATOR)[:-1]

original_texts = original_texts[:2]

all_paraphrased = []
for text in original_texts:
  sentences = text.split(".")
  sentences = sentences[:len(sentences)-1]
  paraphrased = []
  for sentence in sentences:
    paraphrased.append(get_response(sentence, num_return_sequences, num_beams))
  all_paraphrased.append(paraphrased)

for paraphrase in all_paraphrased:
  print(SEPARATOR)
  for sentence in paraphrase:
    print(sentence)

