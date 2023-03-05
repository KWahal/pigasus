from datasets import load_dataset

# Download data.
dataset = load_dataset('billsum', split="ca_test") # list of dictionaries

train_dataset = Dataset(dataset[:865].update(dataset[1051:]))


print(type(dataset))
# # doc_data = dataset["text"]
train_data = dataset[:865]  # why isnt this a list of dictionaries
print(type(train_data))

# train_data.append(dataset[1051:]) # combine train + val
# test_data = dataset[865:1051]

# print(train_data[-1])