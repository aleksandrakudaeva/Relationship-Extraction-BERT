import re
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset, random_split


def rel_dict(relations): 
    rel2idx = {}
    idx2rel = {}
    
    n_classes = 0
    for relation in relations:
        if relation not in rel2idx.keys():
            rel2idx[relation] = n_classes
            n_classes += 1
    
    for key, value in rel2idx.items():
        idx2rel[value] = key    

    return rel2idx, idx2rel


def get_semeval(path):
    # read data
    with open(path, encoding="utf-8") as f:
        dataset = f.readlines()

    # create separate instances for sentences and labels in train set
    sentences = dataset[::4]
    labels = dataset[1::4]

    # extract quoted sentences
    sentences = [re.findall("\"(.+)\"", sent)[0] for sent in sentences]

    # create dataframe
    df = pd.DataFrame(data={'sents': sentences, 'relations': labels})

    return df

def preprocess_semeval(df, rel2idx, tokenizer, e1_id, e2_id):         

    # encode relatioships
    relations_id = df.apply(lambda x: rel2idx[x['relations']], axis=1)

    # tokenize sentences
    input_ids =[]
    attention_masks = []
    e1 = []
    e2 = []

    for sent_number, sent in enumerate(df['sents']):
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 256,          # Pad & truncate all sentences.
                            truncation = True,
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )

        # define positions of entity start tokens
        e1_start = [torch.tensor(i) for i, e in enumerate(encoded_dict['input_ids'][0]) if e == e1_id][0]
        e2_start = [torch.tensor(i) for i, e in enumerate(encoded_dict['input_ids'][0]) if e == e2_id][0]
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        # Add entity starts to the list
        e1.append(e1_start)
        e2.append(e2_start)
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.LongTensor(relations_id)
    e1, e2 = torch.LongTensor(e1), torch.LongTensor(e2)

    return input_ids, attention_masks, labels, e1, e2