import torch
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, SequentialSampler
from data_preprocessing import get_semeval, preprocess_semeval
from helpers import rel_dict, get_e1e2_start , evaluate
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig
import pandas as pd
import pickle
import os
from model import BertForRE
from argparse import ArgumentParser
from config import Config
print('all packages loaded...')
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import f1_score
from tqdm import tqdm


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True
print('device defined: ', device)

# Parameters
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 1, 
          'train_data': 'data/English/TRAIN_FILE_en.TXT',
          'test_data1': 'data/German/TEST_FILE_de.TXT', 
          'test_data2': 'data/English/TEST_FILE_en.TXT'
          }
max_epochs = 11
num_labels = 19 # ToDo: check

# Config
config = Config('src/config.ini')

bertconfig = BertConfig.from_pretrained(
        config.pretrained_model_name, 
        num_labels = num_labels, 
        finetuning_task = config.task_name
        )

# Model
model = BertForRE.from_pretrained(config.pretrained_model_name, config = bertconfig)
print("Model loaded")

# check if train loader is already created 
if os.path.isfile('./checkpoints/train_loader.pkl'):
        # load train loader and tokenizer
        with open('./checkpoints/tokeniser.pkl', 'rb') as t:
                tokenizer = pickle.load(t)
        with open('./checkpoints/train_loader.pkl', 'rb') as trl:
                train_dataloader = pickle.load(trl)
        with open('./checkpoints/test_loader_de.pkl', 'rb') as tsl:
                test_dataloader_de = pickle.load(tsl)
        with open('./checkpoints/test_loader_en.pkl', 'rb') as tsl:
                test_dataloader_en = pickle.load(tsl)

        print('Dataloaders loaded...')
else:
        # read and transform original data
        train_df = get_semeval(params['train_data'])
        test_df_de = get_semeval(params['test_data1'])
        test_df_en = get_semeval(params['test_data2'])
        print('data read...')

        # create dictionary 
        rel2idx, idx2rel = rel_dict(train_df['relations'])
        additional_special_tokens = ['<e1>', '</e1>', '<e2>', '</e1>', '[BLANK]']

        # create tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False) 
        tokenizer.add_tokens(additional_special_tokens)
        e1_id = tokenizer.convert_tokens_to_ids('<e1>')
        e2_id = tokenizer.convert_tokens_to_ids('<e2>')
        print('tokenizer created...')

        # preprocess data
        tr_ids, tr_labels, tr_attention_mask, tr_e1, tr_e2 = preprocess_semeval(train_df, rel2idx, tokenizer, e1_id, e2_id)
        print('train sample preprocessed...')
        vl_ids_de, vl_labels_de, vl_attention_mask_de, vl_e1_de, vl_e2_de = preprocess_semeval(test_df_de, rel2idx, tokenizer, e1_id, e2_id)
        print('German test sample preprocessed...')
        vl_ids_en, vl_labels_en, vl_attention_mask_en, vl_e1_en, vl_e2_en = preprocess_semeval(test_df_en, rel2idx, tokenizer, e1_id, e2_id)
        print('English test sample preprocessed...')

        # Combine the inputs into a TensorDataset
        train_dataset = TensorDataset(tr_ids, tr_attention_mask, tr_labels, tr_e1, tr_e2)
        test_dataset_de  = TensorDataset(vl_ids_de, vl_attention_mask_de, vl_labels_de, vl_e1_de, vl_e2_de)
        test_dataset_en  = TensorDataset(vl_ids_en, vl_attention_mask_en, vl_labels_en, vl_e1_en, vl_e2_en)

        # Generators
        # Sample training samples in random order
        train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), 
                batch_size = params['batch_size'] 
                )

        # For validation we load samples in a fixed order sequentially
        test_dataloader_de = DataLoader(
                test_dataset_de, 
                sampler = SequentialSampler(test_dataset_de), 
                batch_size = params['batch_size']
                )

        test_dataloader_en = DataLoader(
                test_dataset_en, 
                sampler = SequentialSampler(test_dataset_en), 
                batch_size = params['batch_size']
                )

        print("Generators created!")

        # save tokeniser and generators
        with open('./checkpoints/tokeniser.pkl', 'wb') as t:
                pickle.dump(tokenizer, t)
        with open('./checkpoints/train_loader.pkl', 'wb') as trl:
                pickle.dump(train_dataloader, trl)
        with open('./checkpoints/test_loader_de.pkl', 'wb') as tsl:
                pickle.dump(test_dataloader_de, tsl)
        with open('./checkpoints/test_loader_en.pkl', 'wb') as tsl:
                pickle.dump(test_dataloader_en, tsl)

###################################################################################################################################

# Resize model embeddings
model.resize_token_embeddings(len(tokenizer))

# Transfer the model on the GPU
model.to(device)

# freeze most of the hidden layers
unfrozen = ['classifier', 'pooler', 'encoder.layer.11']
for name, param in model.named_parameters():
    if not any([layer in name for layer in unfrozen]):
        param.requires_grad = False

# optimizer and scheduler
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0, 
        num_training_steps = len(train_dataloader) * max_epochs
        )
criterion = nn.CrossEntropyLoss(ignore_index=-1)

print("Loss function, Optimizer and Sceduler initialized")

# create empty lists to store accuracies and f1 scores per epoch for train and test samples
acc_tr, acc_ts_de, acc_ts_en, f1_tr, f1_ts_de, f1_ts_en = [], [], [], [], [], []

# Loop over epochs
for epoch in range(max_epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, max_epochs))
    print('Training...')

    # Set up training parameters
    model.train(); total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; accuracy_per_batch = []
    
    # Create empty tensors for true_labels, predicted_labels
    true_labels, predicted_labels = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

    # Training
    for local_batch in tqdm(train_dataloader):
        
        input_ids, labels, attention_mask, e1, e2 = local_batch
       
        # Transfer to GPU
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        e1 = e1.to(device)
        e2 = e2.to(device)

        # Model computations
        model.zero_grad()

        # Calculate logits and loss
        logits = model(input_ids, attention_mask=attention_mask, labels = labels, e1=e1, e2=e2)
        loss = criterion(logits, labels)

        # backpropagate
        loss.backward()

        # accumulate losses
        total_loss += loss

        # normalize gradient
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()

        # calculate train accuracy 
        o, l = evaluate(logits, labels)

        true_labels = torch.cat((true_labels, o), 0)
        predicted_labels = torch.cat((predicted_labels, l), 0)

        #print(true_labels)

    # calculate accuracy, F1 for train sample
    train_accuracy = (true_labels == predicted_labels).sum().item()/len(predicted_labels)
    train_f1 = f1_score(true_labels.cpu(), predicted_labels.cpu(), average = 'weighted')

    acc_tr.append(train_accuracy)
    f1_tr.append(train_f1)

    print('Train Accuracy: ', train_accuracy)
    print('Train F1-score: ', train_f1)
    print('Train Loss: ', total_loss.item())
    
    print("")
    print('Evaluation German...')

    # Create empty tensors for true_labels, predicted_labels
    true_labels_test, predicted_labels_test = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

    # calculate accuracy, F1 for test sample
    model.eval()
    with torch.no_grad():
        for local_batch in tqdm(test_dataloader_de):
                input_ids, test_labels, attention_mask, e1, e2 = local_batch
                
                # Transfer to GPU
                input_ids = input_ids.to(device)
                test_labels = test_labels.to(device)
                attention_mask = attention_mask.to(device)
                e1 = e1.to(device)
                e2 = e2.to(device)

                test_logits = model(input_ids, attention_mask=attention_mask, labels = test_labels, e1=e1, e2=e2)
                o, l = evaluate(test_logits, test_labels)

                true_labels_test = torch.cat((true_labels_test, o), 0)
                predicted_labels_test = torch.cat((predicted_labels_test, l), 0)
    
    # calculate model accuracy and F1 score on test sample
    test_accuracy = (true_labels_test == predicted_labels_test).sum().item()/len(predicted_labels_test)
    test_f1 = f1_score(true_labels_test.cpu(), predicted_labels_test.cpu(), average = 'weighted')

    acc_ts_de.append(test_accuracy)
    f1_ts_de.append(test_f1)

    print('Test Accuracy: ', test_accuracy)
    print('Test F1-score: ', test_f1)

    print("")
    print('Evaluation English...')

    # Create empty tensors for true_labels, predicted_labels
    true_labels_test, predicted_labels_test = torch.Tensor([]).to(device), torch.Tensor([]).to(device)

    # calculate accuracy, F1 for test sample
    model.eval()
    with torch.no_grad():
        for local_batch in tqdm(test_dataloader_en):
                input_ids, test_labels, attention_mask, e1, e2 = local_batch
                
                # Transfer to GPU
                input_ids = input_ids.to(device)
                test_labels = test_labels.to(device)
                attention_mask = attention_mask.to(device)
                e1 = e1.to(device)
                e2 = e2.to(device)

                test_logits = model(input_ids, attention_mask=attention_mask, labels = test_labels, e1=e1, e2=e2)
                o, l = evaluate(test_logits, test_labels)

                true_labels_test = torch.cat((true_labels_test, o), 0)
                predicted_labels_test = torch.cat((predicted_labels_test, l), 0)
    
    # calculate model accuracy and F1 score on test sample
    test_accuracy = (true_labels_test == predicted_labels_test).sum().item()/len(predicted_labels_test)
    test_f1 = f1_score(true_labels_test.cpu(), predicted_labels_test.cpu(), average = 'weighted')

    acc_ts_en.append(test_accuracy)
    f1_ts_en.append(test_f1)

    print('Test Accuracy: ', test_accuracy)
    print('Test F1-score: ', test_f1)

print('')
print('Training finished!')

# save the checkpoint
torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
}, './checkpoints/model_checkpoint.pth.tar')

torch.save({
        'train accuracy': acc_tr, 
        'test accuracy de': acc_ts_de,
        'test accuracy en': acc_ts_en,
        'train f1': f1_tr,
        'test f1 de': f1_ts_de,
        'test f1 en': f1_ts_en
}, './checkpoints/preformance.pt')



