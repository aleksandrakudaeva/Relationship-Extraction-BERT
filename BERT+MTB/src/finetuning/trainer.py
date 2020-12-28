#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:53:55 2019

@author: weetee
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_state, load_results, evaluate_, evaluate_results
from ..pretraining.misc import save_as_pickle, load_pickle
import matplotlib.pyplot as plt
import time
import logging
from model.modeling_bert import BertModel as Model

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def train_and_fit(args):
    
    amp = None
    cuda = torch.cuda.is_available()
    
    train_loader, test_loader, train_len, test_len = load_dataloaders(args)
    logger.info("Loaded %d Training samples." % train_len)
    
    # load pre-trained BERT model
    net = Model.from_pretrained(args.model_name, force_download=False, \
                            model_size=args.model_size,
                            task='classification',\
                            n_classes_=args.num_classes)
    
    
    # load saved tokeniser
    tokenizer = load_pickle("BERT_tokenizer.pkl")
    net.resize_token_embeddings(len(tokenizer))
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1
    
    # check if GPU is available
    if cuda:
        net.cuda()

    # freezing all hidden lazers except the last one   
    logger.info("FREEZING MOST HIDDEN LAYERS...")
    unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                           "classification_layer", "blanks_linear", "lm_linear", "cls"]
        
    # print out list of layers and their status
    for name, param in net.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            print("[FREE]: %s" % name)
            param.requires_grad = True
    
    
    # load MTB pre-trained model weights
    if args.use_pretrained_blanks == 1:
        logger.info("Loading model pre-trained on blanks at ./data/test_checkpoint_MTB.pth.tar...")
        checkpoint_path = "./data/test_checkpoint_MTB.pth.tar"
        checkpoint = torch.load(checkpoint_path)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        net.load_state_dict(pretrained_dict, strict=False)
        del checkpoint, pretrained_dict, model_dict
    

    # define hyperparameters
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                      24,26,30], gamma=0.8)
    
    # load the last saved checkpoint
    start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)  
    
    losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch = load_results()
    
    logger.info("Starting training process...")
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id

    # 
    update_size = len(train_loader)//10
    
    # training loop
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        net.train() # model in training mode
        total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; accuracy_per_batch = [] 
        
        for i, data in enumerate(train_loader, 0):
            x, e1_e2_start, labels, _,_,_ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                
            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None, e1_e2_start=e1_e2_start)
            
            loss = criterion(classification_logits, labels.squeeze(1))
            loss = loss/args.gradient_acc_steps   
            loss.backward()
            
            grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)
            
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            total_acc += evaluate_(classification_logits, labels, \
                                   ignore_idx=-1)[0]
            
            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                accuracy_per_batch.append(total_acc/update_size)
                print('[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f' %
                      (epoch + 1, (i + 1)*args.batch_size, train_len, losses_per_batch[-1], accuracy_per_batch[-1]))
                total_loss = 0.0; total_acc = 0.0
        
        scheduler.step()
        
        results = evaluate_results(net, test_loader, pad_id, cuda)
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
        test_f1_per_epoch.append(results['f1'])
        
        print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
        print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
        print("Train accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
        print("Test f1 at Epoch %d: %.7f" % (epoch + 1, test_f1_per_epoch[-1]))
        
        if accuracy_per_epoch[-1] > best_pred:
            best_pred = accuracy_per_epoch[-1]
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./data/" , "task_test_model_best_BERT.pth.tar"))
        
        if (epoch % 1) == 0:
            save_as_pickle("task_test_losses_per_epoch_BERT.pkl", losses_per_epoch)
            save_as_pickle("task_train_accuracy_per_epoch_BERT.pkl", accuracy_per_epoch)
            save_as_pickle("task_test_f1_per_epoch_BERT.pkl", test_f1_per_epoch)
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./data/" , "task_test_checkpoint_BERT.pth.tar"))
    
    logger.info("Finished Training!")
    
    return net

def evaluate(args):
    
    amp = None
    cuda = torch.cuda.is_available()
    
    train_loader, test_loader, train_len, test_len = load_dataloaders(args)
    logger.info("Loaded %d Training samples." % train_len)
    
    # load pre-trained BERT model 
    net = Model.from_pretrained(args.model_name, force_download=False, \
                            model_size=args.model_size,
                            task='classification',\
                            n_classes_=args.num_classes)
    
    
    # load saved tokeniser
    tokenizer = load_pickle("BERT_tokenizer.pkl")
    net.resize_token_embeddings(len(tokenizer))
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1
    
    # check if GPU is available
    if cuda:
        net.cuda()

    # freezing all hidden lazers except the last one   
    logger.info("FREEZING MOST HIDDEN LAYERS...")
    unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                           "classification_layer", "blanks_linear", "lm_linear", "cls"]
        
    # print out list of layers and their status
    for name, param in net.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            print("[FROZE]: %s" % name)
            param.requires_grad = False
        else:
            print("[FREE]: %s" % name)
            param.requires_grad = True
    
    
    # load MTB pre-trained model weights
    if args.use_pretrained_blanks == 1:
        logger.info("Loading model pre-trained on blanks at ./data/test_checkpoint_BERT.pth.tar...")
        checkpoint_path = "./data/test_checkpoint_BERT.pth.tar"
        checkpoint = torch.load(checkpoint_path)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        net.load_state_dict(pretrained_dict, strict=False)
        del checkpoint, pretrained_dict, model_dict
    

    # define hyperparameters
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                      24,26,30], gamma=0.8)
    
    # load the last saved checkpoint
    start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)  
    
    losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch = load_results()
    
    logger.info("Starting training process...")
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id

    # 
    update_size = len(train_loader)//10
    
    # training loop
    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        net.train() # model in training mode
        total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; accuracy_per_batch = [] 
        
        for i, data in enumerate(train_loader, 0):
            x, e1_e2_start, labels, _,_,_ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                
            classification_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                          e1_e2_start=e1_e2_start)
            
            
            loss = criterion(classification_logits, labels.squeeze(1))
            loss = loss/args.gradient_acc_steps   
            loss.backward()
            
            grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)
            
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            total_acc += evaluate_(classification_logits, labels, \
                                   ignore_idx=-1)[0]
            
            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                accuracy_per_batch.append(total_acc/update_size)
                print('[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f' %
                      (epoch + 1, (i + 1)*args.batch_size, train_len, losses_per_batch[-1], accuracy_per_batch[-1]))
                total_loss = 0.0; total_acc = 0.0
        
        scheduler.step()
        
        results = evaluate_results(net, test_loader, pad_id, cuda)
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
        test_f1_per_epoch.append(results['f1'])
        
        print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
        print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
        print("Train accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
        print("Test f1 at Epoch %d: %.7f" % (epoch + 1, test_f1_per_epoch[-1]))
        
        if accuracy_per_epoch[-1] > best_pred:
            best_pred = accuracy_per_epoch[-1]
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./data/" , "task_test_model_best_BERT.pth.tar"))
        
        if (epoch % 1) == 0:
            save_as_pickle("task_test_losses_per_epoch_BERT.pkl", losses_per_epoch)
            save_as_pickle("task_train_accuracy_per_epoch_BERT.pkl", accuracy_per_epoch)
            save_as_pickle("task_test_f1_per_epoch_BERT.pkl", test_f1_per_epoch)
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./data/" , "task_test_checkpoint_BERT.pth.tar"))
    
    logger.info("Finished Training!")
    
    return net