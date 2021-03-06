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
from .train_funcs import Two_Headed_Loss, load_state, load_results, evaluate_
from .misc import save_as_pickle, load_pickle
import matplotlib.pyplot as plt
import time
import logging
from model.modeling_bert import BertModel as Model

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def pretrain(args):
    
    amp = None
    cuda = torch.cuda.is_available()
    
    train_loader = load_dataloaders(args)
    train_len = len(train_loader)
    logger.info("Loaded %d pre-training samples." % train_len)
    
    net = Model.from_pretrained(args.model_name, force_download=False, \
                            model_size=args.model_size)
    
    tokenizer = load_pickle("BERT_tokenizer.pkl")
    net.resize_token_embeddings(len(tokenizer))
    e1_id = tokenizer.convert_tokens_to_ids('[E1]')
    e2_id = tokenizer.convert_tokens_to_ids('[E2]')
    assert e1_id != e2_id != 1
    
    if cuda:
        net.cuda()
        
    if args.freeze == 1:
        logger.info("FREEZING MOST HIDDEN LAYERS...")
        unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", "encoder.layer.10",\
                               "encoder.layer.9", "blanks_linear", "lm_linear", "cls"]
        
            
        for name, param in net.named_parameters():
            if not any([layer in name for layer in unfrozen_layers]):
                print("[FROZE]: %s" % name)
                param.requires_grad = False
            else:
                print("[FREE]: %s" % name)
                param.requires_grad = True
       
    criterion = Two_Headed_Loss(lm_ignore_idx=tokenizer.pad_token_id, use_logits=True, normalize=False)
    optimizer = optim.Adam([{"params":net.parameters(), "lr": args.lr}])
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                      24,26,30], gamma=0.8)
    
    start_epoch, best_pred, amp_checkpoint = load_state(net, optimizer, scheduler, args, load_best=False)
    
    losses_per_epoch, accuracy_per_epoch = load_results()
    
    logger.info("Starting training process...")
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    update_size = len(train_loader)//10

    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        net.train() # set model to train mode
        total_loss = 0.0; losses_per_batch = []; total_acc = 0.0; lm_accuracy_per_batch = []
        
        for i, data in enumerate(train_loader, 0):
            x, masked_for_pred, e1_e2_start, _, blank_labels, _,_,_,_,_ = data
            masked_for_pred1 =  masked_for_pred
            masked_for_pred = masked_for_pred[(masked_for_pred != pad_id)]
            if masked_for_pred.shape[0] == 0:
                print('Empty dataset, skipping...')
                continue
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()

            if cuda:
                x = x.cuda(); masked_for_pred = masked_for_pred.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
            
            blanks_logits, lm_logits = net(x, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                          e1_e2_start=e1_e2_start)
            lm_logits = lm_logits[(x == mask_id)]
            
            #return lm_logits, blanks_logits, x, e1_e2_start, masked_for_pred, masked_for_pred1, blank_labels, tokenizer # for debugging now
            if (i % update_size) == (update_size - 1):
                verbose = True
            else:
                verbose = False
                
            loss = criterion(lm_logits, blanks_logits, masked_for_pred, blank_labels, verbose=verbose)
            loss = loss/args.gradient_acc_steps
            
            loss.backward()
            
            grad_norm = clip_grad_norm_(net.parameters(), args.max_norm)
            
            if (i % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            total_acc += evaluate_(lm_logits, blanks_logits, masked_for_pred, blank_labels, \
                                   tokenizer, print_=False)[0]
            
            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(args.gradient_acc_steps*total_loss/update_size)
                lm_accuracy_per_batch.append(total_acc/update_size)
                print('[Epoch: %d, %5d/ %d points] total loss, lm accuracy per batch: %.3f, %.3f' %
                      (epoch + 1, (i + 1), train_len, losses_per_batch[-1], lm_accuracy_per_batch[-1]))
                total_loss = 0.0; total_acc = 0.0
                logger.info("Last batch samples (pos, neg): %d, %d" % ((blank_labels.squeeze() == 1).sum().item(),\
                                                                    (blank_labels.squeeze() == 0).sum().item()))
        
        scheduler.step()
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        accuracy_per_epoch.append(sum(lm_accuracy_per_batch)/len(lm_accuracy_per_batch))
        print("Epoch finished, took %.2f seconds." % (time.time() - start_time))
        print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
        print("Accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
        
        if accuracy_per_epoch[-1] > best_pred:
            best_pred = accuracy_per_epoch[-1]
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./data/" , "test_model_best_BERT.pth.tar"))
        
        if (epoch % 1) == 0:
            save_as_pickle("test_losses_per_epoch_BERT.pkl", losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_BERT.pkl", accuracy_per_epoch)
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': accuracy_per_epoch[-1],\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                    'amp': amp.state_dict() if amp is not None else amp
                }, os.path.join("./data/" , "test_checkpoint_BERT.pth.tar"))
    
    logger.info("Finished Training!")
    
    return net