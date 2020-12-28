import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertPreTrainedModel, BertTokenizer
from torch.nn import MSELoss, CrossEntropyLoss


class BertForRE(BertPreTrainedModel):
#add description here
    def __init__(self, config):
        super(BertForRE, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        classifier_size = config.hidden_size*2
        self.classifier = nn.Linear(classifier_size, config.num_labels)

        self.init_weights

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, e1=None, e2=None, labels=None,
                position_ids=None, head_mask=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,\
                            attention_mask=attention_mask, head_mask=head_mask)
       
        pooled_output   = outputs[1] # output for classification token
        sequence_output = outputs[0] # outputs for all tokens

        # extract outputs for the entity start tokens
        e1_h = sequence_output[:,e1,:]
        e2_h = sequence_output[:,e2,:]

        output_stacked = []
        for i in range(sequence_output.size()[0]): 
            e1_h_clean = e1_h[i,i,:]
            e2_h_clean = e2_h[i,i,:]
            # concatenate entity start tokens representations
            output = torch.cat([e1_h_clean, e2_h_clean], dim=-1)
            output_stacked.append(output)
        
        # reformat to pytorch tensor
        pooled_output = torch.stack([a for a in output_stacked], dim=0)
    
        # pass representation to the classification layer
        logits = self.classifier(pooled_output)

        return logits