import torch

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

def get_e1e2_start(x, e1_id, e2_id):
    try:
        e1_e2_start = ([i for i, e in enumerate(x) if e == e1_id][0],\
                        [i for i, e in enumerate(x) if e == e2_id][0])
    except Exception as e:
        e1_e2_start = None
        print(e)
    return e1_e2_start

def evaluate(logits, labels):
    o = torch.softmax(logits, dim = 1).max(1)[1]
    l = labels.squeeze()

    return o, l

