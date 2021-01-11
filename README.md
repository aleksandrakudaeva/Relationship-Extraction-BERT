# Relationship-Extraction-BERT

This directory contains the implementation of several models described in the Master thesis: multilingual BERT-based relationship extraction.  

Folder "BERT+MTB" contains all core scripts, results of which were documented in the thesis. The scripts were based on an open-source implementation available at https://github.com/plkmo/BERT-Relation-Extraction. 

Folder "BERT fine-tuning (alternative)" contains an alternative implementation of the model. 

The scripts use several datasets: 
1) SemEval2010-8 (data/English) accessed from https://github.com/sahitya0000/Relation-Classification/blob/master/corpus/SemEval2010_task8_all_data.zip
2) SemEval2010-8-de (data/German) self-created dataset in German
3) Multilingual dataset containing examples in English and German (data/Multilingual)

## Requirements

torch 1.7.0+cuda11
transformers 3.5.1
spacy 2.3.2
