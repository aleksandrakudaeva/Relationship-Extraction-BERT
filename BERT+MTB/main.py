from src.finetuning.trainer import train_and_fit, evaluate
from src.pretraining.trainer import pretrain
from argparse import ArgumentParser
import logging

'''
This fine-tunes the BERT model on SemEval2010-8 task
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, default='./data/German/TRAIN_FILE_DE.TXT', \
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str, default='./data/German/TEST_FILE_DE.TXT', \
                        help="test data .txt file path")
    parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--num_epochs", type=int, default=11, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
    parser.add_argument("--model_size", type=str, default='base', help="base or large")
    parser.add_argument("--model_name", type=str, default='bert-base-german-dbmdz-uncased', help="'bert-base-uncased', \
                                                                                                    'bert-base-multilingual-uncased'\
                                                                                                    'bert-base-multilingual-cased'")
    parser.add_argument("--pretrain", type=int, default=0, help="0: MTB pre-training, 1: fine-tuning")                                                                                                
    parser.add_argument("--train", type=int, default=0, help="0: Don't train, 1: train")
    parser.add_argument("--eval", type=int, default=1, help="0: Don't eval, 1: eval")
    
    args = parser.parse_args()

    if (args.pretrain == 1):
        net = pretrain(args)
    elif (args.train == 1):
        net = train_and_fit(args)
    elif (args.eval == 1):
        net = evaluate(args)