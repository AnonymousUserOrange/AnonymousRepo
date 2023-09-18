import argparse
import logging
import random
import numpy as np

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
my_logger = logging.getLogger(__name__)

import os
import torch
from utils.config import Configurable
from Codet5.Codet5 import train, test

seed = 4
random.seed(seed)  # Python random module.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)  # Torch CPU random seed module.
torch.cuda.manual_seed(seed)  # Torch GPU random seed module.
torch.cuda.manual_seed_all(seed)  # Torch multi-GPU random seed module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--phase", type=str, required=True,
                           help="train/test")
    argparser.add_argument("--module", type=str, required=True,
                           help="collect log from project/module, e.g. rocketmq/acl")
    
    # model
    argparser.add_argument("--approach", default='codet5', type=str,
                           help="approach to generate test oracle")
    argparser.add_argument("--embedding", default='codet5', type=str,
                           help="embedding model")

    #data
    argparser.add_argument("--output_dir", default='model', type=str,
                           help="The output directory where the model predictions and checkpoints will be written.")
    argparser.add_argument("--num_processes", default=8, type=int,
                           help="number of process on data collecting.")

    #training
    argparser.add_argument("--train_batch_size", default=16, type=int,
                           help="Batch size per GPU/CPU for training.")
    argparser.add_argument("--eval_batch_size", default=4, type=int,
                           help="Batch size per GPU/CPU for evaluation.")
    argparser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                           help="Number of updates steps to accumulate before performing a backward/update pass.")
    argparser.add_argument("--learning_rate", default=1e-4, type=float,
                           help="The initial learning rate for Adam. GCB:2e-5,gru:1e-4")
    argparser.add_argument("--weight_decay", default=0.0, type=float,
                           help="Weight decay if we apply some.")
    argparser.add_argument("--adam_epsilon", default=1e-8, type=float,
                           help="Epsilon for Adam optimizer.")
    argparser.add_argument("--max_grad_norm", default=1.0, type=float,
                           help="Max gradient norm.")
    argparser.add_argument("--epochs", default=100, type=int,
                           help="Total number of training epochs to perform.")
    argparser.add_argument("--max_steps", default=-1, type=int,
                           help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    argparser.add_argument("--warmup_steps", default=0, type=int,
                           help="Linear warmup over warmup_steps.")

    args = argparser.parse_args()
    phase = args.phase
    
    config = Configurable(args, 'config/default.ini')
    args.config = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    my_logger.warning("device: %s", device)

    process_path = os.path.join('dataset','process')
    if not os.path.exists(process_path):
        os.makedirs(process_path)
    
    if phase == 'train':
        train(args,my_logger)
    elif phase == 'test':
        test(args,my_logger)
    else:
        pass

if __name__ == '__main__':
    main()
