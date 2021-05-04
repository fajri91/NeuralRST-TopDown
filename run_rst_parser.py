import sys, time
import numpy as np
import random
from datetime import datetime

sys.path.append(".")

import argparse
import torch

from in_out.reader import Reader
from in_out.util import load_embedding_dict, get_logger
from in_out.preprocess import create_alphabet
from in_out.preprocess import batch_data_variable
from models.vocab import Vocab
from models.metric import Metric
from models.config import Config
from models.architecture import MainArchitecture
from train_rst_parser import predict

def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config_path', required=True)
    args = args_parser.parse_args()
    config = Config(None)
    config.load_config(args.config_path)
    
    logger = get_logger("RSTParser (Top-Down) RUN", config.use_dynamic_oracle, config.model_path)
    word_alpha, tag_alpha, gold_action_alpha, action_label_alpha, relation_alpha, nuclear_alpha, nuclear_relation_alpha, etype_alpha = create_alphabet(None, config.alphabet_path, logger)
    vocab = Vocab(word_alpha, tag_alpha, etype_alpha, gold_action_alpha, action_label_alpha, relation_alpha, nuclear_alpha, nuclear_relation_alpha)
    
    network = MainArchitecture(vocab, config) 

    if config.use_gpu:
        network.load_state_dict(torch.load(config.model_name))
        network = network.cuda()
    else:
        network.load_state_dict(torch.load(config.model_name, map_location=torch.device('cpu')))

    network.eval()
    
    logger.info('Reading dev instance, and predict...')
    reader = Reader(config.dev_path, config.dev_syn_feat_path)
    dev_instances  = reader.read_data()
    predict(network, dev_instances, vocab, config, logger)

    logger.info('Reading test instance, and predict...')
    reader = Reader(config.test_path, config.test_syn_feat_path)
    test_instances  = reader.read_data()
    predict(network, test_instances, vocab, config, logger)

if __name__ == '__main__':
    main()
