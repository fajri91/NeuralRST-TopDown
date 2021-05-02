import os
import torch

class Config(object):
    def __init__(self, args):
        if args is None:
            return
        self.use_gpu = torch.cuda.is_available()
        self.use_dynamic_oracle = args.use_dynamic_oracle == 1
        self.flag_oracle = False
        self.word_embedding = args.word_embedding
        self.word_embedding_file = args.word_embedding_file
    
        self.train_path = args.train
        self.test_path = args.test
        self.dev_path = args.dev
        self.train_syn_feat_path = args.train_syn_feat
        self.test_syn_feat_path = args.test_syn_feat
        self.dev_syn_feat_path = args.dev_syn_feat
        self.model_path = args.model_path +'/'+ args.experiment
        self.model_name = args.model_name
        self.alphabet_path = os.path.join(self.model_path, 'alphabets/')

        self.max_iter = args.max_iter
        self.word_dim = args.word_dim
        self.tag_dim = args.tag_dim
        self.etype_dim = args.etype_dim
        self.syntax_dim = args.syntax_dim
        self.max_sent_size = args.max_sent_size
        self.max_edu_size = args.max_edu_size
        self.max_state_size = args.max_state_size
        self.hidden_size = args.hidden_size
        self.hidden_size_tagger = args.hidden_size_tagger
        
        self.freeze = args.freeze
        self.drop_prob = args.drop_prob
        self.num_layers = args.num_layers

        self.batch_size = args.batch_size
        self.opt = args.opt
        self.lr = args.lr
        self.ada_eps = args.ada_eps
        self.momentum = 0.9
        self.beta1 = args.beta1
        self.beta2 = args.beta2 
        self.betas = (self.beta1, self.beta2)
        self.gamma = args.gamma
        self.start_decay = args.start_decay
        self.grad_accum = args.grad_accum

        self.clip = args.clip

        self.loss_nuc_rel = args.loss_nuc_rel
        self.loss_seg = args.loss_seg
        self.activate_nuc_rel_loss = args.activate_nuc_rel_loss
        self.activate_seg_loss = args.activate_seg_loss

        self.decay = args.decay
        self.oracle_prob = args.oracle_prob
        self.start_dynamic_oracle = args.start_dynamic_oracle
        self.early_stopping = args.early_stopping
    
        self.beam_search = args.beam_search
        self.depth_alpha = args.depth_alpha
        self.elem_alpha = args.elem_alpha
        self.seed = args.seed

    def save(self):
        f = open(self.model_path + '/config.cfg', 'w')
        f.write("use_gpu = " + str(self.use_gpu) + '\n')
        f.write("use_dynamic_oracle = "+ str(self.use_dynamic_oracle) + '\n')
        f.write("flag_oracle = " + str(self.flag_oracle) + '\n')
        f.write("word_embedding = " + str(self.word_embedding) + '\n')
        f.write("word_embedding_file = " + str(self.word_embedding_file) + '\n')
    
        f.write("train_path = " + str(self.train_path) + '\n')
        f.write("test_path = " + str(self.test_path) + '\n')
        f.write("dev_path = " + str(self.dev_path) + '\n')
        f.write("train_syn_feat_path = " + str(self.train_syn_feat_path) + '\n')
        f.write("test_syn_feat_path = " + str(self.test_syn_feat_path) + '\n')
        f.write("dev_syn_feat_path = " + str(self.dev_syn_feat_path) + '\n')
        f.write("model_path = " + str(self.model_path) + '\n')
        f.write("model_name = " + str(self.model_name) + '\n')
        f.write("alphabet_path = " + str(self.alphabet_path) + '\n')

        f.write("max_iter = " + str(self.max_iter) + '\n')
        f.write("word_dim = " + str(self.word_dim) + '\n')
        f.write("tag_dim = " + str(self.tag_dim) + '\n')
        f.write("etype_dim = " + str(self.etype_dim) + '\n')
        f.write("syntax_dim = " + str(self.syntax_dim) + '\n')
        f.write("max_sent_size = " + str(self.max_sent_size) + '\n')
        f.write("max_edu_size = " + str(self.max_edu_size) + '\n')
        f.write("max_state_size = " + str(self.max_state_size) + '\n')
        f.write("hidden_size = " + str(self.hidden_size) + '\n')
        f.write("hidden_size_tagger = " + str(self.hidden_size_tagger) + '\n')
        
        f.write("freeze = " + str(self.freeze) + '\n')
        f.write("drop_prob = " + str(self.drop_prob) + '\n')
        f.write("num_layers = " + str(self.num_layers) + '\n')

        f.write("batch_size = " + str(self.batch_size) + '\n')
        f.write("opt = " + str(self.opt) + '\n')
        f.write("lr = " + str(self.lr) + '\n')
        f.write("ada_eps = " + str(self.ada_eps) + '\n')
        f.write("momentum = " + str(self.momentum) + '\n')
        f.write("beta1 = " + str(self.beta1) + '\n')
        f.write("beta2 = " + str(self.beta2) + '\n')
        f.write("gamma = " + str(self.gamma) + '\n')
        f.write("start_decay = " + str(self.start_decay) + '\n')
        f.write("grad_accum = " + str(self.grad_accum) + '\n')

        f.write("clip = " + str(self.clip) + '\n')
        
        f.write("loss_nuc_rel = " + str(self.loss_nuc_rel) + '\n')
        f.write("loss_seg = " + str(self.loss_seg) + '\n')
        f.write("activate_nuc_rel_loss = " + str(self.activate_nuc_rel_loss) + '\n')
        f.write("activate_seg_loss = " + str(self.activate_seg_loss) + '\n')

        f.write("decay = " + str(self.decay) + '\n')
        f.write("oracle_prob = " + str(self.oracle_prob) + '\n')
        f.write("start_dynamic_oracle = " + str(self.start_dynamic_oracle) + '\n')
        f.write("early_stopping = " + str(self.early_stopping) + '\n')
        
        f.write("beam_search = " + str(self.beam_search) + '\n')
        f.write("depth_alpha = " + str(self.depth_alpha) + '\n')
        f.write("elem_alpha = " + str(self.elem_alpha) + '\n')
        f.write("seed = " + str(self.seed) + '\n')
        f.close()

    def load_config(self, path):
        f = open(path, 'r')
        self.use_gpu = f.readline().strip().split(' = ')[-1] == 'True'
        self.use_dynamic_oracle = f.readline().strip().split(' = ')[-1] == 'True'
        self.flag_oracle = f.readline().strip().split(' = ')[-1] == 'True'
        self.word_embedding = f.readline().strip().split(' = ')[-1] 
        self.word_embedding_file = f.readline().strip().split(' = ')[-1] 
    
        self.train_path = f.readline().strip().split(' = ')[-1] 
        self.test_path = f.readline().strip().split(' = ')[-1] 
        self.dev_path = f.readline().strip().split(' = ')[-1] 
        self.train_syn_feat_path = f.readline().strip().split(' = ')[-1] 
        self.test_syn_feat_path = f.readline().strip().split(' = ')[-1] 
        self.dev_syn_feat_path = f.readline().strip().split(' = ')[-1] 
        self.model_path = f.readline().strip().split(' = ')[-1] 
        self.model_name = f.readline().strip().split(' = ')[-1] 
        self.alphabet_path = f.readline().strip().split(' = ')[-1] 

        self.max_iter = int(f.readline().strip().split(' = ')[-1])
        self.word_dim = int(f.readline().strip().split(' = ')[-1])
        self.tag_dim = int(f.readline().strip().split(' = ')[-1])
        self.etype_dim = int(f.readline().strip().split(' = ')[-1])
        self.syntax_dim = int(f.readline().strip().split(' = ')[-1])
        self.max_sent_size = int(f.readline().strip().split(' = ')[-1])
        self.max_edu_size = int(f.readline().strip().split(' = ')[-1])
        self.max_state_size = int(f.readline().strip().split(' = ')[-1])
        self.hidden_size = int(f.readline().strip().split(' = ')[-1])
        self.hidden_size_tagger = int(f.readline().strip().split(' = ')[-1])
        
        self.freeze = f.readline().strip().split(' = ')[-1] == 'True'
        self.drop_prob = float(f.readline().strip().split(' = ')[-1])
        self.num_layers = int(f.readline().strip().split(' = ')[-1])

        self.batch_size = int(f.readline().strip().split(' = ')[-1])
        self.opt = f.readline().strip().split(' = ')[-1] 
        self.lr = float(f.readline().strip().split(' = ')[-1])
        self.ada_eps = float(f.readline().strip().split(' = ')[-1])
        self.momentum = float(f.readline().strip().split(' = ')[-1])
        self.beta1 = float(f.readline().strip().split(' = ')[-1])
        self.beta2 = float(f.readline().strip().split(' = ')[-1])
        self.betas = (self.beta1, self.beta2)
        self.gamma = float(f.readline().strip().split(' = ')[-1])
        self.start_decay = int(f.readline().strip().split(' = ')[-1])
        self.grad_accum = int(f.readline().strip().split(' = ')[-1])

        self.clip = float(f.readline().strip().split(' = ')[-1])
        
        self.loss_nuc_rel = float(f.readline().strip().split(' = ')[-1])
        self.loss_seg = float(f.readline().strip().split(' = ')[-1])
        self.activate_nuc_rel_loss = int(f.readline().strip().split(' = ')[-1])
        self.activate_seg_loss = int(f.readline().strip().split(' = ')[-1])

        self.decay = int(f.readline().strip().split(' = ')[-1])
        self.oracle_prob = float(f.readline().strip().split(' = ')[-1])
        self.start_dynamic_oracle = int(f.readline().strip().split(' = ')[-1])
        self.early_stopping = int(f.readline().strip().split(' = ')[-1])
        
        self.beam_search = int(f.readline().strip().split(' = ')[-1])
        self.depth_alpha = float(f.readline().strip().split(' = ')[-1])
        self.elem_alpha = float(f.readline().strip().split(' = ')[-1])
        self.seed = int(f.readline().strip().split(' = ')[-1])
        f.close()
