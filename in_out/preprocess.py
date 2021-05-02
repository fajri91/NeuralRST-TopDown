import os
import numpy as np
import torch
import numpy as np
import copy

from models.metric import Metric
from models.alphabet import Alphabet
from in_out.util import lower_with_digit_transform
from transition.state import CState
from torch.autograd import Variable

def construct_embedding_table(alpha, hidden_size, freeze, pretrained_embed = None):
    if alpha is None:
        return None
    scale = np.sqrt(6.0 / (alpha.size()+hidden_size))
    table = np.empty([alpha.size(), hidden_size], dtype=np.float32)
    for word, index, in alpha.alpha2id.items():
        if pretrained_embed is not None:
            if word in pretrained_embed:
                embedding = pretrained_embed[word]
            elif word.lower() in pretrained_embed:
                embedding = pretrained_embed[word.lower()]
            else:
                embedding = np.zeros([1, hidden_size]).astype(np.float32) if freeze else np.random.uniform(-scale, scale, [1, hidden_size]).astype(np.float32)
        else:
            embedding = np.random.uniform(-scale, scale, [1, hidden_size]).astype(np.float32)
        table[index, :] = embedding
    return torch.from_numpy(table)


def create_alphabet(instances, alphabet_directory, logger):
    word_size = 0
    gold_size = 0
        
    word_stat = {}
    tag_stat = {}
    gold_action_stat = {}
    action_label_stat = {}
    etype_stat = {}
    relation_stat = {}
    nuclear_stat = {}
    nuclear_relation_stat = {}

    if not os.path.isdir(alphabet_directory):
        print("Creating Alphabets")
        for instance in instances:
            for i in range(len(instance.total_words)):
                word = lower_with_digit_transform(instance.total_words[i].strip())
                tag = instance.total_tags[i]
                word_stat[word] = word_stat.get(word, 0) + 1
                tag_stat[tag] = tag_stat.get(tag, 0) + 1

            for action in instance.gold_actions:
                if (not action.is_shift() and not action.is_finish()):
                    action_label_stat[action.label] = action_label_stat.get(action.label, 0) + 1
                gold_action_stat[action.get_str()] = gold_action_stat.get(action.get_str(), 0) + 1
            
            for k in range(len(instance.edus)):
                etype_stat[instance.edus[k].etype] = etype_stat.get(instance.edus[k].etype, 0) + 1
        
            for k in range(len(instance.gold_top_down.relation)):
                g = instance.gold_top_down
                relation_stat[g.relation[k]] = relation_stat.get(g.relation[k], 0) + 1
                nuclear_stat[g.nuclear[k]] = nuclear_stat.get(g.nuclear[k], 0) + 1
                key = g.nuclear[k] + ' - ' + g.relation[k]
                nuclear_relation_stat[key] = nuclear_relation_stat.get(key, 0) + 1

        word_alpha = Alphabet(word_stat, 'word_alpha')
        tag_alpha = Alphabet(tag_stat, 'tag_alpha')
        gold_action_alpha = Alphabet(gold_action_stat, 'gold_action_alpha', for_label_index=True)
        action_label_alpha = Alphabet(action_label_stat, 'action_label_alpha', for_label_index=True)
        relation_alpha = Alphabet(relation_stat, 'relation_alpha', for_label_index=True)
        nuclear_alpha = Alphabet(nuclear_stat, 'nuclear_alpha', for_label_index=True)
        nuclear_relation_alpha = Alphabet(nuclear_relation_stat, 'nuclear_relation_alpha', for_label_index=True)
        etype_alpha = Alphabet(etype_stat, 'etype_alpha')

        word_alpha.save(alphabet_directory)
        tag_alpha.save(alphabet_directory)
        gold_action_alpha.save(alphabet_directory)
        action_label_alpha.save(alphabet_directory)
        relation_alpha.save(alphabet_directory)
        nuclear_alpha.save(alphabet_directory)
        nuclear_relation_alpha.save(alphabet_directory)
        etype_alpha.save(alphabet_directory)
    else:
        logger.info("The path exist, loading Alphabets for experiment directory")
        word_alpha = Alphabet(word_stat, 'word_alpha')
        tag_alpha = Alphabet(tag_stat, 'tag_alpha')
        gold_action_alpha = Alphabet(gold_action_stat, 'gold_action_alpha')
        action_label_alpha = Alphabet(action_label_stat, 'action_label_alpha')
        relation_alpha = Alphabet(relation_stat, 'relation_alpha')
        nuclear_alpha = Alphabet(nuclear_stat, 'nuclear_alpha')
        nuclear_relation_alpha = Alphabet(nuclear_relation_stat, 'nuclear_relation_alpha')
        etype_alpha = Alphabet(etype_stat, 'etype_alpha')
        
        word_alpha.load(alphabet_directory)
        tag_alpha.load(alphabet_directory)
        gold_action_alpha.load(alphabet_directory, for_label_index=True)
        action_label_alpha.load(alphabet_directory, for_label_index=True)
        relation_alpha.load(alphabet_directory, for_label_index=True)
        nuclear_alpha.load(alphabet_directory, for_label_index=True)
        nuclear_relation_alpha.load(alphabet_directory, for_label_index=True)
        etype_alpha.load(alphabet_directory)

    logger.info("Word alphabet size: " + str(word_alpha.size()))
    logger.info("Tag alphabet size: " + str(tag_alpha.size()))
    logger.info("Gold action alphabet size: " + str(gold_action_alpha.size()))
    logger.info("Action Label alphabet size: " + str(action_label_alpha.size()))
    logger.info("Relation alphabet size: " + str(relation_alpha.size()))
    logger.info("Nuclear alphabet size: " + str(nuclear_alpha.size()))
    logger.info("Nuclear - Relation alphabet size: " + str(nuclear_relation_alpha.size()))
    logger.info("Etype alphabet size: " + str(etype_alpha.size()))
    return word_alpha, tag_alpha, gold_action_alpha, action_label_alpha, relation_alpha, nuclear_alpha, nuclear_relation_alpha, etype_alpha


def validate_gold_actions(instances, maxStateSize):
    shift_num = 0; reduce_nn_num = 0; reduce_ns_num = 0; reduce_sn_num = 0
    span = Metric(); nuclear = Metric(); relation = Metric(); full = Metric()

    for inst in instances:
        for ac in inst.gold_actions:
            if ac.is_shift():
                shift_num+=1
            if ac.is_reduce():
                if ac.nuclear == 'NN':
                    reduce_nn_num += 1
                elif ac.nuclear == 'NS':
                    reduce_ns_num += 1
                elif ac.nuclear == 'SN':
                    reduce_sn_num += 1
                else:
                    raise Exception('Reduce error, this must have nuclearity')
                # something is here
                assert(ac.label_id != -1)

    print("Reduce NN: " + str(reduce_nn_num))
    print("Reduce NS: " + str(reduce_ns_num))
    print("Reduce SN: " + str(reduce_sn_num))
    print("Shift: " + str(shift_num))

    print("Checking the gold Actions, it will be interrupted if there is error assertion, takes a while, you can comment this part next time")
    all_states = [CState() for i in range(maxStateSize)]
    for inst in instances:
        step = 0
        gold_actions = inst.gold_actions
        action_size = len(gold_actions)
        all_states[0].ready(len(inst.edus))
        while(not all_states[step].is_end()):
            assert(step < action_size)
            all_states[step+1] = all_states[step].move(all_states[step+1], gold_actions[step])
            step += 1
        assert(step == action_size)
        result = all_states[step].get_result()
        span, nuclear, relation, full = inst.evaluate(result, span, nuclear, relation, full)
        if not span.bIdentical() or not nuclear.bIdentical() or not relation.bIdentical() or not full.bIdentical():
            raise Exception('Error state conversion!! ')


def validate_gold_top_down(instances):
    try:
        for instance in instances:
            instance.check_top_down()
    except:
        raise Exception('Error conversion from top-down discourse labels!! ')


def batch_data_variable(data, indices, vocab, config, is_training=True):
    batch_size  = len(indices)
    indices = indices.tolist()
    batch = data[indices]
    max_edu_len = -1
    max_edu_num = -1
    for data in batch:
        edu_num = len(data.edus)
        if edu_num > max_edu_num: max_edu_num = edu_num
        for edu in data.edus:
            edu_len = len(edu.words)
            if edu_len > max_edu_len: max_edu_len = edu_len
    if max_edu_len > config.max_sent_size: max_edu_len = config.max_sent_size
    
    edu_words = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    edu_types = Variable(torch.LongTensor(batch_size, max_edu_num).zero_(), requires_grad=False)
    edu_syntax = np.zeros([batch_size, max_edu_num, max_edu_len, config.syntax_dim], dtype=np.float32)
     
    word_mask = Variable(torch.Tensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    edu_tags = Variable(torch.LongTensor(batch_size, max_edu_num, max_edu_len).zero_(), requires_grad=False)
    edu_mask = Variable(torch.Tensor(batch_size, max_edu_num).zero_(), requires_grad=False)
    word_denominator = Variable(torch.ones(batch_size, max_edu_num).type(torch.FloatTensor) * -1, requires_grad=False)
    len_edus = np.zeros([batch_size], dtype=np.int64)
    gold_segmentation = Variable(torch.Tensor(batch_size, max_edu_num-1, max_edu_num).zero_(), requires_grad=False)
    gold_nuclear = Variable(torch.ones(batch_size, max_edu_num-1).type(torch.LongTensor) * vocab.nuclear_alpha.size(), requires_grad=False)
    gold_relation = Variable(torch.ones(batch_size, max_edu_num-1).type(torch.LongTensor) * vocab.relation_alpha.size(), requires_grad=False)
    gold_nuclear_relation = Variable(torch.ones(batch_size, max_edu_num-1).type(torch.LongTensor) * vocab.nuclear_relation_alpha.size(), requires_grad=False)
    gold_span = []
    gold_depth = []

    len_golds = np.zeros([batch_size], dtype=np.int64)

    for idx in range(batch_size):
        for idy in range(len(batch[idx].edus)):
            len_edus[idx] = len(batch[idx].edus)
            edu = batch[idx].edus[idy]
            edu_mask[idx, idy] = 1
            edu_types[idx, idy] = vocab.etype_alpha.word2id(edu.etype)
            edu_len = min(len(edu.words), max_edu_len)
            word_denominator[idx, idy] = edu_len
            for idz in range(edu_len):
                word = edu.words[idz]
                tag = edu.tags[idz]
                edu_words[idx, idy, idz] = vocab.word_alpha.word2id(word)
                edu_tags[idx, idy, idz] = vocab.tag_alpha.word2id(tag)
                edu_syntax[idx, idy, idz] = edu.syntax_features[idz]
                word_mask[idx, idy, idz] = 1
    
        if is_training:
            for idy in range(len(batch[idx].gold_top_down.edu_span)):
                cut_index = batch[idx].gold_top_down.segmentation[idy]
                gold_nuclear[idx, idy] = vocab.nuclear_alpha.word2id(batch[idx].gold_top_down.nuclear[idy])
                gold_relation[idx, idy] = vocab.relation_alpha.word2id(batch[idx].gold_top_down.relation[idy])
                gold_nuclear_relation[idx, idy] = vocab.nuclear_relation_alpha.word2id(batch[idx].gold_top_down.nuclear_relation[idy])
                index_gold = cut_index - batch[idx].gold_top_down.edu_span[idy][0]
                gold_segmentation[idx, idy, index_gold] = 1
            gold_span.append(batch[idx].gold_top_down.edu_span)
            gold_depth.append(batch[idx].gold_top_down.depth)
            
            len_golds[idx] = len(batch[idx].gold_top_down.edu_span)

    span = copy.deepcopy(gold_span)
    depth = copy.deepcopy(gold_depth)
    edu_syntax = Variable(torch.from_numpy(edu_syntax), volatile=False, requires_grad=False)
    if config.use_gpu:
        edu_words = edu_words.cuda()
        edu_tags = edu_tags.cuda()
        edu_types = edu_types.cuda()
        edu_mask = edu_mask.cuda()
        word_mask = word_mask.cuda()
        word_denominator = word_denominator.cuda()
        edu_syntax = edu_syntax.cuda()
        gold_nuclear = gold_nuclear.cuda()
        gold_relation = gold_relation.cuda()
        gold_nuclear_relation = gold_nuclear_relation.cuda()
        gold_segmentation = gold_segmentation.cuda()

    return edu_words, edu_tags, edu_types, edu_mask, word_mask, len_edus, \
            word_denominator, edu_syntax, gold_nuclear, gold_relation, gold_nuclear_relation, gold_segmentation, \
            span, len_golds, depth

 
