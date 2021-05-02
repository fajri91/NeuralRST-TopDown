import numpy as np
from transition.action import CAction

class Vocab(object):
    def __init__(self, word_alpha, tag_alpha, etype_alpha, gold_action_alpha, action_label_alpha, relation_alpha, nuclear_alpha, nuclear_relation_alpha):
        self.word_alpha = word_alpha
        self.tag_alpha = tag_alpha
        self.etype_alpha = etype_alpha
        self.gold_action_alpha = gold_action_alpha
        self.action_label_alpha = action_label_alpha
        self.relation_alpha = relation_alpha
        self.nuclear_alpha = nuclear_alpha
        self.nuclear_relation_alpha = nuclear_relation_alpha

        self.id2action = {}
        for key in self.gold_action_alpha.id2alpha.keys():

            if key != self.gold_action_alpha.size():
                self.id2action[key] = self.get_action(key)
        
        self.mask_reduce = np.array([False] * self.gold_action_alpha.size())
        self.mask_no_action = np.array([False] * self.gold_action_alpha.size())
        self.mask_shift = np.array([False] * self.gold_action_alpha.size())
        self.mask_pop_root = np.array([False] * self.gold_action_alpha.size())
        for key in self.gold_action_alpha.id2alpha.keys():
            if 'SHIFT' in self.gold_action_alpha.id2alpha[key]:
                self.mask_shift[key] = True
            if 'REDUCE' in self.gold_action_alpha.id2alpha[key]:
                self.mask_reduce[key] = True
            if 'POPROOT' in self.gold_action_alpha.id2alpha[key]:
                self.mask_pop_root[key] = True
            if 'NOACTION' in self.gold_action_alpha.id2alpha[key]:
                self.mask_no_action[key] = True

    def get_action(self, id_selected_action):
        mapper = {'SHIFT': 'SH', 'REDUCE': 'RD', 'POPROOT': 'PR', 'NOACTION': ''}
        str_selected_action = self.gold_action_alpha.id2word(id_selected_action).split('_')
        selected_action = CAction(mapper[str_selected_action[0]],
                                  str_selected_action[1],
                                  str_selected_action[2])
        return selected_action
