from transition.atom_feature import CNode, AtomFeat
from transition.action import CAction
from in_out.instance import Instance
from in_out.instance import SubTree
from in_out.instance import CResult

import copy
import numpy as np

NUCLEAR = 'NUCLEAR'
SATELLITE = 'SATELLITE'
SPAN = 'span'
MAX_LENGTH= 512
class CState(object):
    def __init__(self):
        self.stack = [CNode() for i in range(MAX_LENGTH)] #list of CNode
        self.stack_size = 0 #int
        self.edu_size = 0 #int
        self.next_index = 0 #int
        self.pre_state = None #CState
        self.pre_action = CAction('', '', '') #CAction
        self.is_start = True
        self.atom_feat = AtomFeat() #AtomFeat

    def clear(self):
        self.stack_size = 0 #int
        self.edu_size = 0 #int
        self.next_index = 0 #int
        self.pre_state = None #CState
        self.pre_action = CAction('', '', '') #CAction
        self.is_start = True
        self.atom_feat = AtomFeat() #AtomFeat

    def ready(self, edu_size):
        self.edu_size = edu_size

    def is_end(self):
        if (self.pre_action.is_finish()):
            return True
        else:
            return False

    def copy_state(self, cstate):
        cstate.stack = copy.deepcopy(self.stack)
        cstate.edu_size = self.edu_size
        cstate.pre_state = self
    
    def done_mark(self):
        self.stack[self.stack_size].clear()

    def shift(self, cstate):
        cstate.stack_size = self.stack_size + 1
        cstate.next_index = self.next_index + 1
        self.copy_state(cstate)
        top = cstate.stack[cstate.stack_size - 1]
        top.clear()
        top.is_validate = True
        top.edu_start  = self.next_index
        top.edu_end = self.next_index
        
        cstate.pre_action.set('SH', '', '')
        cstate.done_mark()

    def reduce(self, cstate, nuclear, label):
        cstate.stack_size = self.stack_size - 1
        cstate.next_index = self.next_index
        self.copy_state(cstate)
        top0 = cstate.stack[self.stack_size - 1]
        top1 = cstate.stack[self.stack_size - 2]
        try:
            assert(top0.edu_start == top1.edu_end + 1)
            assert(top0.is_validate and top1.is_validate)
        except:
            import ipdb; ipdb.set_trace()
        top1.edu_end = top0.edu_end
        top1.nuclear = nuclear
        top1.label = label
        top0.clear()
        
        cstate.stack[self.stack_size - 1] = top0
        cstate.stack[self.stack_size - 2] = top1
        
        cstate.pre_action.set('RD', nuclear, label)
        cstate.done_mark()

    def pop_root(self, cstate):
        assert  self.stack_size == 1 and self.next_index == self.edu_size
        cstate.stack_size = 0
        cstate.next_index = self.edu_size
        self.copy_state(cstate)
        top0 = cstate.stack[self.stack_size - 1]
        # assert(top0.edu_start == 0 and top0.edu_end + 1 == self.edu_size)
        assert(top0.edu_start == 0)
        assert(top0.is_validate)
        top0.clear()
        
        cstate.stack[self.stack_size - 1] = top0
        cstate.pre_action.set('PR', '', '')
        cstate.done_mark()

    #cstate = CState
    #ac = CAction
    def move(self, cstate, ac):
        cstate.is_start = False
        if ac.is_shift():
            self.shift(cstate)
        elif ac.is_reduce():
            self.reduce(cstate, ac.nuclear, ac.label)
        elif ac.is_finish():
            self.pop_root(cstate)
        else:
            raise Exception('Error Action!')
        return cstate

    def get_result(self):
        result = CResult()
        state = self
        while(not state.pre_state.is_start):
            ac = state.pre_action
            st = state.pre_state
            if (ac.is_reduce()):
                assert(st.stack_size >= 2)
                right_node = st.stack[st.stack_size-1]
                left_node = st.stack[st.stack_size-2]
                left_subtree = SubTree()
                right_subtree = SubTree()

                left_subtree.edu_start = left_node.edu_start
                left_subtree.edu_end = left_node.edu_end

                right_subtree.edu_start = right_node.edu_start
                right_subtree.edu_end = right_node.edu_end

                if ac.nuclear == 'NN':
                    left_subtree.nuclear = NUCLEAR
                    right_subtree.nuclear = NUCLEAR
                    left_subtree.relation = ac.label
                    right_subtree.relation = ac.label
                elif ac.nuclear == 'SN':
                    left_subtree.nuclear = SATELLITE
                    right_subtree.nuclear = NUCLEAR
                    left_subtree.relation = ac.label
                    right_subtree.relation = SPAN
                elif ac.nuclear == 'NS':
                    left_subtree.nuclear = NUCLEAR
                    right_subtree.nuclear =SATELLITE
                    left_subtree.relation = SPAN
                    right_subtree.relation = ac.label
                
                result.subtrees.insert(0, right_subtree)
                result.subtrees.insert(0, left_subtree)
            state = state.pre_state
        return result
    
    def allow_shift(self):
        if self.next_index == self.edu_size:
            return False
        return True
    
    def allow_reduce(self):
        if self.stack_size >= 2:
            return True
        return False

    def allow_pop_root(self):
        if self.next_index == self.edu_size and self.stack_size == 1:
            return True
        return False

    def get_candidate_actions(self, vocab):
        mask = np.array([False] * vocab.gold_action_alpha.size())
        if self.allow_reduce():
            mask = mask | vocab.mask_reduce
        if self.is_end():
            mask = mask | vocab.mask_no_action
        if self.allow_shift():
            mask = mask | vocab.mask_shift
        if self.allow_pop_root():
            mask = mask | vocab.mask_pop_root
        return ~mask

    def prepare_index(self):
        if self.stack_size > 0:
            self.atom_feat.s0 = self.stack[self.stack_size - 1]
        else:
            self.atom_feat.s0 = None
        if self.stack_size > 1:
            self.atom_feat.s1 = self.stack[self.stack_size - 2]
        else:
            self.atom_feat.s1 = None
        if self.stack_size > 2:
            self.atom_feat.s2 = self.stack[self.stack_size - 3]
        else:
            self.atom_feat.s2 = None
        if self.next_index >= 0 and self.next_index < self.edu_size:
            self.atom_feat.q0 = self.next_index
        else:
            self.atom_feat.q0 = None
        return self.atom_feat







