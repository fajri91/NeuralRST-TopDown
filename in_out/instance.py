from in_out.tree import Tree
from models.metric import Metric
import math

# This class represents one document
class Instance(object):
    def __init__(self, total_words, total_tags, edus, gold_actions, result):
        self.total_words = total_words
        self.total_tags = total_tags
        self.edus = edus
        self.gold_actions = gold_actions
        self.result = result
        self.tree = self.obtain_tree(result)
        self.gold_top_down = Gold(self.tree)

    def evaluate(self, other_result, span, nuclear, relation, full):
        main_subtrees = self.result.subtrees
        span.overall_label_count += len(main_subtrees)
        span.predicated_label_count += len(other_result.subtrees)
        for i in range (len(other_result.subtrees)):
            for j in range (len(main_subtrees)):
                if other_result.subtrees[i].span_equal(main_subtrees[j]):
                    span.correct_label_count += 1
                    break
        
        nuclear.overall_label_count += len(main_subtrees)
        nuclear.predicated_label_count += len(other_result.subtrees)
        for i in range (len(other_result.subtrees)):
            for j in range (len(main_subtrees)):
                if other_result.subtrees[i].nuclear_equal(main_subtrees[j]):
                    nuclear.correct_label_count += 1
                    break

        relation.overall_label_count += len(main_subtrees)
        relation.predicated_label_count += len(other_result.subtrees)
        for i in range (len(other_result.subtrees)):
            for j in range (len(main_subtrees)):
                if other_result.subtrees[i].relation_equal(main_subtrees[j]):
                    relation.correct_label_count += 1
                    break

        full.overall_label_count += len(main_subtrees)
        full.predicated_label_count += len(other_result.subtrees)
        for i in range (len(other_result.subtrees)):
            for j in range (len(main_subtrees)):
                if other_result.subtrees[i].full_equal(main_subtrees[j]):
                    full.correct_label_count += 1
                    break
        return span, nuclear, relation, full 

    def evaluate_original_parseval(self, pred, span, nuclear, relation, full):
        gold = self.gold_top_down
        
        span.overall_label_count += len(gold.edu_span)
        span.predicated_label_count += len(pred.edu_span)
        
        nuclear.overall_label_count += len(gold.edu_span)
        nuclear.predicated_label_count += len(pred.edu_span)
        
        relation.overall_label_count += len(gold.edu_span)
        relation.predicated_label_count += len(pred.edu_span)
        
        full.overall_label_count += len(gold.edu_span)
        full.predicated_label_count += len(pred.edu_span)
        for i in range (len(pred.edu_span)):
            for j in range (len(gold.edu_span)):
                if pred.edu_span[i] == gold.edu_span[j]: # and pred.segmentation[i] == gold.segmentation[j]:
                    span.correct_label_count += 1
                    if pred.nuclear[i] == gold.nuclear[j]:
                        nuclear.correct_label_count += 1
                    if pred.relation[i] == gold.relation[j]:
                        relation.correct_label_count += 1
                    if pred.nuclear[i] == gold.nuclear[j] and pred.relation[i] == gold.relation[j]:
                        full.correct_label_count += 1
                    break
        return span, nuclear, relation, full 
    
    def check_top_down(self):
        gold = self.gold_top_down
        assert (len(gold.edu_span) == len(gold.segmentation) == len(gold.nuclear) == len(gold.relation))
        result = gold.get_subtree()
        span = Metric(); nuclear = Metric(); relation = Metric(); full = Metric();
        span, nuclear, relation, full = self.evaluate(result, span, nuclear, relation, full)
        assert span.bIdentical() and nuclear.bIdentical() and relation.bIdentical() and full.bIdentical()

    def obtain_tree(self, result):
        p_subtree = {}
        subtrees = self.result.subtrees
        assert len(subtrees) % 2 == 0
        for idx in range(0, len(subtrees), 2):
            edu_span = (subtrees[idx].edu_start, subtrees[idx+1].edu_end)
            nuclear = subtrees[idx].nuclear + " " + subtrees[idx+1].nuclear
            relation = subtrees[idx].relation
            if relation == 'span':
                relation = subtrees[idx+1].relation
            tree = Tree(edu_span, nuclear, relation)
            
            #set child:
            if p_subtree.get(edu_span[0], None) is not None:
                tree.left = p_subtree[edu_span[0]]
            else:
                tree.left = Tree((edu_span[0], edu_span[0]), "", "")
            if p_subtree.get(edu_span[1], None) is not None:
                tree.right = p_subtree[edu_span[1]]
            else:
                tree.right = Tree((edu_span[1], edu_span[1]), "", "")

            p_subtree[edu_span[0]] = tree
            p_subtree[edu_span[1]] = tree
        if len(subtrees) != 0:
            return p_subtree[0]
        else:
            return None


# Gold represents Top-Down Discourse Parser attributes
class Gold(object):
    def __init__(self, tree):
        self.edu_span = []
        self.segmentation = []
        self.nuclear = []
        self.relation = []
        self.nuclear_relation = []
        self.depth = []
        queue = []
        queue.append((tree,1))
        if tree is None:
            return
        while(len(queue) > 0):
            cur_tree, cur_depth = queue.pop(0)
            if cur_tree.left is None:
                continue
            span = cur_tree.edu_span
            span_left = cur_tree.left.edu_span
            self.edu_span.append(span)
            self.depth.append(cur_depth)
            self.segmentation.append(span_left[1])
            self.nuclear.append(cur_tree.nuclear)
            self.relation.append(cur_tree.relation)
            self.nuclear_relation.append(cur_tree.nuclear + ' - ' + cur_tree.relation)
            if cur_tree.left is not None:
                queue.append((cur_tree.left, cur_depth+1))
            if cur_tree.right is not None:
                queue.append((cur_tree.right, cur_depth+1))
    
    def get_span_cut_label(self):
        labels = []
        for idx in range(len(self.edu_span)):
            label = []
            for idy in range(self.edu_span[idx][0], self.edu_span[idx][1]+1):
                if self.segmentation[idx] == idy:
                    label.append(1)
                else:
                    label.append(0)
            labels.append(label)
        return labels

    def get_subtree(self):
        subtrees = []
        for i in range(len(self.edu_span)):
            edu_span = self.edu_span[i]
            n1, n2 = self.nuclear[i].split(' ')
            if n1 == 'SATELLITE' and n2 == 'NUCLEAR':
                r1 = self.relation[i]
                r2 = 'span'
            elif n1 == 'NUCLEAR' and n2 == 'SATELLITE':
                r1 = 'span'
                r2 = self.relation[i]
            else:
                assert n1 == 'NUCLEAR' and n2 == 'NUCLEAR'
                r1 = r2 = self.relation[i]
            cut = self.segmentation[i]
            span1 = (edu_span[0], cut)
            span2 = (cut+1, edu_span[1])
            left = SubTree(); left.set(n1, r1, span1[0], span1[1])
            right = SubTree(); right.set(n2, r2, span2[0], span2[1])
            subtrees.append(left)
            subtrees.append(right)
        result = CResult()
        result.subtrees = subtrees
        return result

    def set_attribute(self, span, segment, nuclear, relation):
        self.edu_span = span
        self.segmentation = segment
        self.nuclear = nuclear
        self.relation = relation


# A single EDU representation
class EDU(object):
    def __init__(self, start_index, end_index):
        self.start_index = start_index # int
        self.end_index = end_index # int
        self.etype = '' # string
        self.words = [] # list of word (string)
        self.tags = [] # list of tag (string)
        self.syntax_features = []


# A single subtree representation
class SubTree(object):
    NUCLEAR='NUCLEAR'
    SATELLITE='SATELLITE'
    SPAN='span'

    def __init__(self):
        self.nuclear = ''
        self.relation = ''
        self.edu_start = -1
        self.edu_end = -1

    def clear(self):
        self.nuclear = ''
        self.relation = ''
        self.edu_start = -1
        self.edu_end = -1

    def set(self, nuclear, relation, edu_start, edu_end):
        self.nuclear = nuclear
        self.relation = relation
        self.edu_start = edu_start
        self.edu_end = edu_end

    def span_equal(self, tree):
        return self.edu_start == tree.edu_start and self.edu_end == tree.edu_end
    
    def nuclear_equal(self, tree):
        return self.edu_start == tree.edu_start and self.edu_end == tree.edu_end and self.nuclear == tree.nuclear

    def relation_equal(self, tree):
        return self.edu_start == tree.edu_start and self.edu_end == tree.edu_end and self.relation == tree.relation

    def full_equal(self, tree):
        return self.edu_start == tree.edu_start and self.edu_end == tree.edu_end and self.relation == tree.relation and self.nuclear == tree.nuclear

    def get_str(self):
        return self.nuclear +' '+self.relation+' edu('+str(self.edu_start)+'-'+str(self.edu_end) +')'


# List of subtrees
class CResult(object):
    def __init__(self):
        self.subtrees = []
    def clear(self):
        self.subtrees = []


# Syntax feature of one word
class SynFeat(object):
    def __init__(self, arc_dep, arc_head, rel_dep, rel_head):
        self.arc_dep = arc_dep
        self.arc_head = arc_head
        self.rel_dep = rel_dep
        self.rel_head = rel_head

    def concat(self):
        return self.arc_dep + self.rel_dep + self.arc_head + self.rel_head

