class Tree(object):
    def __init__(self, edu_span, nuclear, relation):
        self.edu_span = edu_span
        self.nuclear = nuclear
        self.relation = relation
        self.left = None
        self.right = None

    def str(self):
        return self.nuclear + ' ' + self.relation

    def get_id(self, vocab):
        tmp = self.nuclear.split(' ')
        action_str = "REDUCE_" + tmp[0][0] + tmp[1][0] + '_' + self.relation
        return vocab.relation_alpha.word2id(action_str)

    def get_nodes(self, nodes, vocab):
        cur_id = self.get_id (vocab)
        if self.left is not None:
            left_id = self.left.get_id(vocab)
            key = (cur_id, left_id)
            nodes[key] = nodes.get(key, 0) + 1
            nodes = self.left.get_nodes(nodes, vocab)

        if self.right is not None:
            right_id = self.right.get_id(vocab)
            key = (cur_id, right_id)
            nodes[key] = nodes.get(key, 0) + 1
            nodes = self.right.get_nodes(nodes, vocab)
        return nodes
