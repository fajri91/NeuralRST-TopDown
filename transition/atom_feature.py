class CNode(object):
    def __init__(self):
        self.nuclear = ''
        self.label = ''
        self.edu_start = -1
        self.edu_end = -1
        self.is_validate = False

    def clear(self):
        self.nuclear = ''
        self.label = ''
        self.edu_start = -1
        self.edu_end = -1
        self.is_validate = False

class AtomFeat:
    def __init__(self):
        self.s0 = CNode()
        self.s1 = CNode()
        self.s2 = CNode()
        self.q0 = CNode()

    def getFeat(self):
        return self.s0, self.s1, self.s2, self.q0
