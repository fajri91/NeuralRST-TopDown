
class CAction(object):
    """
        Four types of code:
         1. REDUCE = RD
         2. SHIFT = SH
         3. POP_ROOT = PR
         4. NO_ACTION = ''

        Label is the relation, eg: cause, elab, back, same, attr etc.
        There are 19 relations in our dataset
        - label is String
        - label_id is its id in integer

        Three types of Nuclear:
         1. NN
         2. NS
         3. SN
         4. ''
    """
    POP_ROOT='PR'
    REDUCE='RD'
    SHIFT='SH'
    NO_ACTION=''
    
    NN='NN'
    NS='NS'
    SN='SN'
    NO_NUCLEAR=''
    
    # All string except label_id
    def __init__(self, code, nuclear, label):
        self.code = code
        self.label = label
        self.nuclear = nuclear
        self.label_id = -1

    def is_none(self):
        return self.code == ''
    def is_finish(self):
        return self.code == 'PR'
    def is_shift(self):
        return self.code == 'SH'
    def is_reduce(self):
        return self.code == 'RD'

    def set_label_id(self, label_alpha):
        # for leaf the id is set into -1)
        self.label_id = label_alpha.get(self.label, -1)

    def get_str(self):
        if self.is_shift():
            return "SHIFT__"
        elif self.is_reduce():
            if self.nuclear == 'NN':
                return "REDUCE_NN_" + self.label
            if self.nuclear == 'NS':
                return "REDUCE_NS_" + self.label
            if self.nuclear == 'SN':
                return "REDUCE_SN_" + self.label
        elif self.is_finish():
            return "POPROOT__"
        else:
            return "NOACTION__"
    
    def set(self, code, nuclear, label):
        self.code = code
        self.nuclear = nuclear
        self.label = label

    def set_from_object(self, ac):
        self.code = ac.code
        self.label = ac.label
        self.nuclear = ac.nuclear
        self.label_id = ac.label_id

