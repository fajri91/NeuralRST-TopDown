import json, os

UNK_ID = 0

class Alphabet(object):
    def __init__(self, dictionary, name, for_label_index = False):
        self.alpha2id = {}
        self.id2alpha = {}
        self.name = name
        self.for_label_index = for_label_index
        self.alphas = list(dictionary.keys()) 
        
        ids = 0 
        if not for_label_index: # for non label
            self.alpha2id ['UNK'] = 0
            self.id2alpha [0] = 'UNK'
            ids += 1
        
        for alpha in self.alphas:
           self.alpha2id[alpha] = ids
           self.id2alpha[ids] = alpha
           ids += 1
        
        # add PAD for PADDING, it is used for label / action
        if for_label_index:
            self.alpha2id ['PAD'] = ids
            self.id2alpha [ids] = 'PAD'
            self.alphas += ['PAD']

        # add 'UNK' for non label index alphabet
        if not for_label_index:
            self.alphas += ['UNK']

    def get_content(self):
        return {'alpha2id': self.alpha2id, 'id2alpha': self.id2alpha, 'alphas': self.alphas}

    def word2id(self, word):
        if not self.for_label_index:
            return self.alpha2id.get(word, UNK_ID)
        else:
            return self.alpha2id.get(word, self.alpha2id['PAD'])

    def id2word(self, int_id):
        if not self.for_label_index:
            return self.id2alpha.get(int_id, 'UNK')
        else:
            return self.id2alpha.get(int_id, 'PAD')
    
    def __from_json(self, data):
        self.alphas = data["alphas"]
        self.alpha2id = data['alpha2id']
        for index, word in data['id2alpha'].items():
            self.id2alpha[int(index)] = word
        
    def size(self):
        if self.for_label_index:
            return len(self.alphas) - 1
        return len(self.alphas)

    def save(self, output_directory):
        try:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            json.dump(self.get_content(),
                    open(os.path.join(output_directory, self.name + ".json"), "w"), indent=4)

        except Exception as e:
            self.logger.warn("Alphabet is not saved: %s" % repr(e))

    def load(self, input_directory, for_label_index=False):
        self.__from_json(json.load(open(os.path.join(input_directory, self.name + ".json"))))
        self.for_label_index = for_label_index
