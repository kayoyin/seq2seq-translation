import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData():
    asl, en = Lang("asl"), Lang("en")
    with open("../data/en_train.txt", "r") as file:
        en_train = file.readlines()
    with open("../data/asl_train.txt", "r") as file:
        asl_train = file.readlines()
    print("Counting words...")
    for a,e in zip(asl_train, en_train):
        asl.addSentence(a)
        en.addSentence(e)
    print("Counted words:")
    print(asl.name, asl.n_words)
    print(en.name, en.n_words)
    return asl, en, asl_train, en_train

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(asl, en, asl_txt, en_txt):
    input_tensor = tensorFromSentence(asl, asl_txt)
    target_tensor = tensorFromSentence(en, en_txt)
    return (input_tensor, target_tensor)

if __name__ == "__main__":
    prepareData()