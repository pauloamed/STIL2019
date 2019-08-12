import torch
from pos_tagger.utils import send_output

def build_char_dict(datasets):
    send_output("\n>> Building char dict...", 1)

    extracted_chars = set()
    for dataset in datasets:
      extracted_chars = extracted_chars.union(dataset.extract_chars())
    chars = [' ', 'UNK'] + list(sorted(extracted_chars))

    # Criando estruturas do vocabulário
    char2id = {char: index for index, char in enumerate(chars)}
    id2char = [char for char, _ in char2id.items()]

    send_output("<< Finished building dicts!", 1)
    return char2id, id2char


class Dataset():
    def __init__(self, path_to_files, dataset_name, use_delimiters=True, use_train=True, use_val=True):
        self.name = dataset_name

        send_output("\n>> Initializing {} dataset".format(self.name), 1)
        # Loading to each dataset subset
        send_output(">>> Started loading dataset", 1)
        self.train_data = self.__load_data(path_to_files[0])
        self.val_data = self.__load_data(path_to_files[1])
        self.test_data = self.__load_data(path_to_files[2])

        send_output("<<< Finished loading dataset", 1)

            # Parsing
        send_output(">>> Started parsing data from dataset", 1)
        self.train_data, self.word_train_size = self.__parse_data(self.train_data, use_delimiters)
        self.val_data, self.word_val_size = self.__parse_data(self.val_data, use_delimiters)
        self.test_data, self.word_test_size = self.__parse_data(self.test_data, use_delimiters)

        send_output("<<< Finished parsing data from dataset", 1)

        # Setting bool flags
        self.use_train = use_train
        self.use_val = use_val

        # Setup tag dicts
        self.extract_tag_dict()

        # Train, val and test data size
        self.sent_train_size = len(self.train_data[0])
        self.sent_val_size = len(self.val_data[0])
        self.sent_test_size = len(self.test_data[0])

        # Setting training and val loss
        self.train_loss = 0.0
        self.val_loss = 0.0

        # Setting test counters
        self.class_correct = [0 for _ in range(len(self.tag2id))]
        self.class_total = [0 for _ in range(len(self.tag2id))]

        send_output("<< Finished initializing {} dataset".format(self.name), 1)

    def extract_chars(self):
        send_output(">>> Started extracting chars from {} dataset".format(self.name), 1)
        ret = {c for sample in self.train_data for token in sample for c in token[0]}
        send_output("<<< Finished extracting chars from {} dataset".format(self.name), 1)
        return ret

    def extract_tag_dict(self):
        send_output(">>> Started building tag dict for dataset", 1)
        extracted_tags = {token[1] for sample in self.train_data for token in sample}
        tags = list(sorted(extracted_tags))

        self.tag2id = {"BOS": 0, "EOS": 1}
        for tag in tags:
            if tag not in self.tag2id:
                self.tag2id[tag] = len(self.tag2id)

        # Criando dicionario para as tags
        self.id2tag = [tag for tag, _ in self.tag2id.items()]

        send_output("<<< Finished building tag dict for dataset", 1)

    def prepare(self, char2id):
        send_output("\n>> Started preparing {} dataset".format(self.name), 1)
        self.train_input, self.train_target = self.__prepare_data(self.train_data, char2id)
        self.val_input, self.val_target = self.__prepare_data(self.val_data, char2id)
        self.test_input, self.test_target = self.__prepare_data(self.test_data, char2id)
        send_output("<< Finished preparing {} dataset".format(self.name), 1)
        del self.train_data, self.val_data, self.test_data

    def __prepare_data(self, dataset, char2id):
        inputs = [[torch.LongTensor([char2id.get(c, 1) for c in token[0]]) for token in sample] for sample in dataset]
        targets = [torch.LongTensor([self.tag2id.get(token[1], 0) for token in sample]) for sample in dataset]
        return (inputs, targets)

    def __load_data(self, path_to_file):
        with open(path_to_file, 'r', encoding='utf-8') as f:
            data = f.read()
        return data

    def __parse_data(self, data, use_delimiters):
        counter = 0

        BOW = "\002" if use_delimiters else ""
        EOW = "\003" if use_delimiters else ""
        BOS = [["\001", "BOS"]] if use_delimiters else []
        EOS = [["\004", "EOS"]] if use_delimiters else []

        dataset = []
        for sample in data.split('\n'):
            if sample == '':
                continue

            s = sample.strip().split(' ')
            counter += len(s)
            middle = [[BOW + token.rsplit('_', 1)[0] + EOW, token.rsplit('_', 1)[1]]
                                                                    for token in s]
            dataset.append(BOS + middle + EOS)
        return dataset, counter

    def __str__(self):
        ret = ""
        ret += ("=================================================================\n")
        ret += ("{} Dataset\n".format(self.name))
        ret += ("Train dataset #sents: {} #words: {}\n".format(len(self.train_input), self.word_train_size))
        ret += ("Val dataset #sents: {} #words: {}\n".format(len(self.val_input), self.word_val_size))
        ret += ("Test dataset #sents: {} #words: {}\n".format(len(self.test_input),self.word_test_size))
        ret += ("Tag set: [{}]\n".format(", ".join(self.id2tag)))
        ret += ("=================================================================\n")

        return ret
