import torch
from torch import nn

class POSTagger(nn.Module):

    def __init__(self, charBILSTM, wordBILSTM1, wordBILSTM2, n_bilstm_layers, n_bilstm_hidden, datasets, device):
        super().__init__()
        super(POSTagger, self).__init__()

        # Setting the current device
        self.device = device

        # Retrieving the model size (#layers and #units)
        self.n_tag_bilstm_layers = n_bilstm_layers
        self.n_tag_bilstm_hidden = n_bilstm_hidden

        # Retrieving the word emebedding size from the embedding model
        word_embedding_size = charBILSTM.word_embedding_size

        # Setting the embedding model as the feature extractor
        self.charBILSTM = charBILSTM
        self.wordBILSTM1 = wordBILSTM1
        self.wordBILSTM2 = wordBILSTM2

        # Defining the bilstm layer(s)
        if n_bilstm_layers == 1:
            # If there is only one layer, there can be no dropout
            self.tag_bilstm = nn.LSTM(word_embedding_size, self.n_tag_bilstm_hidden,
                                      self.n_tag_bilstm_layers, batch_first=True,
                                      bidirectional=True)
        else:
            # Setting dropout when there is more than one layer
            self.tag_bilstm = nn.LSTM(word_embedding_size, self.n_tag_bilstm_hidden,
                                      self.n_tag_bilstm_layers, dropout=0.5,
                                      batch_first=True,  bidirectional=True)

        # Setting the final layer (classifier) for each dataset being used
        classifiers = []
        self.dataset2id = dict()
        for d in datasets:
           classifiers.append(nn.Linear(self.n_tag_bilstm_hidden * 2, len(d.id2tag)))
           self.dataset2id[d.name] = len(self.dataset2id)
        self.classifiers = nn.ModuleList(classifiers)

        self.dropout = nn.Dropout(0.4)

    def forward(self, inputs, hidden, dataset_name):
        # Passing the input through the embeding model in order to retrieve the
        # embeddings
        embeddings1 = self.charBILSTM(inputs)
        embeddings2, _ = self.wordBILSTM1(embeddings1)
        embeddings3, _ = self.wordBILSTM2(embeddings2)

        # Passing the embeddings through the bilstm layer(s)
        out, hidden = self.tag_bilstm(embeddings3, hidden)

        # Applying dropout
        out = self.dropout(out)

        # Passing through the final layer for the current dataset
        out = self.classifiers[self.dataset2id[dataset_name]](out.contiguous().view(-1, self.n_tag_bilstm_hidden*2))

        return out, hidden

    def init_hidden(self, batch_size):

        # Initializing the memory
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_tag_bilstm_layers*2, batch_size, self.n_tag_bilstm_hidden).zero_(),
                  weight.new(self.n_tag_bilstm_layers*2, batch_size, self.n_tag_bilstm_hidden).zero_())
        hidden = (hidden[0].to(self.device), hidden[1].to(self.device))

        return hidden
