import torch
from torch import nn
from torch.nn.utils import rnn

class POSTagger(nn.Module):

    def __init__(self, charBILSTM, wordBILSTM1, wordBILSTM2, n_bilstm_hidden, datasets):
        super().__init__()

        # Retrieving the model size (#layers and #units)
        self.n_tag_bilstm_hidden = n_bilstm_hidden

        # Retrieving the word emebedding size from the embedding model
        word_embedding_size = charBILSTM.word_embedding_size

        # Setting the embedding model as the feature extractor
        self.charBILSTM = charBILSTM
        self.wordBILSTM1 = wordBILSTM1
        self.wordBILSTM2 = wordBILSTM2

        # Defining the bilstm layer(s)
        self.tag_bilstm = nn.LSTM(word_embedding_size, self.n_tag_bilstm_hidden,
                                  1, batch_first=True,
                                  bidirectional=True)

        # Setting the final layer (classifier) for each dataset being used
        classifiers = []
        self.dataset2id = dict()
        for d in datasets:
            classifiers.append(nn.Linear(self.n_tag_bilstm_hidden * 2, len(d.id2tag)))
            self.dataset2id[d.name] = len(self.dataset2id)
        self.classifiers = nn.ModuleList(classifiers)

        # Saving datasets names

        self.dropout = nn.Dropout(0.4)

    def forward(self, inputs):
        # Passing the input through the embeding model in order to retrieve the
        # embeddings


        # Setting output formatting
        output = {
            "embeddings1" : None, # Context free representations
            "embeddings2": None, # 1-lvl context representations
            "embeddings3": None, # 2-lvl context representations
            "embeddings3_fwd": None, # gonna be used for LM
            "embeddings3_rev": None, # gonna be used for LM
            "embeddings4": None, # pos refined word embeddings
            "length": None # batch length
        }
        # It will be computed the output for all datasets
        '''
            "dataset_1": None, # output for dataset1
            "dataset_2": None, # output for dataset1
            ...
            "dataset_n": None # output for dataset1
        '''
        output.update({dataset: None for dataset in self.dataset2id})


        embeddings1, lens = self.charBILSTM(inputs) # Char BILSTM
        output["embeddings1"] = embeddings1.clone() # Saving output

        embeddings2, lens, _ = self.wordBILSTM1((embeddings1, lens)) # 1-Word BILSTM
        output["embeddings2"] = embeddings2.clone() # Saving output

        embeddings3, lens, (rev_embeddings3, fwd_embeddings3) = self.wordBILSTM2((embeddings2, lens))
        output["embeddings3"] = embeddings3.clone() # Saving output
        output["embeddings3_rev"] = rev_embeddings3 # Saving output
        output["embeddings3_fwd"] = fwd_embeddings3 # Saving output
        output["length"] = max(lens) # Saving output

        # Sequence packing
        embeddings3 = rnn.pack_sequence(embeddings3, enforce_sorted=False)

        # Passing the embeddings through the bilstm layer(s)
        refined_embeddings, _ = self.tag_bilstm(embeddings3)

        refined_embeddings, _ = rnn.pad_packed_sequence(refined_embeddings, batch_first=True)
        output["embeddings4"] = refined_embeddings.clone()

        # Applying dropout
        refined_embeddings = self.dropout(refined_embeddings)

        # Updating view
        # see as: B x L x I (batch_size x length x input_size)
        refined_embeddings = refined_embeddings.contiguous().view(-1, output["length"], self.n_tag_bilstm_hidden*2)

        # Saving final outputs
        # Passing through the final layer for each dataset
        output.update({name : self.classifiers[idx](refined_embeddings)
                            for idx, name in enumerate(self.dataset2id)})

        return output
