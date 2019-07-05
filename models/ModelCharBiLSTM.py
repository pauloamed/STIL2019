import torch
from torch import nn

class CharBILSTM(nn.Module):
    def __init__(self, char_embedding_size, word_embedding_size, char2id, device, padding_id=0):
        super().__init__()
        super(CharBILSTM, self).__init__()

        self.padding_id = padding_id

        # Setting the current device
        self.device = device

        # Setting the embeddings dimensions
        self.char_embedding_size = char_embedding_size
        self.word_embedding_size = word_embedding_size

        # Setting the char to int mapping
        self.char2id = char2id


        # Setting the char embedding lookup table
        self.char_embeddings_table = nn.Embedding(len(char2id),
                                                  char_embedding_size,
                                                  padding_idx=0)

        # Setting up the first BILSTM (char-level/morpho)
        self.bilstm = nn.LSTM(char_embedding_size,
                              word_embedding_size, 1,
                              batch_first=True,
                              bidirectional=True)

        # Setting up the projection layers
        self.projection_layer = nn.Linear(2*word_embedding_size, word_embedding_size)

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs):
        # For each sample (sentence) on the batch, get a list of 1st-level-word embeddings
        word_embeddings_first_char = []
        word_embeddings_last_char = []
        for sample in inputs:
            # Retrieving list of lists of ids
            converted_sample = self.get_converted_sample(sample)

            # Initializing memory for char-bilstm that is going to be used for
            # the words on the sentence altogether
            h = tuple([each.data for each in self.init_hidden(len(converted_sample))])

            # Converts list of lists into torch tensor (and passes it to device)
            inputs_ids = torch.LongTensor(converted_sample).to(self.device)

            # Uses the lookup table for retrieving the char embeddings
            input_vectors = self.char_embeddings_table(inputs_ids)

            # Passes the words on the sentence altogether through the char_bilstm,
            # retrieving for each word the 1st level word embedding
            output, _ = self.bilstm(input_vectors, h)

            # For each word on the sample, save two from the BILSTM outputs
            # Saving the first word on the first word list
            sample_word_embeddings_first_char = output[:,-1,:self.word_embedding_size]
            # Saving the last word on the last word list
            sample_word_embeddings_last_char = output[:,0,self.word_embedding_size:]

            # Saving each minor list (minor list = sample) into the major ones (major list = batch)
            word_embeddings_first_char.append(sample_word_embeddings_first_char)
            word_embeddings_last_char.append(sample_word_embeddings_last_char)

        # Padding the first word list
        word_embeddings_first_char = torch.nn.utils.rnn.pad_sequence(word_embeddings_first_char,
                                                                     batch_first=True)

        # Padding the last word list
        word_embeddings_last_char = torch.nn.utils.rnn.pad_sequence(word_embeddings_last_char,
                                                                    batch_first=True)

        # Concat and passing through porjection layer
        output = torch.cat((word_embeddings_first_char, word_embeddings_last_char), dim=2)

        output = self.projection_layer(output)

        # Applying dropout
        output = self.dropout(output)

        return output

    def get_converted_sample(self, sample):
        # Converts a whole sentence (list of strings) to a list of lists of ids
        converted_sample = []
        for word in sample:
            if word == self.padding_id: # When padding is reached
                break
            converted_sample.append([self.char2id.get(char, 1) for char in word])
        # Returns unpadded list of lists
        return converted_sample


    def init_hidden(self, batch_size):
        # Initializing the memory for one bilstm
        weight = next(self.parameters()).data
        hidden = (weight.new(2, batch_size, self.word_embedding_size).zero_().to(self.device),
                  weight.new(2, batch_size, self.word_embedding_size).zero_().to(self.device))
        return hidden
