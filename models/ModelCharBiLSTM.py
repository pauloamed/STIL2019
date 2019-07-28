import torch
from torch import nn
import torch.nn.utils.rnn as rnn

class CharBILSTM(nn.Module):
    def __init__(self, char_embedding_size, word_embedding_size, char2id):
        super().__init__()

        # Setting the embeddings dimensions
        self.char_embedding_size = char_embedding_size
        self.word_embedding_size = word_embedding_size

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
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        # For each sample (sentence) on the batch, get a list of 1st-level-word embeddings
        outputs, lens = [], []
        for sample in inputs:

            # Retrieving char embeddings for each word in sample
            embedded_sample = [self.dropout(self.char_embeddings_table(word)) for word in sample]

            # Sequence packing
            packed_embedded_sample = rnn.pack_sequence(embedded_sample, enforce_sorted=False)

            # Passes the words on the sentence altogether through the char_bilstm
            output, _ = self.bilstm(packed_embedded_sample)

            # Sequence unpacking
            padded_output, output_lens = rnn.pad_packed_sequence(output, batch_first=True)

            # For each word on the sample, save two outputs from the BILSTM outputs
            # Saving output of reverse lstm (output on 0)
            output_reverse = padded_output[:,0,self.word_embedding_size:]

            # Saving output of forward lstm (output on LENGTH-1)
            output_forward = [padded_output[i, word_len-1, :self.word_embedding_size]
                                           for i, word_len in enumerate(output_lens)]
            output_forward = torch.stack(output_forward)

            # Concat outputs
            concat_embeddings = torch.cat((output_reverse, output_forward), dim=1)

            # Projects on a word_embedding dimension
            word_embeddings = self.projection_layer(concat_embeddings)

            # Saves to output
            outputs.append(self.dropout(word_embeddings))
            lens.append(len(word_embeddings))

        # Padding sequence as needed
        outputs = rnn.pad_sequence(outputs, batch_first=True)

        return outputs, lens
