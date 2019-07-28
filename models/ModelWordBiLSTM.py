import torch
from torch import nn
import torch.nn.utils.rnn as rnn

class WordBILSTM(nn.Module):
    def __init__(self, word_embedding_size):
        super().__init__()

        # Setting the embedding dimension
        self.word_embedding_size = word_embedding_size

        # Setting up the bilstm
        self.bilstm = nn.LSTM(word_embedding_size,
                              word_embedding_size, 1,
                              batch_first=True,
                              bidirectional=True)

        # Setting up the projection layer
        self.projection_layer = nn.Linear(2*word_embedding_size, word_embedding_size)

        # Dropout
        self.dropout = nn.Dropout(0.2)


    def forward(self, inputs):

        # Input parsing
        input_embeddings, lens = inputs

        # Sequence packing
        packed_input = rnn.pack_padded_sequence(input_embeddings, lens, batch_first=True, enforce_sorted=False)

        # Using the second BILSTM with the recently calculated word embeddings in order
        # to retrieve the sintax embeddings (or semantic)
        output, _ = self.bilstm(packed_input)

        # Padding back sequence
        output, lens = rnn.pad_packed_sequence(output, batch_first=True)

        # Split the outputs (forward and reverse outputs) and saves to var
        splitted_output = torch.split(output, self.word_embedding_size, dim=2)

        # Linear transformation into smaller dimension
        output = self.projection_layer(self.dropout(output))+input_embeddings

        # Dropout
        output = self.dropout(output)

        return (output, lens, splitted_output)
