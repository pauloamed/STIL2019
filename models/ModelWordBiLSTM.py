import torch
from torch import nn

class WordBILSTM(nn.Module):
    def __init__(self, word_embedding_size, device):
        super().__init__()
        super(WordBILSTM, self).__init__()

        # Setting the current device
        self.device = device

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

        # Sequence packing
        input_embeddings, lens = inputs

        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings,
                                                             lens, batch_first=True,
                                                             enforce_sorted=False)

        # Using the second BILSTM with the recently calculated word embeddings in order
        # to retrieve the sintax embeddings
        output_embeddings, _ = self.bilstm(packed_seq)

        output_embeddings, output_lens = torch.nn.utils.rnn.pad_packed_sequence(output_embeddings, batch_first=True)


        # Split the outputs (forward and reverse outputs) and define the final vector as a
        # linear combination of the splited output
        splitted_output_embeddings = torch.split(output_embeddings, self.word_embedding_size, dim=2)

        # Linear projection into smaller dimension
        output = self.projection_layer(output_embeddings)+input_embeddings

        output = self.dropout(output)

        return (output, output_lens, splitted_output_embeddings)
