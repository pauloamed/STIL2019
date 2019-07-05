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


    def forward(self, input_embeddings):
        # Retrieving batch size from input vectors
        batch_size = len(input_embeddings)

        # Initializing memory for the second and third BILSTMs
        h = tuple([each.data for each in self.init_hidden(batch_size)])

        # Using the second BILSTM with the recently calculated word embeddings in order
        # to retrieve the sintax embeddings
        output_embeddings, _ = self.bilstm(input_embeddings, h)


        # Split the outputs (forward and reverse outputs) and define the final vector as a
        # linear combination of the splited output
        splitted_output_embeddings = torch.split(output_embeddings, self.word_embedding_size, dim=2)

        # Linear projection into smaller dimension
        output = self.projection_layer(output_embeddings)+input_embeddings

        output = self.dropout(output)

        return (output, splitted_output_embeddings)

    def init_hidden(self, batch_size):
        # Initializing the memory for one bilstm
        weight = next(self.parameters()).data
        hidden = (weight.new(2, batch_size, self.word_embedding_size).zero_().to(self.device),
                  weight.new(2, batch_size, self.word_embedding_size).zero_().to(self.device))
        return hidden
