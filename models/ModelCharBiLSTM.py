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
        outputs = []
        for sample in inputs:

            # Retrieving char embeddings for each word in sample
            sample = [self.char_embeddings_table(x.to(device)) for x in sample]

            # Sequence packing
            packed_seq = torch.nn.utils.rnn.pack_sequence(sample, enforce_sorted=False)

            # Passes the words on the sentence altogether through the char_bilstm,
            # retrieving for each word the 1st level word embedding
            output, _ = self.bilstm(packed_seq)

            # Sequence unpacking
            padded_output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)


            # For each word on the sample, save two outputs from the BILSTM outputs
            # Saving output of reverse (output on 0)
            sample_word_embeddings_first_char = padded_output[:,0,self.word_embedding_size:]

            # Saving output of forward (output on LENGTH-1)
            last = [padded_output[i,x-1,:self.word_embedding_size] for i,x in enumerate(output_lens)]
            sample_word_embeddings_last_char = torch.stack(last)


            concat_embeddings = torch.cat((sample_word_embeddings_first_char,
                                       sample_word_embeddings_last_char), dim=1)

            word_embeddings = self.projection_layer(concat_embeddings)
#             word_embeddings = self.dropout(word_embeddings)

            outputs.append(word_embeddings)


        lens = torch.LongTensor([len(seq) for seq in outputs])
        outputs = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)

        return outputs, lens
