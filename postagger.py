from math import sqrt

import datetime
import random
import sys

import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import pos_tagger.test as test
import pos_tagger.train as train
import pos_tagger.Dataset as ds
import pos_tagger.utils as utils
from pos_tagger.parameters import *

import models.ModelCharBiLSTM as char_model
import models.ModelWordBiLSTM as word_model
import models.ModelPOSTagger as pos_tagger_model


torch.set_printoptions(threshold=10000)


# Seting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################################################################################
#########                                                                    ############
#########                           PREPROCESSING                            ############
#########                                                                    ############
#########################################################################################

macmorpho = ds.Dataset(MACMORPHO_FILE_PATHS, "Macmorpho")
bosque = ds.Dataset(BOSQUE_FILE_PATHS, "Bosque", use_val=False)
gsd = ds.Dataset(GSD_FILE_PATHS, "GSD", use_val=False)
linguateca = ds.Dataset(LINGUATECA_FILE_PATHS, "Linguateca", use_val=False)

datasets = [macmorpho, bosque, gsd, linguateca]

char2id, id2char = ds.build_char_dict(datasets)

for dataset in datasets:
    dataset.prepare(char2id)
for dataset in datasets:
    print(dataset)

#########################################################################################
#########                                                                    ############
#########                     DEFINING MODELS AND TRAINING                   ############
#########                                                                    ############
#########################################################################################


charBILSTM = char_model.CharBILSTM(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, char2id)
wordBILSTM1 = word_model.WordBILSTM(WORD_EMBEDDING_DIM)
wordBILSTM2 = word_model.WordBILSTM(WORD_EMBEDDING_DIM)

pos_model = pos_tagger_model.POSTagger(charBILSTM, wordBILSTM1, wordBILSTM2,
                                       NUM_BILSTM_LAYERS, BILSTM_SIZE,
                                       datasets)

pos_model.to(device)

optimizer = optim.SGD(pos_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM_RATE)

""" Training
"""

print(pos_model)
min_val_loss = np.inf

pos_model, min_val_loss = train.train(device, pos_model, optimizer,
                                      datasets, min_val_loss, STATE_DICT_PATH,
                                      EPOCHS, TRAINING_POLICY, BATCH_SIZE)

# Loading the model with best loss on the validation
pos_model.load_state_dict(torch.load(STATE_DICT_PATH))

#########################################################################################
#########                                                                    ############
#########                        SAVING AND TESTING                          ############
#########                                                                    ############
#########################################################################################

# # Creating a checkpoint with the layers from the classifier, its weights,
# # the minimum validation loss achieved and the cat_to_name dict
checkpoint = {'min_val_loss': min_val_loss,
              'optimizer_sd': optimizer.state_dict()}

# # Saving the checkpoint
torch.save(checkpoint, CHECKPOINT_PATH)

""" Testing
"""
test.accuracy(device, pos_model, datasets)
test.confusion_matrix(device, pos_model, datasets)
test.wrong_samples(device, pos_model, datasets)
