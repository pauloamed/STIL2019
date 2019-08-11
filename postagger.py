from math import sqrt

import datetime
import random
import sys

import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from pos_tagger.test import *
from pos_tagger.train import train
from pos_tagger.Dataset import Dataset, build_char_dict
# from pos_tagger.utils import load_postag_checkpoint, load_pretrain_checkpoint
from pos_tagger.parameters import *

from models.ModelCharBiLSTM import CharBILSTM
from models.ModelWordBiLSTM import WordBILSTM
from models.ModelPOSTagger import POSTagger




torch.set_printoptions(threshold=10000)

# Seting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################################################################################
#########                                                                    ############
#########                           PREPROCESSING                            ############
#########                                                                    ############
#########################################################################################

macmorpho = Dataset(MACMORPHO_FILE_PATHS, "Macmorpho", LOG)
bosque = Dataset(BOSQUE_FILE_PATHS, "Bosque", LOG)
gsd = Dataset(GSD_FILE_PATHS, "GSD", LOG)
linguateca = Dataset(LINGUATECA_FILE_PATHS, "Linguateca", LOG)

datasets = [macmorpho, bosque, gsd, linguateca]

char2id, id2char = build_char_dict(datasets, LOG)

for dataset in datasets:
    dataset.prepare(char2id)

if LOG:
    for dataset in datasets:
        print(dataset)

#########################################################################################
#########                                                                    ############
#########                     DEFINING MODELS AND TRAINING                   ############
#########                                                                    ############
#########################################################################################


charBILSTM = CharBILSTM(CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, char2id)
wordBILSTM1 = WordBILSTM(WORD_EMBEDDING_DIM)
wordBILSTM2 = WordBILSTM(WORD_EMBEDDING_DIM)

pos_model = POSTagger(charBILSTM, wordBILSTM1, wordBILSTM2,
                                       NUM_BILSTM_LAYERS, BILSTM_SIZE,
                                       datasets)

pos_model.to(device)

optimizer = optim.Adadelta(pos_model.parameters())

""" Training
"""
if LOG:
    print(pos_model)
min_val_loss = np.inf

pos_model, min_val_loss = train(device, pos_model, optimizer,
                                      datasets, min_val_loss, STATE_DICT_PATH,
                                      EPOCHS, TRAINING_POLICY, BATCH_SIZE)

try:
    # Loading the model with best loss on the validation
    pos_model.load_state_dict(torch.load(STATE_DICT_PATH))
except:
    print("Was not able to load trained model")

#########################################################################################
#########                                                                    ############
#########                        SAVING AND TESTING                          ############
#########                                                                    ############
#########################################################################################

# # Creating a checkpoint with the layers from the classifier, its weights,
# # the minimum validation loss achieved and the cat_to_name dict
# checkpoint = {'min_val_loss': min_val_loss,
#               'optimizer_sd': optimizer.state_dict()}

# # Saving the checkpoint
# torch.save(checkpoint, CHECKPOINT_PATH)

""" Testing
"""
accuracy(device, pos_model, datasets)
# confusion_matrix(device, pos_model, datasets)
# wrong_samples(device, pos_model, datasets)
# tsnes = set_tsne(device, pos_model, datasets)
# tsne_plot(device, pos_model, char2id, "Fui ao banco comprar um banco .", tsnes)
