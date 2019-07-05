torch.set_printoptions(threshold=10000)


# Seting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################################################################################
#########                                                                    ############
#########                           PREPROCESSING                            ############
#########                                                                    ############
#########################################################################################

macmorpho = Dataset(MACMORPHO_FILE_PATHS, "Macmorpho")

datasets = [macmorpho]

char2id, id2char = build_char_dict(datasets)

for dataset in datasets:
    dataset.prepare(char2id)
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

optimizer = optim.SGD(pos_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM_RATE)

""" Training
"""

print(pos_model)

min_val_loss = np.inf
pos_model, min_val_loss = train(device, pos_model, optimizer,
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
accuracy(device, pos_model, datasets)
confusion_matrix(device, pos_model, datasets)
wrong_samples(device, pos_model, datasets)
