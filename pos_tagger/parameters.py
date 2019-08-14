'''
================================================================================
=========================== EXECUTION PARAMETERS ===============================
================================================================================

TEST_MODE:  dont train, just test. if there is no model on the STATE_DICT_PATH
            (see below) path and TEST_MODE = True, it will be used a newly
            generated model
'''

TEST_MODE = False

'''
================================================================================
============================ DATASETS PARAMETERS ===============================
================================================================================

DATASETS_FOLDER: path to the folder with your dataset.

Fill in the datasets infos following the structure
DATASETS = [
    ('DATSET1_NAME', ('DATASET1_TRAIN_FILE', USE_TRAIN), ('DATASET1_VAL_FILE', USE_VAL),  'DATASET1_TEST_FILE'),
    ('DATSET2_NAME', ('DATASET2_TRAIN_FILE', USE_TRAIN), ('DATASET2_VAL_FILE', USE_VAL),  'DATASET2_TEST_FILE'),
    ...
    ('DATSETN_NAME', ('DATASETN_TRAIN_FILE', USE_TRAIN), ('DATASETN_VAL_FILE', USE_VAL),  'DATASETN_TEST_FILE'),
]

'''

DATASETS_FOLDER = 'data/'
DATASETS = [
    ('Macmorpho', ('macmorpho-train.mm.txt', True), ('macmorpho-dev.mm.txt', True), 'macmorpho-test.mm.txt'),
    ('Bosque', ('pt_bosque-ud-train.mm.txt', True), ('pt_bosque-ud-dev.mm.txt', True), 'pt_bosque-ud-test.mm.txt'),
    ('GSD', ('pt_gsd-ud-train.mm.txt', True), ('pt_gsd-ud-dev.mm.txt', True), 'pt_gsd-ud-test.mm.txt'),
    ('Linguateca', ('lgtc-train.mm.txt', True), ('lgtc-dev.mm.txt', True), 'lgtc-test.mm.txt')
]

'''
================================================================================
============================= MODEL PARAMETERS =================================
================================================================================

WORD_EMBEDDING_DIM:     size of the word embeddings, representations x^(1), x^(2) and x^(3)
CHAR_EMBEDDING_DIM:     size of char embeddings, components of the char look-up table from the $f$ block
BILSTM_SIZE:            size of the pos embeddings, representations from $p$ block

'''

WORD_EMBEDDING_DIM = 350
CHAR_EMBEDDING_DIM = 70
BILSTM_SIZE = 150

'''
================================================================================
============================ TRAINING PARAMETERS ===============================
================================================================================

EPOCHS:             num. of epochs that the model will train
BATCH_SIZE:         size of batch used for training
GRADIENT_CLIPPING:  gradient clipping used on training

'''

EPOCHS = 55
BATCH_SIZE = 32
GRADIENT_CLIPPING = 25

'''
================================================================================
========================== OUTPUT/LOG PARAMETERS ===============================
================================================================================

OUTPUT_PATH:        path to the output file with all the log from the program
LOG_LVL:            filter for the log that is going to be printed on the terminal.
                    if  LOG_LVL == 0: it will be printed erros, warnings, train and test output, tqdm
                        LOG_LVL == 1: it will be printed as in LOG_LVL=0 and success messages, descriptive log

STATE_DICT_PATH:    path for saving the model during training that is going to be
                    loaded for testing

'''

OUTPUT_PATH = 'output.txt'
LOG_LVL = 1
STATE_DICT_PATH = 'postag_sdict_WED_{}_CED_{}_BS_{}.pt'.format(WORD_EMBEDDING_DIM,
                                                                 CHAR_EMBEDDING_DIM,
                                                                 BILSTM_SIZE)
