# Datasets paths
MACMORPHO_FILE_PATHS = ['data/macmorpho-train.mm', 'data/macmorpho-dev.mm', 'data/macmorpho-test.mm']
BOSQUE_FILE_PATHS = ['data/pt_bosque-ud-train.mm', 'data/pt_bosque-ud-dev.mm', 'data/pt_bosque-ud-test.mm']
GSD_FILE_PATHS = ['data/pt_gsd-ud-train.mm', 'data/pt_gsd-ud-dev.mm', 'data/pt_gsd-ud-test.mm']
LINGUATECA_FILE_PATHS = ['data/lgtc-train.mm', 'data/lgtc-dev.mm', 'data/lgtc-test.mm']
EWT_FILE_PATHS = ['data/en_ewt-ud-train.mm','data/en_ewt-ud-dev.mm','data/en_ewt-ud-test.mm']
PTB_FILE_PATHS = ['data/ptb-train.mm','data/ptb-dev.mm','data/ptb-test.mm']

# Model hiperparameters
WORD_EMBEDDING_DIM = 350
CHAR_EMBEDDING_DIM = 60
NUM_BILSTM_LAYERS = 1
BILSTM_SIZE = 150

# Training parameters
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.4
MOMENTUM_RATE = 0.9
TRAINING_POLICY = "visconde"
FIX_WEIGHTS = False
CONTINUE_TRAINING = False
USE_PRETRAINED = False

# Backup locations
CHECKPOINT_PATH = './postag_ckpoint_WED:{}_CED:{}_NBL:{}_BS:{}.tar'.format(WORD_EMBEDDING_DIM,
                                                                    CHAR_EMBEDDING_DIM,
                                                                    NUM_BILSTM_LAYERS,
                                                                    BILSTM_SIZE)

STATE_DICT_PATH = './postag_sdict_WED:{}_CED:{}_NBL:{}_BS:{}.pt'.format(WORD_EMBEDDING_DIM,
                                                                 CHAR_EMBEDDING_DIM,
                                                                 NUM_BILSTM_LAYERS,
                                                                 BILSTM_SIZE)


CHAR_BILSTM_SD_PATH = './charbilstm_sdict_WED:{}_CED:{}.pt'.format(WORD_EMBEDDING_DIM,
                                                                   CHAR_EMBEDDING_DIM)
WORD_BILSTM1_SD_PATH ='./wordbilstm1_sdict_WED:{}_CED:{}.pt'.format(WORD_EMBEDDING_DIM,
                                                                    CHAR_EMBEDDING_DIM)
WORD_BILSTM2_SD_PATH ='./wordbilstm2_sdict_WED:{}_CED:{}.pt'.format(WORD_EMBEDDING_DIM,
                                                                    CHAR_EMBEDDING_DIM)
