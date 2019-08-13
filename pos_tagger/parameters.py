# Datasets paths
MACMORPHO_FILE_PATHS = ['data/macmorpho-train.mm.txt', 'data/macmorpho-dev.mm.txt', 'data/macmorpho-test.mm.txt']
BOSQUE_FILE_PATHS = ['data/pt_bosque-ud-train.mm.txt', 'data/pt_bosque-ud-dev.mm.txt', 'data/pt_bosque-ud-test.mm.txt']
GSD_FILE_PATHS = ['data/pt_gsd-ud-train.mm.txt', 'data/pt_gsd-ud-dev.mm.txt', 'data/pt_gsd-ud-test.mm.txt']
LINGUATECA_FILE_PATHS = ['data/lgtc-train.mm.txt', 'data/lgtc-dev.mm.txt', 'data/lgtc-test.mm.txt']


# Output path for all LOGs
OUTPUT_PATH = 'output.txt'

# Settings
LOG_LVL = 1 # serao impressas na tela mensagens com level menor que ou igual a LOG_LVL
TEST_MODE = True


# Model hiperparameters
WORD_EMBEDDING_DIM = 350
CHAR_EMBEDDING_DIM = 70
NUM_BILSTM_LAYERS = 1
BILSTM_SIZE = 150

# Training parameters
EPOCHS = 0 if TEST_MODE else 55
BATCH_SIZE = 32
TRAINING_POLICY = "visconde"

# Backup locations

STATE_DICT_PATH = 'postag_sdict_WED_{}_CED_{}_NBL_{}_BS_{}.pt'.format(WORD_EMBEDDING_DIM,
                                                                 CHAR_EMBEDDING_DIM,
                                                                 NUM_BILSTM_LAYERS,
                                                                 BILSTM_SIZE)
