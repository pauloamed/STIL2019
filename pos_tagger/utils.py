import random, sys
import tqdm
from pos_tagger.parameters import LOG_LVL, OUTPUT_PATH

# def load_postag_checkpoint(filepath):
#     c = torch.load(filepath)
#     return c['min_val_loss'], c['optimizer_sd'], c['scheduler_sd']
#
# def load_pretrain_checkpoint(filepath):
#     c = torch.load(filepath)
#     return c['char2id'], c['id2char'], c['word2id'], c['wrod2freq'], c['id2word'], c['min_val_loss'], c['optimizer_sd']

def send_output(str, log_level):
    if log_level <= LOG_LVL:
        print(str)
    # try:
    file = open(OUTPUT_PATH, "a")
    file.write(str + "\n")
    file.close()
    # except:
    #     if log_level <= LOG_LVL:
    #         print("Was not able to open output file")

def do_policy(policy, datasets, batch_size, list_samples):
    seed = random.randrange(sys.maxsize)

    list_batches, list_n_batches = [], []

    for i in range(len(datasets)):
        list_n_batches.append(len(list_samples[i][0])//batch_size)
        list_samples[i] = (list_samples[i][0][0:list_n_batches[-1] * batch_size],
                           list_samples[i][1][0:list_n_batches[-1] * batch_size])

    if policy == "emilia":
        for i in range(len(datasets)):
            for ii in range(list_n_batches[i]):
                start = ii * batch_size
                end = (ii+1) * batch_size

                if(list_samples[i][0][start:end] == []):
                    continue

                batch_inputs = list_samples[i][0][start:end]
                batch_targets = list_samples[i][1][start:end]

                list_batches.append((batch_inputs, batch_targets, datasets[i].name))
    elif policy == "visconde":
        for i in range(len(datasets)):
            for ii in range(list_n_batches[i]):
                start = ii * batch_size
                end = (ii+1) * batch_size

                if(list_samples[i][0][start:end] == []):
                   continue

                batch_inputs = list_samples[i][0][start:end]
                batch_targets = list_samples[i][1][start:end]

                list_batches.append((batch_inputs, batch_targets, datasets[i].name))

        random.Random(seed).shuffle(list_batches)
    else:
        pass

    return list_batches

def get_batches(datasets, tvt, batch_size=1, policy="emilia"):
    list_samples = []

    if tvt == "train":
        datasets = [d for d in datasets if d.use_train]
        total_len = sum([dataset.sent_train_size for dataset in datasets])
        list_samples = [(d.train_input, d.train_target) for d in datasets]
    elif tvt == 'val':
        datasets = [d for d in datasets if d.use_val]
        total_len = sum([dataset.sent_val_size for dataset in datasets])
        list_samples = [(d.val_input, d.val_target) for d in datasets]
    elif tvt == 'test':
        total_len = sum([dataset.sent_test_size for dataset in datasets])
        list_samples = [(d.test_input, d.test_target) for d in datasets]


    list_batches = do_policy(policy, datasets, batch_size, list_samples)

    desc = "{}: batch_size={}, policy={}".format(tvt, batch_size, policy)
    for b in tqdm.tqdm(list_batches, desc=desc):
        yield b
