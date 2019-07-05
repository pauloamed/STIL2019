import random, sys

def load_postag_checkpoint(filepath):
    c = torch.load(filepath)
    return c['min_val_loss'], c['optimizer_sd'], c['scheduler_sd']

def load_pretrain_checkpoint(filepath):
    c = torch.load(filepath)
    return c['char2id'], c['id2char'], c['word2id'], c['wrod2freq'], c['id2word'], c['min_val_loss'], c['optimizer_sd']

def get_batches(datasets, tvt, batch_size=1, policy="emilia"):
    seed = random.randrange(sys.maxsize)
    list_inputs, list_targets, list_n_batches, list_batches = [], [], [], []

    for d in datasets:
        if tvt == "train":
            total_len = sum([dataset.sent_train_size for dataset in datasets])
            datasets = [d for d in datasets if d.use_train]
            list_inputs.append(d.train_input)
            list_targets.append(d.train_target)
        elif tvt == 'val':
            total_len = sum([dataset.sent_val_size for dataset in datasets])
            datasets = [d for d in datasets if d.use_val]
            list_inputs.append(d.val_input)
            list_targets.append(d.val_target)
        elif tvt == 'test':
            total_len = sum([dataset.sent_test_size for dataset in datasets])
            list_inputs.append(d.test_input)
            list_targets.append(d.test_target)

    if policy == "visconde":
        for i in range(len(datasets)):
            random.Random(seed).shuffle(list_inputs[i])
            random.Random(seed).shuffle(list_targets[i])

    for i in range(len(datasets)):
        list_n_batches.append(len(list_inputs[i])//batch_size)
        list_inputs[i] = (list_inputs[i][0:list_n_batches[-1] * batch_size])
        list_targets[i] = (list_targets[i][0:list_n_batches[-1] * batch_size])


    if policy == "emilia": # Um de cada vez
        for i in range(len(datasets)):
            for ii in range(list_n_batches[i]):
                start = ii * batch_size
                end = (ii+1) * batch_size

                batch_inputs = list_inputs[i][start:end]
                batch_targets = list_targets[i][start:end]

                yield batch_inputs, batch_targets, datasets[i].name

    elif policy == "visconde": # Shuffle
        for i in range(len(datasets)):
            for ii in range(list_n_batches[i]):
                start = ii * batch_size
                end = (ii+1) * batch_size

                if(list_inputs[i][start:end] == []):
                   continue

                list_batches.append((list_inputs[i][start:end],
                                      list_targets[i][start:end], datasets[i].name))

        yield from list_batches
