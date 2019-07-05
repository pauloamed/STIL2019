import torch
from torch import nn
import numpy as np
import pandas as pd

from collections import defaultdict

import pos_tagger.utils as utils

def accuracy(device, model, datasets, batch_size=1):
    name2dataset = {d.name:d for d in datasets}

    for d in datasets:
        d.class_correct = 0
        d.class_total = 0

    model.eval()
    for itr in utils.get_batches(datasets, "test", batch_size):
        # Getting vars
        inputs, targets, dataset_name, batch_length = itr

        # Setting the input and the target
        targets = torch.LongTensor(targets).to(device)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        val_h = tuple([each.data for each in model.init_hidden(batch_size)])

        # Calculating the output
        output, val_h = model(inputs, val_h, dataset_name)

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # Formatando vetor
        pred = pred.view(batch_size, -1)

        # calculate test accuracy for each object class
        for i in range(batch_size):
            for ii in range(batch_length):
                if targets.data[i][ii].item() <= 1:
                    continue
                if inputs[i][ii] == 0:
                    break

                label, predicted = targets.data[i][ii], pred.data[i][ii]
                name2dataset[dataset_name].class_correct += 1 if label == predicted else 0
                name2dataset[dataset_name].class_total += 1

    soma_correct = np.sum([d.class_correct for d in datasets])
    soma_total = np.sum([d.class_total for d in datasets])

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (100. * soma_correct / soma_total,
                                                          soma_correct, soma_total))

    for d in datasets:
        print('\nTest Accuracy (on {} Dataset): {:.2f}% ({}/{})'.format(d.name,
                                                                        100. * d.class_correct / d.class_total,
                                                                        d.class_correct, d.class_total))


def confusion_matrix(device, model, datasets, batch_size=1):
    conf_matrix = defaultdict(int)
    name2dataset = {d.name:d for d in datasets}

    model.eval()
    for itr in utils.get_batches(datasets, "val", batch_size):
        # Getting vars
        inputs, targets, dataset_name, batch_length = itr

        # Setting the input and the target
        targets = torch.LongTensor(targets).to(device)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        val_h = tuple([each.data for each in model.init_hidden(batch_size)])

        # Calculating the output
        output, val_h = model(inputs, val_h, dataset_name)

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # Formatando vetor
        pred = pred.view(batch_size, -1)

        # calculate test accuracy for each object class
        for i in range(batch_size):
            for ii in range(batch_length):
                if targets.data[i][ii].item() <= 1:
                    continue
                if inputs[i][ii] == 0:
                    break

                label, predicted = targets.data[i][ii], pred.data[i][ii]
                conf_matrix[(dataset_name, predicted.item(), label.item())] += 1

        for d in [d for d in datasets if d.use_val]:
            tagset_size = len(d.id2tag)
            rows = [[conf_matrix[(d.name, linha, coluna)] for coluna in range(tagset_size)]
                                                         for linha in range(tagset_size)]

            df = pd.DataFrame(rows, columns = d.id2tag, index=d.id2tag)
            sum_columns = df.sum(axis=0)
            sum_columns.name = "sum_columns"
            sum_rows = df.sum(axis=1).astype('int64')

            df = df.append(sum_columns)
            df['sum_rows'] = sum_rows
            df.to_csv("matriz_conf_{}".format(d.name), sep='\t', float_format='%d', encoding='utf-8')

def wrong_samples(device, model, datasets, batch_size=2):
    name2dataset = {d.name:d for d in datasets}
    model.eval()
    for itr in utils.get_batches(datasets, "val", batch_size):
        # Getting vars
        inputs, targets, dataset_name, batch_length = itr

        file = open("wrong_samples_{}".format(dataset_name),"w", encoding="utf-8")

        # Setting the input and the target
        targets = torch.LongTensor(targets).to(device)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        val_h = tuple([each.data for each in model.init_hidden(batch_size)])

        # Calculating the output
        output, val_h = model(inputs, val_h, dataset_name)

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)

        # Formatando vetor
        pred = pred.view(batch_size, -1)

        # calculate test accuracy for each object class
        for i in range(batch_size):
            for ii in range(batch_length):
                if targets.data[i][ii].item() <= 1:
                    continue
                if inputs[i][ii] == 0:
                    break

                label, predicted = targets.data[i][ii], pred.data[i][ii]
                if label != predicted:
                    words = ""
                    for j in inputs[i]:
                        if j == 0:
                            break
                        else:
                            words += " {}".format(j.strip())

                    tags = ""
                    for j in range(batch_length):
                        if inputs[i][j] == 0:
                            break
                        tag = name2dataset[dataset_name].id2tag[pred.data[i][j].item()]
                        if pred.data[i][j].item() == targets.data[i][j].item():
                            tags += " {}".format(tag)
                        else:
                            target = name2dataset[dataset_name].id2tag[targets.data[i][j].item()]
                            tags += " P:{}_T:{}".format(tag, target)

                    file.write("{}\n{}\n\n".format(words, tags))
                    continue
        file.close()


def custom_test(sentence, my_model, n_datasets):
    sample = [[sentence.split()]]
    sample = replace(sample)
    my_model.eval()

    for i_dataset in range(n_datasets):
        for x in sample:
            h = tuple([each.data for each in my_model.init_hidden(1)])
            output, h = my_model(x, h, i_dataset)
            _, pred = torch.max(output, 1)

        print([s for s in sample[0][0]])
        print([list_tag_dicts[i_dataset][1][p.item()] for p in pred])
