import torch
from torch import nn
import numpy as np
from pos_tagger.utils import get_batches
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def accuracy(device, model, datasets):
    name2dataset = {d.name:d for d in datasets}

    for d in datasets:
        d.class_correct = 0
        d.class_total = 0

    model.eval()
    for itr in get_batches(datasets, "test"):
        # Getting vars
        inputs, targets, dataset_name = itr

        # Setting the input and the target (seding to GPU if needed)
        inputs = [[word.to(device) for word in sample] for sample in inputs]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)

        # Feeding the model
        output = model(inputs)

        # convert output probabilities to predicted class
        _, pred = torch.max(output[dataset_name], 1)

        # Formatando vetor
        pred = pred.view(1, -1)

        # calculate test accuracy for each object class
        for ii in range(output["length"]):
            if targets.data[0][ii].item() <= 1:
                continue
            if ii >= len(inputs[0][ii]):
                break

            label, predicted = targets.data[0][ii], pred.data[0][ii]
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


def confusion_matrix(device, model, datasets):
    conf_matrix = defaultdict(int)
    name2dataset = {d.name:d for d in datasets}

    model.eval()
    for itr in get_batches(datasets, "val"):
        # Getting vars
        inputs, targets, dataset_name = itr

        # Setting the input and the target (seding to GPU if needed)
        inputs = [[word.to(device) for word in sample] for sample in inputs]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)

        # Feeding the model
        output = model(inputs)

        # convert output probabilities to predicted class
        _, pred = torch.max(output[dataset_name], 1)

        # Formatando vetor
        pred = pred.view(1, -1)

        # calculate test accuracy for each object class
        for ii in range(output["length"]):
            if targets.data[0][ii].item() <= 1:
                continue
            if inputs[0][ii] == 0:
                break

            label, predicted = targets.data[0][ii], pred.data[0][ii]
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

def wrong_samples(device, model, datasets):
    name2dataset = {d.name:d for d in datasets}
    file = open("wrong_samples_{}".format(dataset_name),"w", encoding="utf-8")
    model.eval()
    for itr in get_batches(datasets, "val"):
        # Getting vars
        inputs, targets, dataset_name = itr

        # Setting the input and the target (seding to GPU if needed)
        inputs = [[word.to(device) for word in sample] for sample in inputs]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)

        # Feeding the model
        output = model(inputs)

        # convert output probabilities to predicted class
        _, pred = torch.max(output[dataset_name], 1)

        # Formatando vetor
        pred = pred.view(1, -1)

        # calculate test max_len for each object class
        for ii in range(output["length"]):
            if targets.data[0][ii].item() <= 1:
                continue
            if inputs[0][ii] == 0:
                break

            label, predicted = targets.data[0][ii], pred.data[0][ii]
            if label != predicted:
                words = ""
                for j in inputs[i]:
                    if j == 0:
                        break
                    else:
                        words += " {}".format(j.strip())

                tags = ""
                for j in range(batch_length):
                    if inputs[0][j] == 0:
                        break
                    tag = name2dataset[dataset_name].id2tag[pred.data[0][j].item()]
                    if pred.data[0][j].item() == targets.data[0][j].item():
                        tags += " {}".format(tag)
                    else:
                        target = name2dataset[dataset_name].id2tag[targets.data[0][j].item()]
                        tags += " P:{}_T:{}".format(tag, target)

                file.write("{}\n{}\n\n".format(words, tags))
                continue
    file.close()

# def set_tsne(device, model, datasets):
#     model.eval()
#
#     tsnes = {
#         "embeddings1" : TSNE(),
#         "embeddings2" : TSNE(),
#         "embeddings3" : TSNE(),
#         "embeddings4" : TSNE()
#     }
#
#     for itr in get_batches(datasets, "train"):
#         # Getting vars
#         inputs, targets, dataset_name = itr
#
#         # Setting the input and the target (seding to GPU if needed)
#         inputs = [[word.to(device) for word in sample] for sample in inputs]
#
#         output = model(inputs)
#
#         tsnes["embeddings1"].fit(output["embeddings1"].squeeze().to('cpu').data.numpy()) # char bilstm
#         tsnes["embeddings2"].fit(output["embeddings2"].squeeze().to('cpu').data.numpy()) # 1 word bilstm
#         tsnes["embeddings3"].fit(output["embeddings3"].squeeze().to('cpu').data.numpy()) # 2 word bilstm
#         tsnes["embeddings4"].fit(output["embeddings4"].squeeze().to('cpu').data.numpy()) # tag bilstm
#
#     return tsnes

# def tsne_plot(device, model, char2id, sent, tsnes):
#     model.eval()
#
#     sent = sent.split(" ")
#     isent = [[torch.LongTensor([char2id.get(c,1) for c in word]).to(device) for word in sent]]
#
#     output = model(sent)
#
#     for i in range(1, 5):
#         embd = output["embeddings"+i].squeeze().to('cpu').data.numpy()
#         embd_tsne = tsnes["embeddings"+i].fit_transform(embd)
#
#         for idx in range(len(embd)):
#             plt.scatter(*embed_tsne[idx], color='steelblue')
#             plt.annotate(sent[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
