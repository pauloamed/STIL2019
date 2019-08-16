import torch
from torch import nn
import numpy as np
from pos_tagger.utils import get_batches, send_output

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
        _, pred = torch.max(output[dataset_name], 2)

        # Formatando vetor
        pred = pred.view(1, -1)

        # calculate test accuracy for each object class
        for ii in range(output["length"]):
            if ii >= len(targets[0]):
                break
            if targets.data[0][ii].item() <= 1:
                continue

            label, predicted = targets.data[0][ii], pred.data[0][ii]
            name2dataset[dataset_name].class_correct += 1 if label == predicted else 0
            name2dataset[dataset_name].class_total += 1

    soma_correct = np.sum([d.class_correct for d in datasets])
    soma_total = np.sum([d.class_total for d in datasets])
    accuracy_ = 100. * soma_correct / soma_total
    out_str = '\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (accuracy_, soma_correct,
                                                                soma_total)

    send_output(out_str, 0)

    for d in datasets:
        accuracy_d = 100. * d.class_correct / d.class_total
        out_str = '\nTest Accuracy (on {} Dataset): {:.2f}% ({}/{})'.format(d.name, accuracy_d,
                                                                        d.class_correct, d.class_total)
        send_output(out_str, 0)

def tagged_samples(device, model, datasets, id2char):
    name2dataset = {d.name:d for d in datasets if d.use_val == True}
    name2tagged_samples = {d.name:([],[]) for d in datasets if d.use_val == True}


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
        _, pred = torch.max(output[dataset_name], 2)

        # Formatando vetor
        pred = pred.view(1, -1)

        # lists storing the current sample
        _inputs, gold_tags, pred_tags = [], [], []

        # boolean storing if the sentence had one wrong labeled word
        mistagged = False

        for ii in range(output["length"]):
            # Esse loop inteiro assume que BATCH_SIZE = 1

            if ii >= len(targets[0]):
                break
            if targets.data[0][ii].item() <= 1:
                continue

            label, predicted = targets.data[0][ii], pred.data[0][ii]

            if label != predicted:
                mistagged = True

            _inputs.append("".join([id2char[charid] for charid in inputs[0][ii]]))
            gold_tags.append(name2dataset[dataset_name].id2tag[targets.data[0][ii].item()])
            pred_tags.append(name2dataset[dataset_name].id2tag[pred.data[0][ii].item()])

        if mistagged:
            name2tagged_samples[dataset_name][1].append((_inputs, gold_tags, pred_tags))
        else:
            name2tagged_samples[dataset_name][0].append((_inputs, gold_tags, pred_tags))


    for dataset_name, (correct_samples, mistagged_samples) in name2tagged_samples.items():

        file = open("tagged_samples_{}".format(dataset_name),"w", encoding="utf-8")
        file.write("\n\n================================================================\n")
        file.write("====================  MISTAGGED SAMPLES  =======================\n")
        file.write("================================================================\n\n")

        for sample_input, sample_gold_tags, sample_pred_tags in mistagged_samples:
            file.write("\n\n\n")
            file.write("{}\n".format(" ".join(sample_input)))
            file.write("(token, gold_label, pred_label)\n")
            for i in range(len(sample_input)):
                if sample_gold_tags[i] != sample_pred_tags[i]:
                    file.write(">>>>> ")
                file.write("(\'{}\', {}, {})\n".format(sample_input[i],
                                                   sample_gold_tags[i],
                                                   sample_pred_tags[i]))



        file.write("\n\n================================================================\n")
        file.write("======================  CORRECT SAMPLES  =======================\n")
        file.write("================================================================\n\n")

        for sample_input, sample_gold_tags, sample_pred_tags in correct_samples:
            file.write("\n\n\n")
            file.write("{}\n".format(" ".join(sample_input)))
            file.write("(token, gold_label, pred_label)\n")
            for i in range(len(sample_input)):
                file.write("(\'{}\', {}, {})\n".format(sample_input[i],
                                                   sample_gold_tags[i],
                                                   sample_pred_tags[i]))

        file.close()
