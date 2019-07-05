def accuracy(device, model, datasets, batch_size=1):
    name2dataset = {d.name:d for d in datasets}

    for d in datasets:
        d.class_correct = 0
        d.class_total = 0

    model.eval()
    for itr in get_batches(datasets, "test", batch_size):
        # Getting vars
        inputs, targets, dataset_name = itr

        # Setting the input and the target (seding to GPU if needed)
        inputs = [[word.to(device) for word in sample] for sample in inputs]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)

        # Feeding the model
        output, max_len = model(inputs, dataset_name)

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
    for itr in get_batches(datasets, "val", batch_size):
        # Getting vars
        inputs, targets, dataset_name = itr

        # Setting the input and the target (seding to GPU if needed)
        inputs = [[word.to(device) for word in sample] for sample in inputs]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)

        # Feeding the model
        output, max_len = model(inputs, dataset_name)

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
    file = open("wrong_samples_{}".format(dataset_name),"w", encoding="utf-8")
    model.eval()
    for itr in get_batches(datasets, "val", batch_size):
        # Getting vars
        inputs, targets, dataset_name = itr

        # Setting the input and the target (seding to GPU if needed)
        inputs = [[word.to(device) for word in sample] for sample in inputs]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)

        # Feeding the model
        output, max_len = model(inputs, dataset_name)

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
