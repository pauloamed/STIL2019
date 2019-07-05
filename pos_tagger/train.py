import torch
from torch import nn
import sys
import random

import pos_tagger.utils as utils


def train(device, model, optimizer, datasets, min_val_loss, state_dict_path, epochs, training_policy, batch_size, clip=5):

    name2dataset = {d.name:d for d in datasets}

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        for d in datasets:
            d.train_loss = d.val_loss = 0.0

        model.train()
        for itr in utils.get_batches(datasets, "train", batch_size, training_policy):
            # Getting vars
            inputs, targets, dataset_name, batch_length = itr

            # Setting the input and the target
            targets = torch.LongTensor(targets).to(device)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in model.init_hidden(batch_size)])

            # Running through the model
            output, _ = model(inputs, h, dataset_name)

            # Reseting the gradients
            optimizer.zero_grad()

            # Calculating the loss and the gradients
            loss = criterion(output, targets.view(batch_size*batch_length))
            loss.backward()

            # Adjusting the weights
            optimizer.step()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Updating the train loss
            name2dataset[dataset_name].train_loss += loss.item() * batch_size


        model.eval()
        for itr in utils.get_batches(datasets, "val", batch_size):

            # Getting vars
            inputs, targets, dataset_name, batch_length = itr

            # Setting the input and the target
            targets = torch.LongTensor(targets).to(device)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in model.init_hidden(batch_size)])

            # Calculating the output
            output, h = model(inputs, h, dataset_name)

            # Reseting the gradients
            optimizer.zero_grad()

            # Calculating the loss
            loss = criterion(output, targets.view(batch_size*batch_length))

            # Updating the loss accu
            name2dataset[dataset_name].val_loss += loss.item() * batch_size

        # Normalizing the losses
        for i in range(len(datasets)):
            if datasets[i].use_train:
                datasets[i].train_loss /= datasets[i].sent_train_size
            if datasets[i].use_val:
                datasets[i].val_loss /= datasets[i].sent_val_size

        # Verbose
        print('Epoch: {} \t Learning Rate: {:.3f}\tTotal Training Loss: {:.6f} \tTotal Validation Loss: {:.6f}'.format(
            epoch, optimizer.param_groups[0]['lr'], sum([d.train_loss for d in datasets if d.use_train]),
            sum([d.val_loss for d in datasets if d.use_val])))

        for d in datasets:
            if d.use_train and d.use_val:
                print('>> Dataset {}:\tTraining Loss: {:.6f}\tValidation Loss:{:.6f}'.format(d.name, d.train_loss, d.val_loss))
            elif d.use_train and not d.use_val:
                print('>> Dataset {}:\tTraining Loss: {:.6f}'.format(d.name, d.train_loss))
            elif not d.use_train and d.use_val:
                print('>> Dataset {}:\tValidation Loss: {:.6f}'.format(d.name, d.val_loss))


        # Saving the best model
        compare_val_loss = sum([d.val_loss for d in datasets if d.use_val])
        print('Comparing loss on {} dataset(s)'.format([d.name for d in datasets if d.use_val]))

        if compare_val_loss <= min_val_loss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                                                                                min_val_loss,
                                                                                compare_val_loss))
            torch.save(model.state_dict(), state_dict_path)
            min_val_loss = compare_val_loss


        print("=======================================================================================")

    return model, min_val_loss
