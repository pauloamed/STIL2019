import torch
from torch import nn
import time
from pos_tagger.utils import get_batches

def train(device, model, optimizer, datasets, min_val_loss, state_dict_path, epochs, training_policy, batch_size, clip=20):

    name2dataset = {d.name:d for d in datasets}

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        inicio = time.time()

        for d in datasets:
            d.train_loss = d.val_loss = 0.0

        model.train()
        for itr in get_batches(datasets, "train", batch_size, training_policy):
            # Getting vars
            inputs, targets, dataset_name = itr

            # Setting the input and the target (seding to GPU if needed)
            inputs = [[word.to(device) for word in sample] for sample in inputs]
            targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)

            # Feeding the model
            output = model(inputs)

            # Reseting the gradients
            optimizer.zero_grad()

            # Calculating the loss and the gradients
            loss = criterion(output[dataset_name].view(batch_size*output["length"], -1),
                             targets.view(batch_size*output["length"]))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Adjusting the weights
            optimizer.step()

            # Updating the train loss
            name2dataset[dataset_name].train_loss += loss.item() * batch_size


        model.eval()
        for itr in get_batches(datasets, "val"):
            # Getting vars
            inputs, targets, dataset_name = itr

            # Setting the input and the target (seding to GPU if needed)
            inputs = [[word.to(device) for word in sample] for sample in inputs]
            targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)

            # Feeding the model
            output = model(inputs)

            # Calculating the loss and the gradients
            loss = criterion(output[dataset_name].view(output["length"], -1),
                             targets.view(output["length"]))

            # Updating the loss accu
            name2dataset[dataset_name].val_loss += loss.item() * batch_size

        # Normalizing the losses
        for i in range(len(datasets)):
            if datasets[i].use_train:
                datasets[i].train_loss /= datasets[i].sent_train_size
            if datasets[i].use_val:
                datasets[i].val_loss /= datasets[i].sent_val_size

        # Verbose
        print("\n=======================================================================================")
        current_lr = optimizer.param_groups[0]['lr']
        total_train_loss = sum([d.train_loss for d in datasets if d.use_train])
        total_val_loss = sum([d.val_loss for d in datasets if d.use_val]))
        duration = time.time()-inicio
        print('Epoch: {} \t Learning Rate: {:.3f}\tTotal Training Loss: {:.6f} \tTotal Validation Loss: {:.6f} \t Duration: {:.3f}'.format(
            epoch, current_lr, total_train_loss, total_val_loss, duration)

        for d in datasets:
            if d.use_train and d.use_val:
                print('>> Dataset {}:\tTraining Loss: {:.6f}\tValidation Loss:{:.6f}'.format(d.name, d.train_loss, d.val_loss))
            elif d.use_train and not d.use_val:
                print('>> Dataset {}:\tTraining Loss: {:.6f}'.format(d.name, d.train_loss))
            elif not d.use_train and d.use_val:
                print('>> Dataset {}:\tValidation Loss: {:.6f}'.format(d.name, d.val_loss))

        print("----------------------------------------------------------------------------------------")

        # Saving the best model
        print('Comparing loss on {} dataset(s)'.format([d.name for d in datasets if d.use_val]))

        if total_val_loss <= min_val_loss:
            torch.save(model.state_dict(), state_dict_path)
            min_val_loss = total_val_loss
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                                                                                min_val_loss,
                                                                                compare_val_loss))
        print("=======================================================================================")



    return model, min_val_loss
