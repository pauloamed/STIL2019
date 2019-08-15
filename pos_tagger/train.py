import torch
import numpy as np
import time
from pos_tagger.utils import get_batches, send_output
from pos_tagger.parameters import STATE_DICT_PATH, EPOCHS, BATCH_SIZE, GRADIENT_CLIPPING


def train(device, model, datasets, min_val_loss=np.inf):

    # optimizer and loss function
    optimizer = torch.optim.Adadelta(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    name2dataset = {d.name:d for d in datasets}


    for epoch in range(EPOCHS):
        inicio = time.time()

        for d in datasets:
            d.train_loss = d.val_loss = 0.0

        model.train()
        for itr in get_batches(datasets, "train", BATCH_SIZE, "visconde"):
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
            loss = criterion(output[dataset_name].view(BATCH_SIZE*output["length"], -1),
                             targets.view(BATCH_SIZE*output["length"]))
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)

            # Adjusting the weights
            optimizer.step()

            # Updating the train loss
            name2dataset[dataset_name].train_loss += loss.item() * BATCH_SIZE


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
            name2dataset[dataset_name].val_loss += loss.item()

        # Normalizing the losses
        for i in range(len(datasets)):
            if datasets[i].use_train:
                datasets[i].train_loss /= datasets[i].sent_train_size
            if datasets[i].use_val:
                datasets[i].val_loss /= datasets[i].sent_val_size

        # Verbose
        out_str = "\n======================================================================================="
        current_lr = optimizer.param_groups[0]['lr']
        total_train_loss = sum([d.train_loss for d in datasets if d.use_train])
        total_val_loss = sum([d.val_loss for d in datasets if d.use_val])
        duration = time.time()-inicio
        out_str += ("Epoch: {} \t Learning Rate: {:.3f}\tTotal Training Loss: {:.6f} \tTotal Validation Loss: {:.6f} \t Duration: {:.3f}\n".format(
            epoch, current_lr, total_train_loss, total_val_loss, duration))

        for d in datasets:
            if d.use_train and d.use_val:
                out_str += ('>> Dataset {}:\tTraining Loss: {:.6f}\tValidation Loss:{:.6f}\n'.format(d.name, d.train_loss, d.val_loss))
            elif d.use_train and not d.use_val:
                out_str += ('>> Dataset {}:\tTraining Loss: {:.6f}\n'.format(d.name, d.train_loss))
            elif not d.use_train and d.use_val:
                out_str +=('>> Dataset {}:\tValidation Loss: {:.6f}\n'.format(d.name, d.val_loss))

        out_str += ("----------------------------------------------------------------------------------------\n")

        # Saving the best model
        out_str += ('Comparing loss on {} dataset(s)\n'.format([d.name for d in datasets if d.use_val]))

        if total_val_loss <= min_val_loss:
            torch.save(model.state_dict(), STATE_DICT_PATH)
            out_str += ('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(
                                                                                min_val_loss,
                                                                                total_val_loss))
            min_val_loss = total_val_loss
        out_str += ("=======================================================================================\n")

        send_output(out_str, 0)

    return model, min_val_loss
