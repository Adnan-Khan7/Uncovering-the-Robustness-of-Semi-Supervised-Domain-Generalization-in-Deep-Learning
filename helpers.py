import torch.nn as nn
import torch
import numpy as np
import random
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import shutil

def get_monte_carlo_predictions(input_s,target_s,
                                forward_passes,
                                model,
                                n_classes,
                                n_samples):
    
    
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """
            
    tanh = nn.Tanh()

    #define fc layer with a dropout
    fc_layer = nn.Sequential(
                nn.Dropout(0.2),
                model.fc
            )


    # change device
    fc_layer.cuda()
    input_s = input_s.cuda()
    
    # take out the feature extractor, neglecting the final fc layer
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    # get the feature scores for input data
    y_hat = feature_extractor(input_s).squeeze().squeeze()
    
    all_predictions = []

    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        with torch.no_grad():
            output = fc_layer(y_hat)

        output = output.unsqueeze(1)
        all_predictions.append(output)
    
    dropout_predictions = torch.cat(all_predictions, dim = 1)
    dropout_predictions = dropout_predictions.cpu()
    dropout_predictions = dropout_predictions.numpy()

    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=1) # shape (n_samples, n_classes)

    #standard deviation
    #std = np.std(dropout_predictions, axis=1) # shape (n_samples, n_classes)
    
    #Variance
    var = np.var(dropout_predictions, axis=1) # shape (n_samples, n_classes)


    # std = torch.from_numpy(std)
    # std = 1 - tanh(std)

    var = torch.from_numpy(var)
    var = 1 - tanh(var)
    
    return var
    

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def save_checkpoint(state, is_best,is_best_valid, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
    if is_best_valid:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best_valid.pth.tar'))

# def save_checkpoint(state, is_best,is_best_valid, checkpoint, filename='checkpoint_{}.pth.tar'):
#     if not os.path.exists(checkpoint):
#         os.makedirs(checkpoint)
        
#     filepath = os.path.join(checkpoint, filename.format(state['epoch']))
#     torch.save(state, filepath)
#     if is_best:
#         shutil.copyfile(filepath, os.path.join(checkpoint,
#                                                'model_best.pth.tar'))
#     if is_best_valid:
#         shutil.copyfile(filepath, os.path.join(checkpoint,
#                                                'model_best_valid.pth.tar'))
