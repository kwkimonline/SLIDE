import torch
import torch.nn as nn
import torch.distributions.dirichlet as dirichlet
import torch.nn.functional as F
import contextlib
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from utils import *
from constraints import true_nu


def get_sensitive_directions_full_batch(inputs, sensitives, semi_dimension, dataset, device) :

    # logistic model
    class sensitive_net(nn.Module) :

        def __init__(self, dimension) :
            super(sensitive_net, self).__init__()
            self.dimension = dimension
            self.fc1 = nn.Linear(self.dimension, 1, bias = False)
            
        def forward(self, inputs) :
            logits = self.fc1(inputs)
            probs = torch.sigmoid(logits)
            return logits, probs

    model = sensitive_net(dimension = semi_dimension).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.0)
    criterion = nn.BCELoss()

    loss_new = 1e+5
    for _ in tqdm(range(5000), desc = "Computing SenSR") :

        loss_old = loss_new
        loss_new = 0.0

        inputs, sensitives = inputs.float().to(device), sensitives.float().to(device)
        _, probs = model(inputs)
        sensitives = sensitives.reshape(probs.shape)
        loss = criterion(probs, sensitives)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_new += loss.item()

        if abs(loss_old - loss_new) <= 1e-10 : # convergence?
            print("Algorithm converged.")
            break

    sensitive_directions = None
        
    # get sensitive directions
    sen_onehot = torch.zeros(model.dimension)
    sen_idx = inputs.shape[1] - 1
    sen_onehot[sen_idx] = 1.0 # d dimensional

    for _, param in model.named_parameters() :
        if param.requires_grad :
            weights = param.data.flatten()
    sensitive_directions = weights # d dimensional

    # reshaping
    sensitive_directions = sensitive_directions.reshape(1, sensitive_directions.shape[0])
    sen_onehot = sen_onehot.reshape(1, sen_onehot.shape[0]).to(device = sensitive_directions.device)

    sensitive_directions = torch.cat((sensitive_directions, sen_onehot))

    return sensitive_directions





def train_full_batch(model, proj_compl, inputs, targets, optimizer, scheduler, batch_size, lmda, vat_iter, epsilon, norm, tau, gamma, util_criterion, fair_criterion, device) :
    
    """ train for 1 epoch """

    all_actual_epsilons = []

    inputs, targets = inputs.float().to(device), targets.to(device)

    # feed forwarding
    preds, probs = model(inputs)
    assert probs.shape[0] == inputs.shape[0]
    assert probs.shape[1] == 2

    # get criterions, compute losses
    loss = util_criterion(preds, targets)
    
    if lmda > 0.0 :
        # generate adversarial inputs
        adv_inputs, actual_epsilons = get_adv_inputs(model, inputs, proj_compl, epsilon, num_iter = vat_iter, norm = "two", y_diff = "nokl")
        all_actual_epsilons.append(actual_epsilons)
        adv_inputs = adv_inputs.to(device)

        # compute the probabilities
        pn = get_pn(model, inputs, adv_inputs, targets, device)
        fair_loss = fair_criterion(pn, tau, gamma)

        # epsilons to tensor
        all_actual_epsilons = torch.cat(all_actual_epsilons)

        # GAIF loss
        loss += lmda * fair_loss

    else :
        all_actual_epsilons = torch.tensor([0.0])
    
    # update step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # save losses
    train_loss = loss.item()

    return train_loss, all_actual_epsilons




def test_full_batch(model, inputs, targets, sensitives, device, output_label = False) :

    """ test for 1 epoch """

    with torch.no_grad() :

        inputs, targets = inputs.float().to(device), targets.to(device)
        preds, probs = model(inputs)
        
        pred_targets = preds.argmax(dim = 1)
        assert (preds.argmax(dim = 1) == probs.argmax(dim = 1)).sum().bool()

    if output_label :
        return pred_targets
    else :
        return preds, targets, sensitives


######### evaluate
    

def evaluate(inputs, all_preds_, targets, sensitives) :

    # get all preds
    preds, preds1_, preds2_ = all_preds_

    # compute performances
    preds, preds1_, preds2_ = preds.cpu(), preds1_.cpu(), preds2_.cpu()
    targets, sensitives = targets.cpu(), sensitives.cpu()

    acc, bacc = util_perf(preds, targets, sensitives)
    con = indifair_perf(preds1_, preds2_)
    con10 = indifair_original_perf(inputs, preds, k = 10)
    con50 = indifair_original_perf(inputs, preds, k = 50)
    
    perfs = (acc, bacc, con, con10, con50)   

    return perfs


def util_perf(preds, targets, sensitives) :

    # accuracy
    pred_targets = preds.argmax(dim = 1)
    probs = F.softmax(preds, dim = 1)
    acc = (pred_targets == targets).float().mean() 

    acc_g1_ = (preds[targets == 0].argmax(dim = 1) == targets[targets == 0]).float().mean()
    acc_g2_ = (preds[targets == 1].argmax(dim = 1) == targets[targets == 1]).float().mean()
    bacc = (acc_g1_ + acc_g2_) / 2.0

    return round(acc.item(), 4), round(bacc.item(), 4)


def indifair_perf(preds1_, preds2_) :

    con = torch.FloatTensor([1 if preds1_[i] == preds2_[i] else 0 for i in range(len(preds1_))]).mean()

    return round(con.item(), 4)


def indifair_original_perf(inputs, preds, k = 10):
    
    inputs = inputs.cpu().numpy()
    preds = preds.cpu().numpy().argmax(1)
    
    nbrs = NearestNeighbors(n_neighbors = k + 1, algorithm = "ball_tree").fit(inputs)
    _, indices = nbrs.kneighbors(inputs)
    
    nbr_preds = preds[indices]
    
    con_preds = (nbr_preds[:, 1:] == nbr_preds[:, 0].reshape(nbr_preds.shape[0], 1)) * 1
    con_preds = con_preds.mean(1)
    con_preds = con_preds.mean()
    
    return con_preds
    