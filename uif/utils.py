import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
import contextlib
from tqdm import tqdm


""" VAT pytorch ver. from https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py """

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VAT(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1, y_diff="nokl", q=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        :param y_diff: measure of ys
        """
        super(VAT, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.y_diff = y_diff
        self.q = q

    def forward(self, model, x):
        with torch.no_grad():
            _, probs = model(x)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        # with _disable_tracking_bn_stats(model):
        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()
            adv_preds, adv_probs = model(x + self.xi * d)
            if self.y_diff == "nokl" :
                y_measure = d_y(q = self.q)
                adv_distance = y_measure(adv_probs, probs)
                adv_distance = adv_distance.mean()
            else :
                log_adv_preds = F.log_softmax(adv_preds, dim = 1)
                adv_distance = F.kl_div(log_adv_preds, probs, reduction = "batchmean")
            adv_distance.backward()
            d = _l2_normalize(d.grad)
            model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps

        return r_adv

def compl_svd_projector(names, svd=-1):
    if svd > 0:
        tSVD = TruncatedSVD(n_components=svd)
        tSVD.fit(names)
        basis = tSVD.components_.T
        print('Singular values:')
        print(tSVD.singular_values_)
    else:
        basis = names.T

    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj_compl = np.eye(proj.shape[0]) - proj
    return proj_compl

def fair_dist(proj, w=0.):

    if proj is not None :

        pt_proj = torch.from_numpy(proj).float()
        if w > 0 :
            return lambda x, y: (torch.mm(x - y, pt_proj)**2).sum(axis = 1) + w * (torch.mm(x - y, torch.eye(proj.shape[0]) - pt_proj)**2).sum(axis = 1)
        else :
            return lambda x, y: (torch.mm(x - y, pt_proj)**2).sum(dim = 1)

    else :
        return None


def sensr_metric(x, y, proj):

    pt_proj = torch.from_numpy(proj).float()
    return (torch.mm(x - y, pt_proj)**2).sum(dim = 1)

def d_y(q = 2) :

    if q == 2 :
        return lambda x, y : ((x - y)**2).sum(dim = 1) / 2
    else :
        return lambda x, y : (( (abs(x - y) + 1e-8) **q).mean(dim = 1))**(1/q)
    
def get_adv_inputs(model, inputs, proj_compl, epsilon, num_iter = 1, norm = "two", y_diff = "nokl", q = 1) :

    vat = VAT(xi = 10.0, eps = epsilon, ip = num_iter, y_diff = y_diff, q = q)
    inputs_clone = inputs.clone()
    adv_directions = vat(model, x = inputs_clone)
    
    with torch.no_grad() :
        adv_inputs = inputs_clone.cpu() + adv_directions.cpu()
        actual_epsilons = sensr_metric(inputs_clone.cpu(), adv_inputs, proj_compl)

    return adv_inputs, actual_epsilons


def get_pn(model, x, x_adv, targets, device) :

    x, x_adv = x.float(), x_adv.float()
    dist_y = d_y(q = 2)
    
    _, probs = model(x)
    _, adv_probs = model(x_adv)

    pn = dist_y(probs, adv_probs)

    return pn
