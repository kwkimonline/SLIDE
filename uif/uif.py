import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from train import *
from constraints import fair_penalty
from load_data import *
from utils import *
from updates import *

# torch CPU restriction
torch.set_num_threads(4)

# Model defined

class DNN(nn.Module) :

    def __init__(self, dimension) :

        super(DNN, self).__init__()
        self.dimension = dimension
        self.hidden = 100
        self.fc1 = nn.Linear(self.dimension, self.hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden, 2)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, inputs) :

        outputs = self.fc1(inputs)
        outputs = self.relu(outputs)
        logits = self.fc2(outputs)
        probs = self.softmax(logits)

        return logits, probs

class Linear(nn.Module) :

    def __init__(self, dimension) :

        super(Linear, self).__init__()
        self.dimension = dimension
        self.fc1 = nn.Linear(self.dimension, 2)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, inputs) :

        logits = self.fc1(inputs)
        probs = self.softmax(logits)

        return logits, probs



# Run UIF

def uif(args, proj_compl) :

    print("=================================================================================")

    # fix seed for all
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.set_num_threads(4)
    torch.set_num_threads(10)

    print(f"Dataset: {args.dataset}, Lmda: {args.lmda}, Seed: {args.seed}")

    # when is today?
    from datetime import datetime
    today = datetime.today()
    date = today.strftime("%Y%m%d")

    # device
    device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.cuda("cpu")

    # load data
    if args.dataset == "toy" :
        (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test) = load_toy_dataset()
    elif args.dataset == "adult" :
        (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test) = load_adult_dataset()
    elif args.dataset == "bank" :
        (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test) = load_bank_dataset()
    elif args.dataset == "law" :
        (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test) = load_law_dataset()
    else :
        raise NotImplementedError
        
    # to tensor    
    xs_train, x_train, y_train, s_train = torch.from_numpy(xs_train), torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(s_train)
    xs_test, x_test, y_test, s_test = torch.from_numpy(xs_test), torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(s_test)

    # get altered test datasets
    _, _, xs_train0_, xs_train1_ = flip_sen_datasets(xs_train)
    _, _, xs_test0_, xs_test1_ = flip_sen_datasets(xs_test)

    # define the dimension
    d = xs_train.shape[1]
    assert d == xs_test.shape[1]


    # define model to use
    if args.model_type == "dnn" :
        model = DNN(dimension = d)
        model.to(device)
    elif args.model_type == "linear" :
        model = Linear(dimension = d)
        model.to(device)
    else :
        print("DNN or Linear model only provided")
        raise NotImplementedError

    # define optimizer to use
    if args.opt == "sgd" :
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = 0.1, weight_decay = 0.2)
    elif args.opt == "adam" :
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)            
    else :
        print("only SGD and Adam optimizer to use!")
        raise NotImplementedError
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.5)

    # criterions
    util_criterion = nn.CrossEntropyLoss()
    fair_criterion = fair_penalty(mode = args.mode, gamma = args.gamma, tau = args.tau)

    # define fair metric
    fair_metric = fair_dist(proj_compl, w = 0.)

    # learning
    train_losses = []
    for epoch in tqdm(range(args.epochs), desc = "UIF learning") :

        learning_stats = {"acc" : [], "bacc" : [], "con" : [], "con10": [], "con50": []}

        # train for one epoch
        model.train()
        train_loss, all_actual_epsilons = train_full_batch(model, proj_compl, xs_train, y_train, optimizer, scheduler, args.batch_size, args.lmda, args.vat_iter, args.epsilon, args.norm, args.tau, args.gamma, util_criterion, fair_criterion, device)
        train_losses.append(train_loss)

        # print at each 500 epoch
        if (epoch + 1) % 500 == 0 or epoch == args.epochs - 1:

            print("+"*20)
            print("Epoch: {}".format(epoch + 1))

            # test at the last
            model.eval()
            with torch.no_grad() :

                preds_, all_targets, all_sensitives = test_full_batch(model, xs_test, y_test, s_test, device)
                preds0_ = test_full_batch(model, xs_test0_, y_test, s_test, device, output_label = True)
                preds1_ = test_full_batch(model, xs_test1_, y_test, s_test, device, output_label = True)

                all_inputs = xs_test
                all_preds_ = (preds_, preds0_, preds1_)

                test_perfs = evaluate(all_inputs, all_preds_, all_targets, all_sensitives)

            # print at cmd
            print("Dataset : {}, Mode : {}, Lambda : {}, Gamma : {}, Tau : {}".format(args.dataset, args.mode, 
                                                                                      args.lmda, args.gamma, args.tau))
            print("(acc,  bacc,  con, con10, con50)")
            print(test_perfs)

    # update learning stats
    learning_stats = update_perfs(test_perfs, learning_stats)

    # save performances
    options = "results/{}/".format(args.dataset)
    file_name = options + "perfs_lr{}_epochs{}_opt{}_arch{}.csv".format(args.lr, args.epochs, args.opt, args.model_type)
    if not os.path.exists(args.result_path + options) :
        os.makedirs(args.result_path + options)    
    write_perfs(args.result_path, file_name, args.mode, args.lr, args.epochs, args.opt, args.lmda, args.gamma, args.tau, learning_stats)




################################
       
def sensr_proj(args) :

    # fix seed for all
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # when is today?
    from datetime import datetime
    today = datetime.today()
    date = today.strftime("%Y%m%d")

    # device
    device = torch.device("cuda:{}".format(args.device)) if torch.cuda.is_available() else torch.cuda("cpu")

    
    # load data
    if args.dataset == "toy" :
        (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test) = load_toy_dataset()
    elif args.dataset == "adult" :
        (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test) = load_adult_dataset()
    elif args.dataset == "bank" :
        (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test) = load_bank_dataset()
    elif args.dataset == "law" :
        (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test) = load_law_dataset()
    else :
        raise NotImplementedError
        
    # to tensor    
    xs_train, x_train, y_train, s_train = torch.from_numpy(xs_train), torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(s_train)
    xs_test, x_test, y_test, s_test = torch.from_numpy(xs_test), torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(s_test)
    
    
    from copy import deepcopy
    sen_idx = xs_train.shape[1] - 1
    zeroed_xs_train = deepcopy(xs_train)
    zeroed_xs_train[:, sen_idx] = 0.
    direction_dataset = TensorDataset(zeroed_xs_train, s_train)
    batch_size = len(direction_dataset)
    direction_loader = DataLoader(direction_dataset, batch_size = batch_size)
    semi_dimension = zeroed_xs_train.shape[1]
    

    # learning
    sensitive_directions = get_sensitive_directions_full_batch(zeroed_xs_train, s_train, semi_dimension, dataset = args.dataset, device = args.device)
    
    # projection complement
    proj_compl = compl_svd_projector(sensitive_directions.cpu().numpy())
    if not os.path.exists("proj_compls/"):
        os.makedirs("proj_compls/")
    np.save("proj_compls/{}_proj_compl_{}.npy".format(args.dataset, date), proj_compl)

    return proj_compl