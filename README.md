# SLIDE

This is the official implementation of "SLIDE: A surrogate fairness constraint to ensure fairness consistency".
This source codes can learn (group) fair classifier with any custom dataset.

## Usage

1. Locate your custom_dataset in the directory "datasets/{custom_dataset}".

2. Add loading function in "load.data_py" that should return (xs, x, y, s)
where xs = torch.cat([x, s.reshape(s.size(0), 1)], dim=1).

3. run SLIDE as the command: "python main.py --dataset {custom_dataset} --lmda {lmda}"
where lmda is the fairness hyper-parameter, higher lmda increases the level of fairness (demographic parity or disparate impact).

## Environments

These codes are based on the following environments and versions of the corresponding libraries.

python >= 3.6

torch >= 1.8.0
