# SLIDE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official pytorch implementation of ["SLIDE: A surrogate fairness constraint to ensure fairness consistency"](https://www.sciencedirect.com/science/article/pii/S0893608022002891) published in [Neural Networks](https://www.journals.elsevier.com/neural-networks) (Volume 154, 2022, Pages 441-454) by Kunwoong Kim, Ilsang Ohn, Sara Kim, and Yongdai Kim.


## Usage

1. Locate your custom_dataset in the directory "datasets/{custom_dataset}".

2. Add a data loading function in "load.data_py" returning a tuple ```(xs, x, y, s)``` consisting four variables of type ``` torch.tensor ```
where ```xs = torch.cat([x, s.reshape(s.size(0), 1)], dim=1).```

3. Run SLIDE using: "python main.py --dataset {custom_dataset} --lmda {lmda}"
where lmda is the hyper-parameter (higher lmda, higher fairness level).

An example of the command using the law school dataset is
```python
python main.py --dataset law --lmda 5.0
```

## Environments

python >= 3.6
torch >= 1.8.0
