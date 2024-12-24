# **SLIDE**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## **Overview**
This repository contains the official PyTorch implementation of **SLIDE**, the algorithm proposed in the paper:  
["SLIDE: A surrogate fairness constraint to ensure fairness consistency"](https://www.sciencedirect.com/science/article/pii/S0893608022002891) by *Kunwoong Kim, Ilsang Ohn, Sara Kim, and Yongdai Kim.*

The paper was published in **[Neural Networks](https://www.journals.elsevier.com/neural-networks)** (Volume 154, 2022, Pages 441-454).

---

## **Usage**

### **Steps to Run SLIDE**
1. **Prepare dataset**
   Place your custom dataset in the directory:  
   ```bash
   datasets/{custom_dataset}

2. **Add a data loading function**
   In the file "load.data_py", implement a function that returns a tuple: ```(xs, x, y, s)``` where
   - ```x```: input features (```torch.tensor```)
   - ```y```: target labels (```torch.tensor```)
   - ```s```: sensitive attributes (```torch.tensor```)
   - ```xs```: the concatenation of ```x``` and ```s``` (```torch.tensor```) = ```xs = torch.cat([x, s.reshape(s.size(0), 1)], dim=1).```

3. **Run SLIDE**
   Using the following command: ```python main.py --dataset {custom_dataset} --lmda {lmda}```
   - ```lmda``` is the hyper-parameter (higher ```lmda```, higher fairness level).

### Example Command:

An example of the command using the law school dataset is
```python
python main.py --dataset law --lmda 5.0
```

### Environments

- Python >= 3.6
- torch >= 1.8.0

## Citation

```
@article{KIM2022441,
  title = {SLIDE: A surrogate fairness constraint to ensure fairness consistency},
  journal = {Neural Networks},
  volume = {154},
  pages = {441-454},
  year = {2022},
  issn = {0893-6080},
  doi = {https://doi.org/10.1016/j.neunet.2022.07.027},
  url = {https://www.sciencedirect.com/science/article/pii/S0893608022002891},
  author = {Kunwoong Kim and Ilsang Ohn and Sara Kim and Yongdai Kim}
}
```
