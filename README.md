# TCPOSS
The codes for paper: A Dataset and Experimental Study towards Visual Terrain Classification with OOD Challenges in Real World

The datasets will be later published at [www.poss.pku.edu.cn](http://www.poss.pku.edu.cn).

Sample images of 10 terrain classes of TCPOSS:
![img|center](./img_exm/fig3.png)
## Installation

### Requirements

the version in () is the version recommended, not the only version.
- Python (3.9)
- Pytorch ('1.10.1+cu111')


## Data Preparation

### TCPOSS
```
└── TCPOSS
    ├── 103101/ 
    |   ├── 1698727797/
    |   |   ├── imgs/
    |   |   |   ├── 1698727797038674736.jpg
    |   |   |   ├── 1698727797xxxxxxxxx.jpg
    |   |   |   └── ...
    |   ├── 1698727798/
    |   └── ...
    ├── 103102/
    └── ...
    ├── label.csv
    ├── h5py/
    ├── train_test_split/
    |   ├── train_test_split_easy.xlsx
    |   ├── train_test_split_medium.xlsx
    |   ├── train_test_split_hard.xlsx  
    |   └── makeh5py.py
    └── visialize_data.ipynb
```

## Training

## TODO List
### Release Data
- [ ] Make a brief video of TCPOSS
- [ ] Upload data 
- [ ] upload instructions for usage

### Release Codes for Terrain Classification Model
- [ ] Release codes for training (various configurations and 3 backbone)
- [ ] Release codes for evaluating
- [ ] Release codes for plotting Fig.7 T-SNE feature space

### Release Codes for Confidence Estimation Methods
- [ ] Release codes for confidence methods (MCDropout, Ensemble, Mahalanobis Distance, Evidential Deep Learning)
- [ ] Release codes for plotting Fig.8 PR Curve
- [ ] Release codes for plotting Fig.9 Confidence distributions
- [ ] Release codes for Fig.10 Calculating KLConf

### Others
- [ ] Release requirements
