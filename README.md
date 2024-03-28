# A Dataset and Experimental Study towards Visual Terrain Classification with OOD Challenges in Real World
The codes of our work **A Dataset and Experimental Study towards Visual Terrain Classification with OOD Challenges in Real World**

The dataset TCPOSS will be later published at [www.poss.pku.edu.cn](http://www.poss.pku.edu.cn).

Sample images of 10 terrain classes of TCPOSS:
![img|center](./img_exm/fig3.png)
## Installation

### Requirements

the version in () is the version recommended, not the only version.
- Python (3.9)
- Pytorch ('1.10.1+cu111')
- pandas (2.2.1)
- cv2 (opencv-python 4.5.4.60)
- sklearn (scikit-learn 1.3.2)
- matplotlib (3.8.3)
- openpyxl (3.1.2)
- tqdm (4.66.2)

## Data Preparation

### TCPOSS
```
└── TCPOSS_Data
    ├── 103101/ 
    |   ├── 1698727798/
    |   |   ├── imgs/
    |   |   |   ├── 1698727798037949165.jpg
    |   |   |   ├── 1698727798xxxxxxxxx.jpg
    |   |   |   └── ...
    |   ├── 1698727799/
    |   ├── ...
    |   └── valid.txt
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
The detailed description of the command line parameters please refer to readme_code.txt.

For trail, using mobilnet backbone and four levels of dataset, just run:

```
sh start_train.sh
```

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
