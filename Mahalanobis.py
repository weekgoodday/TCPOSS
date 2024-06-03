'''
(2018 NIPS) A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. Lee et al.

Attention that use 1280-dimensional feature, the covariance matrix is most likely to be a singular matrix,
so we use PCA to reduce the dimension of feature space, and then calculate the precision.
We also find use a covariance matrix for each class is better than use a total covariance matrix,
so we use a precision matrix for each class with PCA containing 99% feature, rather than a total precision matrix.
'''

import torch
from torch import nn
import h5py
from torchvision import transforms
from dataset import ImageDataset
from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# revise the model and data path
model_path = "./model_used/mobilenet_easy.pth"
train_datapath = "./data_used/train_easy.hdf5"
test_datapath = "./data_used/test_easy.hdf5"

#hyperparameters
batch_size = 64
n_classes = 10

# model architecture
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2')
classifier = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(1280, 32),
    nn.Dropout(0.25),
    nn.ReLU(),
    nn.Linear(32, n_classes),
)
model.classifier = classifier
feature_hook = []
def hook(module, input, output):
    feature_hook.append(input[0].cpu().numpy())

model.load_state_dict(torch.load(model_path))
model.eval()

# prepare data
def load_data(datapath):
    f = h5py.File(datapath, 'r')
    labels = f['labels']['labels'][:]
    images = f['images']['images'][:]
    print(images.shape,"# of images")
    print(labels.shape,"# of labels")
    f.close()
    images = images.reshape(images.shape[0], 224, 224, 3)/255.0 # normalize to 0-1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # normilize helps a little
    ])
    data = ImageDataset(labels, images, transform=transform)
    return data
print("trainset:")
train_data = load_data(train_datapath)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
model.classifier[1].register_forward_hook(hook)  # 1280-dimensional feature is most likely to be a singular matrix, because the number of samples in a certain category may < 1280
gt = np.empty((0,),dtype=np.int32)
pd = np.empty((0,),dtype=np.int32)
with torch.no_grad():
    for X,y in train_dataloader:
        X = X.to(device)
        yhat = model(X)
        gt = np.concatenate((gt,y.cpu().numpy()))
        pd = np.concatenate((pd,torch.argmax(yhat, dim=1).cpu().numpy()))
feature_space = np.empty((0,1280),dtype=np.double)
for feature_batch in feature_hook:
    feature_space = np.concatenate((feature_space, feature_batch), axis=0)
feature_dim = [] # len:10
feature_mean = [] # len:10 (feature_dim[c],)
# feature_covariance_inverse = [] # len:10 (1280,1280) # maybe not inversable use pinv   sklearn.covariance.EmpiricalCovariance
feature_precision = [] # len:10 (feature_dim[c],feature_dim[c])
# group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
pca_list = []  # n_classes, PCA
for c in range(n_classes):
    masks = (gt == c)
    feature_space_c = np.array(feature_space)[masks]  # Nc,1280
    print(feature_space_c.shape)
    pca = PCA(n_components=0.99)
    pca.fit(feature_space_c)
    feature_space_c_pca = pca.transform(feature_space_c)
    pca_list.append(pca)
    feature_dim_c = feature_space_c_pca.shape[1]
    feature_dim.append(feature_dim_c)
    feature_mean_c_pca = np.mean(feature_space_c_pca, axis=0)
    feature_mean.append(feature_mean_c_pca)
    feature_covariance_c_pca = 1/(feature_space_c_pca.shape[0]) * np.dot((feature_space_c_pca-feature_mean_c_pca).T, (feature_space_c_pca-feature_mean_c_pca))
    feature_precision_c_pca = np.linalg.inv(feature_covariance_c_pca)
    feature_precision.append(feature_precision_c_pca)

# test set
print("testset:")
test_data = load_data(test_datapath)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
# clear hook
feature_hook = []
Mdis_conf = []
pd = np.empty((0,),dtype=np.int32)
gt = np.empty((0,),dtype=np.int32)
with torch.no_grad():
    for X,y in test_dataloader:
        X = X.to(device)
        yhat = model(X)
        pred = torch.argmax(yhat, dim=1)
        gt = np.concatenate((gt,y.cpu().numpy()))
        pd = np.concatenate((pd,pred.cpu().numpy()))
feature_space = np.empty((0,1280),dtype=np.double)
for feature_batch in feature_hook:
    feature_space = np.concatenate((feature_space, feature_batch), axis=0)
Mdis_conf = np.empty((0,),dtype=np.float32)
for index in range(feature_space.shape[0]):
    feature_space_pca = pca_list[pd[index]].transform(feature_space[index].reshape(1,-1))
    Mdis_conf = np.append(Mdis_conf, (feature_space_pca-feature_mean[pd[index]])@feature_precision[pd[index]]@((feature_space_pca-feature_mean[pd[index]]).T))    

# Plot remained acc vs confidence threshold
# Mdis should be sorted from small to large, here we use negative Mahalanobis distance to represent confidence
index_conf_Mdis = np.argsort(Mdis_conf)  # get original index after sorting from small to large
rank_conf_Mdis = np.argsort(index_conf_Mdis)  # get every number's position after sorting from small to large
datasize = len(Mdis_conf)
intervals = 50
acc_record = np.zeros((intervals,))
for i in range(1, intervals+1):
    masks = (rank_conf_Mdis < datasize*i/intervals)
    acc = np.sum(gt[masks]==pd[masks])/np.sum(masks)
    print("top",i*100/intervals,"%",acc)
    acc_record[i-1] = acc
reject_acc = np.flip(acc_record)
AUC = np.trapz(reject_acc, dx=1/intervals)
print("Area Under Reject Accuracy curve:", AUC)
# plot reject_rate vs reject_acc 
# x lim 0-1 y lim 0-1
plt.figure()
plt.plot(np.linspace(0,1,intervals+1)[:-1],reject_acc)
plt.xlabel("reject_rate")
plt.ylabel("remained_acc")
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
