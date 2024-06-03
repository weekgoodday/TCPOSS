
'''
(2016 ICML) Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. Yarin Gal, Zoubin Ghahramani. University of Cambridge.
'''
import torch
from torch import nn
import h5py
from torchvision import transforms
from dataset import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np 
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
# revise the model and data path
model_path = "./model_used/mobilenet_easy.pth"
test_datapath = "./data_used/test_easy.hdf5"

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
def h(a): # just *, two cubes multiply accordingly 
    '''
    a.shape:[N,num_class]
    b.shape:[N,]
    '''
    b=np.zeros(a.shape[0])
    b=-np.sum(a*np.log(a+1e-50),axis=1) # in case of log(0) nan
    return b
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
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
print("testset:")
test_data = load_data(test_datapath)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
gt = np.empty((0,), dtype=np.int32)
pd = np.empty((0,), dtype=np.int32)
with torch.no_grad():
    for X,y in test_dataloader:
        imgs = X.to(device)
        label = y
        yhat = model(imgs)
        pred = torch.argmax(yhat, dim=1)
        gt = np.append(gt, label.cpu().numpy())
        pd = np.append(pd, pred.cpu().numpy())
# open dropout to get uncertainty
model.apply(apply_dropout)
T=5
with torch.no_grad():
    for i in range(T):
        probs_ = np.empty((0,n_classes),dtype=np.float32)
        for X,y in test_dataloader:
            imgs = X.to(device)
            label = y
            yhat = model(imgs)
            probs_t = F.softmax(yhat, dim=1)
            probs_ = np.append(probs_, probs_t.cpu().numpy(), axis=0)
        if(i==0):
            probs = probs_[np.newaxis,:,:]
        else:
            probs = np.append(probs, probs_[np.newaxis,:,:], axis=0)    
tu = h(np.mean(probs, axis=0))  # total uncertainty
datasize = tu.shape[0]
au_ = np.empty((T,datasize))  
for i in range(T):
    au_[i] = h(probs[i,:,:])
au = np.mean(au_, axis=0) # aleatoric uncertainty
eu = tu-au  # epstemic uncertainty

# Plot remained acc vs confidence threshold
# tu should be sorted from small to large, here we use total negative total uncertainty to represent confidence
index_conf_tu = np.argsort(tu)  # get original index after sorting from small to large
rank_conf_tu = np.argsort(index_conf_tu)  # get every number's position after sorting from small to large
intervals = 50
acc_record = np.zeros((intervals,))
for i in range(1, intervals+1):
    masks = (rank_conf_tu < datasize*i/intervals)
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
