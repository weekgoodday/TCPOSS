'''
(2017 ICLR) A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. Hendrycks et al.
'''
import torch
from torch import nn
import h5py
from torchvision import transforms
from dataset import ImageDataset
from torch.utils.data import DataLoader
import numpy as np 
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
# revise the model and data path
model_path = "./model_used/mobilenet_easy.pth"
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
model.load_state_dict(torch.load(model_path))

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
# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
gt = np.empty((0,),dtype=np.int32)
pd = np.empty((0,),dtype=np.int32)
probs = np.empty((0,n_classes),dtype=np.float32)
with torch.no_grad():
    for X,y in test_dataloader:
        imgs = X.to(device)
        label = y
        yhat = model(imgs)
        pred = torch.argmax(yhat, dim=1)
        prob = F.softmax(yhat, dim=1)
        gt = np.append(gt, label.cpu().numpy())
        pd = np.append(pd, pred.cpu().numpy())
        probs = np.append(probs, prob.cpu().numpy(), axis=0)
conf_sm = np.max(probs, axis=1)
datasize = conf_sm.shape[0]

# Confidence should be sorted from large to small
index_conf_sm = np.argsort(-conf_sm)  # get original index after sorting from large to small
rank_conf_sm = np.argsort(index_conf_sm)  # get every number's position after sorting from large to small
intervals = 50
acc_record = np.zeros((intervals,))
for i in range(1, intervals+1):
    masks = (rank_conf_sm < datasize*i/intervals)
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
