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
import time
import pandas
# revise the data and model path
data_type = "entry"  # select from "entry","easy","medium","hard"
test_datapath = "./data_used/test_entry.hdf5" # corresponding dataset (testset) path to data_type
model_type = "mobilenet" # select from "mobilenet", "resnet", "densenet"
model_path = "./model_used/mobilenet_entry.pth" # corresponding model path to data_type

#hyperparameters
batch_size = 64
n_classes = 10
feature_dim = {"mobilenet": 1280, "resnet": 2048, "densenet": 1024}
model_name = {"mobilenet": "mobilenet_v2", "resnet": "resnet50", "densenet": "densenet121"}
data = {'random':np.zeros((5,)), 'easy':np.zeros((5,)), 'medium':np.zeros((5,)), 'hard':np.zeros((5,))}
classes = ['asphalt', 'grass', 'cement', 'board', 'brick', 'gravel', 'sand', 'flagstone', 'plastic', 'soil']
modes = ['random', 'easy', 'medium', 'hard']
# model architecture
model = torch.hub.load('pytorch/vision:v0.10.0', model_name[model_type])
classifier = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(feature_dim[model_type], 32),
    nn.Dropout(0.25),
    nn.ReLU(),
    nn.Linear(32, n_classes),
)
if model_type == 'resnet':
    model.fc = classifier
else:
    model.classifier = classifier
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False
model.load_state_dict(torch.load(model_path))
model.eval()
def load_data(datapath):
    f = h5py.File(datapath, 'r')
    labels = f['labels']['labels'][:]
    images = f['images']['images'][:]
    print(images.shape,"# of images")
    print(labels.shape,"# of labels")
    f.close()
    images = images.reshape(images.shape[0], 224, 224, 3)/255.0
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
loss_fn = nn.CrossEntropyLoss()
# start evaluation
gt = np.empty((0,),dtype=np.int32)
pd = np.empty((0,),dtype=np.int32)
probs = np.empty((0,n_classes),dtype=np.float32)
total_correct = 0
total = 0
start_time = time.time()
with torch.no_grad():
    pbar = tqdm(total=len(test_dataloader))
    for X,y in test_dataloader:
        imgs = X.to(device)
        label = y
        yhat = model(imgs)
        pred = torch.argmax(yhat, dim=1)
        prob = F.softmax(yhat, dim=1)
        gt = np.append(gt, label.numpy())
        pd = np.append(pd, pred.cpu().numpy())
        correct = np.sum(pred.cpu().numpy() == y.numpy())
        total_correct += correct
        total += len(y)
        probs = np.append(probs, prob.cpu().numpy(), axis=0)
        pbar.set_postfix(accuracy=f"{total_correct/total:>7f}")
        pbar.update(1)
    pbar.close()
coding=n_classes*gt+pd  # Utilize unique encoding.
confusion_matrix_1d=np.bincount(coding)
confusion_matrix=confusion_matrix_1d.reshape(n_classes,n_classes)  # confusion matrix
print("Category matrix (testset):", confusion_matrix)
# Calculate recall and precision for each category.
recall = np.zeros((n_classes,))
precision = np.zeros((n_classes,))
print("Data size (testset):",np.sum(confusion_matrix,axis=1))
for i in range(n_classes):
    recall[i] = confusion_matrix[i,i]/np.sum(confusion_matrix,axis=1)[i]
    precision[i] = confusion_matrix[i,i]/np.sum(confusion_matrix,axis=0)[i]
print("Recall of each category (divide every row):", recall)
print("Precision of each category (divide every column):", precision)
for classi in range(n_classes):
    print("class: ", classes[classi], ", total samples: ",np.sum(confusion_matrix,axis=1)[classi],"  recall: ",recall[classi] , "  precision: ", precision[classi])
total_acc = np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)
print("total accurracy is: ",total_acc)
end_time = time.time()
print("time cost: ", end_time - start_time)
# Save confusion_matrix as .csv file
import pandas 
df = pandas.DataFrame(confusion_matrix)
df.columns = classes
df.index = classes
df.to_csv(f"./confusion_matrix/{data_type}_confusion_matrix.csv")
recall =pandas.Series(recall)
recall.index = classes
recall.name = 'recall'
recall.to_csv(f"./confusion_matrix/{data_type}_recall.csv")
