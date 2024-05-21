'''
Draw the T-SNE graph by merging the training and test sets together. 
Because T-SNE is not a simple projection of coordinates, it is necessary to optimize the training set and test set together.
Minimizes the Kullback-Leiber divergence of the Gaussians P.
'''
import torch
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import h5py
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset
import torch.nn.functional as F
# revise the model and data path
model_path = "./model_used/mobilenet_entry.pth"  
train_datapath = "./data_used/train_entry.hdf5"
test_datapath = "./data_used/test_entry.hdf5"
savepath_tsne_train = "./tsne/plot_tsne_entry_train.png"  # Save path for training set tsne image
savepath_tsne_test = "./tsne/plot_tsne_entry_test.png"  # Save path for test set tsne image

n_classes = 10
batch_size = 64
features_in_hook1 = []
def hook1(module, input, output):
    features_in_hook1.append(input[0].cpu().numpy())
labelp = ['asphalt','grass','cement','board','brick','gravel','sand','flagstone','plastic','soil']
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
model.classifier[1].register_forward_hook(hook1) 
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
train_data = load_data(train_datapath)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
gt_train = np.empty((0,),dtype=np.int32)
pd_train = np.empty((0,),dtype=np.int32)
probs_train = np.empty((0,n_classes),dtype=np.float32)
with torch.no_grad():
    for X,y in train_dataloader:
        imgs = X.to(device)
        label = y
        yhat = model(imgs)
        pred = torch.argmax(yhat, dim=1)
        prob = F.softmax(yhat, dim=1)
        gt_train = np.append(gt_train, label.cpu().numpy())
        pd_train = np.append(pd_train, pred.cpu().numpy())
        probs_train = np.append(probs_train, prob.cpu().numpy(), axis=0)
# concat the element in features_in_hook1 to an array
features_train = np.concatenate(features_in_hook1, axis=0)
datasize_train = features_train.shape[0]

test_data = load_data(test_datapath)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
gt_test = np.empty((0,),dtype=np.int32)
pd_test = np.empty((0,),dtype=np.int32)
probs_test = np.empty((0,n_classes),dtype=np.float32)
features_in_hook1 = []
with torch.no_grad():
    for X,y in test_dataloader:
        imgs = X.to(device)
        label = y
        yhat = model(imgs)
        pred = torch.argmax(yhat, dim=1)
        prob = F.softmax(yhat, dim=1)
        gt_test = np.append(gt_test, label.cpu().numpy())
        pd_test = np.append(pd_test, pred.cpu().numpy())
        probs_test = np.append(probs_test, prob.cpu().numpy(), axis=0)
features_test = np.concatenate(features_in_hook1, axis=0)
features_total = np.concatenate((features_train, features_test), axis=0)
gt = np.concatenate((gt_train, gt_test), axis=0)
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(features_total)
mask_data_train = np.zeros((X_2d.shape[0],), dtype=bool)
mask_data_train[:datasize_train] = True
mask_data_test = ~mask_data_train

plt.figure()
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
for i, c in zip(range(n_classes), colors):
    masks = (gt == i) & mask_data_train
    plt.scatter(X_2d[masks, 0], X_2d[masks, 1], c=c, label=labelp[i], s=1)
plt.legend(bbox_to_anchor=(0.5,-0.1), ncol=5, loc='upper center', markerscale=5)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# plt.show()
plt.savefig(savepath_tsne_train, bbox_inches='tight')
plt.figure()
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
for i, c in zip(range(n_classes), colors):
    masks = (gt == i) & mask_data_test
    plt.scatter(X_2d[masks, 0], X_2d[masks, 1], c=c, label=labelp[i], s=1)
plt.legend(bbox_to_anchor=(0.5,-0.1), ncol=5, loc='upper center', markerscale=5)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
# plt.show()
plt.savefig(savepath_tsne_test, bbox_inches='tight')

