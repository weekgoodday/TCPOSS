train.py : Image classification, loading dataset and training, each time will record hyperparameters, loss and acc.
Optional hyperparameters :
    String variables :
    --what_we_want (Why do you want to do this experiment, express in words, default="Default")
    --repeat_time (The number of times the training needs to be repeated, to observe the fluctuations caused by the randomness of initialization and data augmentation, default=5)
    --epochs (Training epochs, default=40)
    --train_datapath (Training set path, default="./TCPOSS_data/h5py/train_entry.hdf5")
    --test_datapath (Test set path, default="./TCPOSS_data/h5py/test_entry.hdf5")
    --save_path (Model parameter record path, default="./model/imgmodel_entry.pth")
    --batch_size (Batch size, default=64)
    --learning_rate (Learning rate, default=1e-3)
    --lr_decay (Learning rate decay coefficient, exponential decay, default=0.9)
    --n_classes (Number of classification categories, default=10)
    Boolean variables（6, default is False when not added, True when added）:
    --is_save (Whether to save the model parameters, by default only the last repeat model parameters are saved, the path is save_path)
    --save_all (Whether to save all the model parameters of each repeated training, the path will be saved with "_number", when is_save is False, save_all does not work)
    --is_frozen (Whether to freeze the feature extraction network)
    --is_pretrained (Whether to load pretrained parameters)
    --is_augment (Whether to incorporate data augmentation)
    --is_dropout (Whether to add two layers of Dropout to the classifier)
Run :
python train.py --what_we_want start training, use easy split --repeat_time 1 --model_type mobilenet --is_pretrained --is_augment --is_dropout --epochs 40 --train_datapath "./TCPOSS_data/h5py/train_easy.hdf5" --test_datapath "./TCPOSS_data/h5py/train_easy.hdf5" --save_path "./model/mobilenet_easy.pth"
Note: If you run python train.py without any command line parameters, the code will read the struct in ./config/config_train.py as hyperparameters

evaluate.py : Based on the trained network parameters, infer on the test set and output the confusion matrix.

dataset.py : Helps to build Dataset class, do not run separately

plot_tsne.py : Draw T-SNE graph, observe the feature distribution of training set and test set.
Because of the optimization characteristics of T-SNE, the two sets are combined for TSNE optimization inference, and then they are observed separately.
There are no command line parameters, pay attention to modify the model and training set and test set path before running.

5 Confidence Methods:

SoftmaxConf.py : uses logits after Softmax layer.
Plot remained accuracy (the accuracy of the data with confidence larger than threshold) with the confidence threshold increases. Print the area under this remained acc curve.
There are no command line parameters, pay attention to modify the model and test set path before running.

MCdropout.py : MCdropout calculates total uncertainty, aleatoric uncertainty, epistemic uncertainty in evaluation. Use negative total uncertainty as confidence.
Plot remained accuracy (the accuracy of the data with confidence larger than threshold) with the confidence threshold increases. Print the area under this remained acc curve.
There are no command line parameters, pay attention to modify the model and test set path before running.

Ensemble.py : Like MCDropout, calculates negative total uncertainty as confidence. Need 5 models trained before.
There are no command line parameters, pay attention to modify the model and test set path before running.

Mahalanobis.py : calculates negative Mahalanobis distance as confidence. We use a specific covariance matrix for a specific class, to ensure the reversibility of covariance matrix, we use PCA.
Some experimental results show our implementation performs better than original, although Mahalanobis distance itself performs not so well on our task.
There are no command line parameters, pay attention to modify the model and test set path before running.

EDL:
    helpers.py : auxiliary codes for EDL training and testing.
    losses.py : auxiliary codes for EDL training and testing.
    train_EDL.py : codes to train EDL model, because EDL needs a different training approach.
    EDL.py : codes to test EDL model.