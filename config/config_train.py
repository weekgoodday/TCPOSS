class config():
    epochs = 40
    train_datapath = "./TCPOSS_data/h5py/train_easy.hdf5"
    test_datapath = "./TCPOSS_data/h5py/test_easy.hdf5"
    batch_size = 64
    learning_rate = 1e-3
    lr_decay = 0.9
    n_classes = 10
    is_save = True
    save_path = "./model/mobilenet_easy.pth"
    what_we_want = "start training, use easy split"
    is_frozen = False
    is_pretrained = True
    is_augment = True
    repeat_time = 1
    save_all = False    
    is_dropout = True
    model_type = "mobilenet"