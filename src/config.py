import os


class Config:
    seed_val = 21
    base_dir = os.path.abspath('../')
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    logs_dir = os.path.join(base_dir, 'logs')
    labels = list(
        "_ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    img_H = 180  # 90
    img_W = 300  # 282
    target_seq_len = 11
    test_bs = 64  #32
    train_bs = 64  #32
    validation_pct = .2
    num_workers = 2
    base_model = 'resnet18'  # seresnext26d_32x4d # efficientnet_b3 # seresnet152d # resnet34 resnet50
    num_decoder_layers = 2
    dropout_rate = .25
    decoder_hidden_size = 64
    decoder_input_size = 128
    optimizer = "adamw"
    reduce_lr_on_plateau = False
    lr = 3e-3
    weight_decay = 0.001
    eps = 1e-08
    cooldown = 0
    precision = 32
    accumulate_grad_batches = 1
    n_folds = None
    stratified = False