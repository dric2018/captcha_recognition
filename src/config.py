import os


class Config:
    seed_val = 21
    base_dir = os.path.abspath('../')
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    logs_dir = os.path.join(base_dir, 'logs')
    img_H = 90
    img_W = 282
    test_bs = 64
    train_bs = 128
    validation_pct = .2
    num_workers = 2