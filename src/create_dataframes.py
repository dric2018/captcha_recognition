import os
import pandas as pd
import numpy as np
from config import Config

if __name__ == '__main__':
    images_list = os.listdir(os.path.join(Config.data_dir, 'images'))
    labels = [l.split('.')[0] for l in images_list]

    df = pd.DataFrame({"image": images_list, "label": labels})

    print(df.tail())

    train = df.reset_index(drop=True).sample(frac=0.8,
                                             random_state=Config.seed_val)
    test = df.drop(train.index)

    train.to_csv(path_or_buf=os.path.join(Config.data_dir, 'Train.csv'),
                 index=False)
    test.to_csv(path_or_buf=os.path.join(Config.data_dir, 'Test.csv'),
                index=False)
