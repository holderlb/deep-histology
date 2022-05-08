"""
Model Training

Usage: python3 train.py <tile_dir> <tissue_type> <pathology> [<tile_size>] [<batch_size>]

Used to train tile classifying models. The arguments to the script
are as follows:
- <tile_dir> Path to directory of tiles for training
- <tissue_type> help='Name of the tissue type. i.e., "breast" tissue
- <pathology> Name of the pathology you want to classify. It will be the positive class for 
  new binary classification model. Every other class will be treated as the negative class
- [<tile_size>] (Optional) Resolution of tiles used for neural network input
- [<batch_size] (Optional) Batch size for training neural networks


Authors: Colin Greeley and Larry Holder, Washington State University
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB2, EfficientNetB5
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import layers,Model, optimizers
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
import time
import argparse
import cv2
import os
    

class HistoPathology_Classifier:
    
    def __init__(self, data_dir: str, tissue_type: str, tile_size: int, pathology: str, batch_size: int, ensemble_size: int):
        self.data_dir = data_dir
        self.tissue_type = tissue_type
        self.tile_size = tile_size
        self.pathology = pathology
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        
    def freeze_model(self, m, block):
        m.trainable = True
        i = 0
        while i < len(m.layers):
            if 'block{}'.format(block) in m.layers[i].name:
                break
            m.layers[i].trainable = False
            i += 1
        while i < len(m.layers):
            if isinstance(m.layers[i], layers.BatchNormalization):
                m.layers[i].trainable = False
            i += 1
        return m
    
    def make_model(self):
        ensemble_input = [layers.Input(shape=(self.tile_size, self.tile_size, 3)) for i in range(self.ensemble_size)]
        if self.tile_size >= 512:
            cb = EfficientNetB5(include_top=False, drop_connect_rate=0.4, input_shape=(self.tile_size, self.tile_size, 3))
        if self.tile_size >= 256:
            cb = EfficientNetB2(include_top=False, drop_connect_rate=0.4, input_shape=(self.tile_size, self.tile_size, 3))
        else:
            cb = EfficientNetB0(include_top=False, drop_connect_rate=0.4, input_shape=(self.tile_size, self.tile_size, 3))
        self.freeze_model(cb, 4)
        X = [cb(x) for x in ensemble_input]
        X = [layers.GlobalAveragePooling2D()(x) for x in X]
        X = [layers.Dropout(0.4)(x) for x in X]
        output = [layers.Dense(2, activation='softmax')(x) for x in X]
        model = Model(ensemble_input, output)
        model.compile(optimizer=optimizers.Adam(1e-4), loss=CategoricalCrossentropy(), metrics=[Precision(class_id=0, name='pr'), Recall(class_id=0, name="rc")])
        model.summary()
        return model

    def load_data(self):
        diseased_dir = self.data_dir + self.pathology + '/'
        non_diseased_dirs = sorted([ndir + '/' for ndir in os.listdir(self.data_dir) if (self.pathology != ndir and 
                                                                                    'csv' not in ndir and 
                                                                                    'Ignore' not in ndir and 
                                                                                    'Other' not in ndir)])
        start = time.time()
        print("\n[INFO] Gathering images and converting them to np arrays")
        dim = (self.tile_size, self.tile_size)
        diseased_images = []
        for im_path in os.listdir(diseased_dir):
            diseased_images.append(cv2.resize(cv2.imread(diseased_dir + im_path), dim, cv2.INTER_AREA))
        non_diseased_images = []
        nd_length = np.sum([len(os.listdir(self.data_dir + non_diseased_dir)) for non_diseased_dir in non_diseased_dirs])
        for non_diseased_dir in non_diseased_dirs:
            for im_path in os.listdir(self.data_dir + non_diseased_dir)[:]:
                #if np.random.random() < 5_000 / nd_length:    # 50,000 non-diseased images at most
                non_diseased_images.append(cv2.resize(cv2.imread(self.data_dir + non_diseased_dir + im_path), dim, cv2.INTER_AREA))
        self.diseased_images = np.asarray(diseased_images)
        self.non_diseased_images = np.asarray(non_diseased_images)
        del diseased_images
        del non_diseased_images
        if len(self.diseased_images) > len(self.non_diseased_images):
            self.diseased_images = self.diseased_images[np.random.choice(len(self.diseased_images), len(self.non_diseased_images)-1, replace=False)]
        print(self.diseased_images.shape)
        print(self.non_diseased_images.shape)
        print("[INFO] Conversion took {} seconds".format(round(time.time() - start, 2)))

        print("\nDiseased images:", self.diseased_images.shape[0])
        print("Non_diseased images:", self.non_diseased_images.shape[0])
        print("Total images:", len(self.diseased_images) + len(self.non_diseased_images), '\n\n')
    
    def setup_bagging_datasets(self):
        self.d_train_idxs = [np.random.choice(len(self.diseased_images), len(self.diseased_images), replace=True) for _ in range(self.ensemble_size)]
        self.nd_train_idxs = [np.random.choice(len(self.non_diseased_images), len(self.non_diseased_images), replace=True) for _ in range(self.ensemble_size)]
        
        d_val_idxs = [np.setdiff1d(np.arange(len(self.diseased_images)), d_train_idx) for d_train_idx in self.d_train_idxs]
        nd_val_idxs = [np.setdiff1d(np.arange(len(self.non_diseased_images)), nd_train_idx) for nd_train_idx in self.nd_train_idxs]
        d_min_val_length = np.min([len(d_val_idx) for d_val_idx in d_val_idxs])
        nd_min_val_length = np.min([len(nd_val_idx) for nd_val_idx in nd_val_idxs])
        self.d_val_idxs = [d_val_idx[:d_min_val_length] for d_val_idx in d_val_idxs]
        self.nd_val_idxs = [nd_val_idx[:nd_min_val_length] for nd_val_idx in nd_val_idxs]
        
        self.d_val_batch = ceil(self.batch_size * (d_min_val_length / (d_min_val_length + nd_min_val_length)))
        self.nd_val_batch = floor(self.batch_size * (nd_min_val_length / (d_min_val_length + nd_min_val_length)))
        
    def train_generator(self, augment_data=False):
        y = ([np.concatenate([[[1,0] for _ in range(self.batch_size//2)], [[0,1] for _ in range(self.batch_size//2)]], axis=0) for _ in range(self.ensemble_size)])
        while True:
            d_samples = [self.diseased_images [np.random.choice(d_train_idx, self.batch_size//2)] for d_train_idx in self.d_train_idxs]
            nd_samples = [self.non_diseased_images [np.random.choice(nd_train_idx, self.batch_size//2)] for nd_train_idx in self.nd_train_idxs]
            X = [np.concatenate([d_sample, nd_sample], axis=0) for d_sample, nd_sample in zip(d_samples, nd_samples)]
            if augment_data:
                X = [tf.image.random_flip_left_right(x) for x in X]
                X = [tf.image.random_flip_up_down(x) for x in X]
                X = [tf.image.random_brightness(x, 0.2) for x in X]
                X = [tf.image.random_contrast(x, 0.7, 1.3) for x in X]
                X = [tf.image.random_saturation(x, 0.8, 1.2) for x in X]
                X = [tf.image.random_hue(x, 0.2) for x in X]
            yield X, y
             
    def val_generator(self):
        y = ([np.concatenate([[[1,0] for _ in range(self.d_val_batch)], [[0,1] for _ in range(self.nd_val_batch)]], axis=0) for _ in range(self.ensemble_size)])
        while True:
            d_samples = [self.diseased_images [np.random.choice(d_val_idx, self.d_val_batch)] for d_val_idx in self.d_val_idxs]
            nd_samples = [self.non_diseased_images [np.random.choice(nd_val_idx, self.nd_val_batch)] for nd_val_idx in self.nd_val_idxs]
            X = [np.concatenate([d_sample, nd_sample], axis=0) for d_sample, nd_sample in zip(d_samples, nd_samples)]
            yield X, y
            
    def lr_scheduler(self, epoch, lr):
        if epoch == 0:
            return tf.constant(1e-5)
        if epoch == 1:
            return tf.constant(1e-4)
        else:
            return lr
    
    def predict(self, model, X):
        batch_size = self.batch_size
        print("[Model predictions]")
        preds = []
        i = 0
        while i < len(X):
            batch_pred = model([X[i:i+batch_size] for _ in range(self.ensemble_size)])
            batch_pred = np.mean(batch_pred, axis=0)
            preds.extend(batch_pred)
            i += batch_size
            if i % (len(X) // 5) < batch_size:
                print("[INFO] Predicted", i, "/", len(X), "images")
        return np.argmax(preds, axis=-1)
    
    def train(self, max_epochs=200):

        print('\n-------------------\n    Training Model\n-------------------')

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 1:
            mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:{}".format(i) for i in range(len(physical_devices))])
            with mirrored_strategy.scope():
                model = self.make_model()
        else:
            model = self.make_model()

        self.load_data()
        self.setup_bagging_datasets()
        train_gen = self.train_generator(augment_data=True)
        val_gen = self.val_generator()
        
        es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)
        reduce = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=10, verbose=1)
        warmup = LearningRateScheduler(self.lr_scheduler)
        
        history = model.fit(train_gen, steps_per_epoch=100, epochs=max_epochs, validation_data=val_gen, validation_steps=100, callbacks=[es, reduce, warmup], verbose=2).history
        if not os.path.exists("./models/{}".format(self.tissue_type)):
            os.makedirs("./models/{}".format(self.tissue_type))
        model.save('./models/{}/{}{}.h5'.format(self.tissue_type, self.pathology, self.tile_size))
        return history
                
    def plot_results(self, history):
        if not os.path.exists("./results/{}".format(self.tissue_type)):
            os.makedirs("./results/{}".format(self.tissue_type))
        ids = [''] + ['_'+str(i) for i in range(1, self.ensemble_size)]
        plt.figure()
        plt.title("{}-{}{} F1-score".format(self.tissue_type, self.pathology, self.tile_size))
        plt.xlabel("Epochs")
        plt.ylabel("F1-Score")
        for i in range(self.ensemble_size):
            pr = history["val_dense{}_pr".format(ids[i])]
            rc = history["val_dense{}_rc".format(ids[i])]
            f1v = 2 * (np.multiply(pr, rc) / np.add(pr, rc))
            plt.plot(range(len(f1v)), f1v, label="val_f1-{}".format(i+1))
        for i in range(self.ensemble_size):
            pr = history["dense{}_pr".format(ids[i])]
            rc = history["dense{}_rc".format(ids[i])]
            f1v = 2 * (np.multiply(pr, rc) / np.add(pr, rc))
            plt.plot(range(len(f1v)), f1v, label="val_f1-{}".format(i+1))
        plt.legend()
        plt.savefig("./results/{}/{}{}_f1_results.png".format(self.tissue_type, self.pathology, self.tile_size))
        plt.clf()
        plt.figure()
        plt.title("{}-{}{}_loss".format(self.tissue_type, self.pathology, self.tile_size))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        for i in range(self.ensemble_size):
            loss = history["val_dense{}_loss".format(ids[i])]
            plt.plot(range(len(loss)), loss, label="val_loss-{}".format(i+1))
        for i in range(self.ensemble_size):
            loss = history["dense{}_loss".format(ids[i])]
            plt.plot(range(len(loss)), loss, label="train_loss-{}".format(i+1))
        plt.legend()
        plt.savefig("./results/{}/{}{}_loss_results.png".format(self.tissue_type, self.pathology, self.tile_size))
        plt.clf()

    def run(self):
        history = self.train()
        self.plot_results(history)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_dir', type=str, required=True, help='Path to directory of tiles for training')
    parser.add_argument('--tissue_type', type=str, required=True, help='Name of the tissue type. i.e., "breast" tissue')
    parser.add_argument('--pathology', type=str, required=True, help='Name of the pathology you want to classify. It will be the positive class for new binary classification model. Every other class will be treated as the negative class.')
    parser.add_argument('--tile_size', type=int, required=False, default=256, help='Resolution of tiles used for neural network input')
    parser.add_argument('--batch_size', type=int, required=False, default=50, help='Batch size for training neural networks')
    return parser.parse_args()

    
if __name__ == "__main__":

    args = get_args()
    base_dir = args.tile_dir
    tissue_type = args.tissue_type
    pathology = args.pathology
    tile_size = args.tile_size
    batch_size = args.batch_size
    enseble_size = 5
    if '/' in tissue_type:
        tissue_type = tissue_type.replace('/', '-')
        
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Available GPUs:", len(physical_devices))
    print("Tissue type:", tissue_type)
    print("Pathology type:", pathology)
        
    HPC = HistoPathology_Classifier(data_dir=base_dir, 
                                    tissue_type=tissue_type, 
                                    tile_size=tile_size, 
                                    pathology=pathology, 
                                    batch_size=batch_size, 
                                    ensemble_size=enseble_size)
    HPC.run()
