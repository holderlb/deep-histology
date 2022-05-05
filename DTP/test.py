"""
Model Testing

Usage: python3 test.py <tile_dir> <tissue_type> <pathology> [<tile_size>] [<batch_size>]

Used to test the performance of this model on tile data. Performance 
metrics are recorded to "results". The arguments to the script are as follows:
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
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import cv2
import os
import gc
    

class HistoPathology_Classifier:
    
    def __init__(self, data_dir: str, tissue_type: str, tile_size: int, pathology: str, batch_size: int):
        self.data_dir = data_dir
        self.tissue_type = tissue_type
        self.tile_size = tile_size
        self.pathology = pathology
        self.batch_size = batch_size
        
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
    
    def make_model(self, num_classes):
        if self.tile_size >= 512:
            cb = EfficientNetB5(include_top=False, drop_connect_rate=0.4, input_shape=(self.tile_size, self.tile_size, 3))
        if self.tile_size >= 256:
            cb = EfficientNetB2(include_top=False, drop_connect_rate=0.4, input_shape=(self.tile_size, self.tile_size, 3))
        else:
            cb = EfficientNetB0(include_top=False, drop_connect_rate=0.4, input_shape=(self.tile_size, self.tile_size, 3))
        self.freeze_model(cb, 3)
        #cb.trainable = False
        x = cb.output
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dropout(0.4)(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        model = Model(cb.input, x)
        model.compile(optimizer=optimizers.Adam(1e-4), loss=CategoricalCrossentropy(), metrics=['acc', Precision(class_id=0, name='pr'), Recall(class_id=0, name="rc")])
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
            for im_path in os.listdir(self.data_dir + non_diseased_dir):
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

        print("\nDiseased images:", diseased_images.shape[0])
        print("Non_diseased images:", non_diseased_images.shape[0])
        print("Total images:", len(diseased_images) + len(non_diseased_images))
    
    def get_val_data(self, validation_size=0.1):
        d_validation_size = int(len(self.diseased_images) * validation_size)
        nd_validation_size = int(len(self.non_diseased_images) * validation_size)
        d_val_idxs = np.random.choice(len(self.diseased_images), d_validation_size, replace=False)
        nd_val_idxs = np.random.choice(len(self.non_diseased_images), nd_validation_size, replace=False)
        d_val = self.diseased_images[d_val_idxs]
        nd_val = self.non_diseased_images[nd_val_idxs] 
        validation_data = np.concatenate([d_val, nd_val], axis=0)
        validation_labels = np.concatenate([[[1,0] for i in range(len(d_val))], [[0,1] for i in range(len(nd_val))]], axis=0)
        
        self.d_train_idxs = np.setdiff1d(np.arange(len(self.diseased_images)), d_val_idxs)
        self.nd_train_idxs = np.setdiff1d(np.arange(len(self.non_diseased_images)), nd_val_idxs)
        return shuffle(validation_data, validation_labels)
    
    def get_test_data(self, test_size=0.1):
        d_test_size = int(len(self.diseased_images[self.d_train_idxs]) * test_size)
        nd_test_size = int(len(self.non_diseased_images[self.nd_train_idxs]) * test_size)
        d_test_idxs = np.random.choice(len(self.diseased_images[self.d_train_idxs]), d_test_size, replace=False)
        nd_test_idxs = np.random.choice(len(self.non_diseased_images[self.nd_train_idxs]), nd_test_size, replace=False)
        d_test = self.diseased_images[self.d_train_idxs][d_test_idxs]
        nd_test = self.non_diseased_images[self.nd_train_idxs][nd_test_idxs] 
        test_data = np.concatenate([d_test, nd_test], axis=0)
        test_labels = np.concatenate([[[1,0] for i in range(len(d_test))], [[0,1] for i in range(len(nd_test))]], axis=0)
        
        self.d_train_idxs = np.setdiff1d(np.arange(len(self.diseased_images)), d_test_idxs)
        self.nd_train_idxs = np.setdiff1d(np.arange(len(self.non_diseased_images)), nd_test_idxs)
        return shuffle(test_data, test_labels)
        
    def train_generator(self, augment_data=False):
        y = np.concatenate([[[1,0] for _ in range(self.batch_size//2)], 
                                       [[0,1] for _ in range(self.batch_size//2)]], axis=0)
        train_diseased_images = self.diseased_images [self.d_train_idxs]
        train_non_diseased_images = self.non_diseased_images [self.nd_train_idxs]
        while True:
            d_sample = train_diseased_images [np.random.choice(len(self.d_train_idxs), self.batch_size//2)]
            nd_sample = train_non_diseased_images [self.nd_train_idxs] [np.random.choice(len(self.nd_train_idxs), self.batch_size//2)]
            X = np.concatenate([d_sample, nd_sample], axis=0)
            if augment_data:
                X = tf.image.random_flip_left_right(X)
                X = tf.image.random_flip_up_down(X)
                X = tf.image.random_brightness(X, 0.2)
                X = tf.image.random_contrast(X, 0.7, 1.3)
                X = tf.image.random_saturation(X, 0.8, 1.2)
                X = tf.image.random_hue(X, 0.2)
            #X, y = shuffle(X, train_labels)
            yield X, y
            
    def lr_scheduler(self, epoch, lr):
        if epoch == 0:
            return tf.constant(1e-5)
        if epoch == 1:
            return tf.constant(1e-4)
        else:
            return lr
        
    def eval(self, model, X, Y):
        batch_size = self.batch_size
        print("[Model eval]")
        res = []
        i = 0
        while i < len(X):
            res.append(model.evaluate(X[i:i+batch_size], Y[i:i+batch_size], verbose=0)[:-2])
            i += batch_size
            if i % (len(X) // 10) < batch_size:
                print("[INFO] Processed", i, "/", len(X), "images")
        result = list(np.average(res, axis=0))
        print(result)
        return result
    
    def predict(self, model, X):
        batch_size = self.batch_size
        print("[Model predictions]")
        preds = []
        i = 0
        while i < len(X):
            batch_pred = model(X[i:i+batch_size])
            preds.extend(batch_pred)
            i += batch_size
            if i % (len(X) // 10) < batch_size:
                print("[INFO] Predicted", i, "/", len(X), "images")
        return np.argmax(preds, axis=-1)
    
    def train(self, data, iteration, exp_type, max_epochs=200):

        print('\n-------------------\n    Testing Model\n-------------------')

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 1:
            mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:{}".format(i) for i in range(len(physical_devices))])
            with mirrored_strategy.scope():
                model = self.make_model(2)
        else:
            model = self.make_model(2)

        self.load_data()
        val_data = self.get_val_data(validation_size=0.2)
        x_test, y_test = self.get_val_data(validation_size=0.2)
        generator = self.train_generator(augment_data=True)
        
        es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)
        reduce = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=10, verbose=1)
        warmup = LearningRateScheduler(self.lr_scheduler)
        history = model.fit(generator, steps_per_epoch=100, epochs=max_epochs, validation_data=val_data, validation_batch_size=self.batch_size, callbacks=[es, reduce, warmup]).history
        
        result = self.eval(model, x_test, y_test)
        preds = self.predict(model, x_test)
        result.append(f1_score(np.argmax(y_test, axis=-1), preds, pos_label=0))
        result.append(len(history["loss"]))
        result = [round(r, 4) for r in result]
        cm = confusion_matrix(np.argmax(y_test, axis=-1), preds)
        return (result, cm), history
        
    def write_results(self, result_data):
        results = [i[0] for i in result_data]
        avg_result = np.mean(np.stack(results), axis=0)
        cms = [i[1] for i in result_data]
        avg_cm = np.mean(cms, axis=0).astype(int)
        if not os.path.exists("./results/{}".format(self.tissue_type)):
            os.makedirs("./results/{}".format(self.tissue_type))
        with open("./results/{}/{}{}_results.txt".format(self.tissue_type, self.pathology, self.tile_size), 'w') as f:
            f.write("Results\n\n")
            f.write("Average evaluation loss, accuracy, F1, and epochs score across {} experiment iterations\n".format(len(result_data)))
            f.write(str(avg_result))
            f.write("\n\nAverage confusion matrix across {} experiment iterations\n".format(len(result_data)))
            f.write(str(avg_cm))
            f.write("\n\n\n\nIndividual iteration results")
            for i, (result, cm) in enumerate(result_data):
                f.write("\n\n\nIteration: {}\n".format(i+1))
                f.write("Evaluation loss, accuracy, F1, and epochs:\n")
                f.write(str(result))
                f.write("\n\nConfusion matrix:\n")
                f.write(str(cm))
                
    def plot_results(self, histories):
        plt.figure()
        plt.title("{}-{}{} F1-score".format(self.tissue_type, self.pathology, self.tile_size))
        plt.xlabel("Epochs")
        plt.ylabel("F1-Score")
        for i, h in enumerate(histories):
            f1v = 2 * (np.multiply(h["val_pr"], h["val_rc"]) / np.add(h["val_pr"], h["val_rc"]))
            plt.plot(range(len(f1v)), f1v, label="val_f1-{}".format(i+1))
        for i, h in enumerate(histories):
            f1t = 2 * (np.multiply(h["pr"], h["rc"]) / np.add(h["pr"], h["rc"]))
            plt.plot(range(len(f1t)), f1t, label="train_f1-{}".format(i+1))
        plt.legend()
        plt.savefig("./results/{}/{}{}_f1_results.png".format(self.tissue_type, self.pathology, self.tile_size))
        plt.clf()
        plt.figure()
        plt.title("{}-{}{}_loss".format(self.tissue_type, self.pathology, self.tile_size))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        for i, h in enumerate(histories):
            plt.plot(range(len(h["val_loss"])), h["val_loss"], label="val_loss-{}".format(i+1))
        for i, h in enumerate(histories):
            plt.plot(range(len(h["val_pr"])), h["loss"], label="train_loss-{}".format(i+1))
        plt.legend()
        plt.savefig("./results/{}/{}{}_loss_results.png".format(self.tissue_type, self.pathology, self.tile_size))
        plt.clf()


    def run(self, k=3):
        results_list = []
        history_list = []
        for i in range(k):
            print("\nIteration:", i+1, "\n")
            result_data, history = self.test(i)
            results_list.append(result_data)
            history_list.append(history)
            tf.keras.backend.clear_session()
            gc.collect()
        self.write_results(results_list)
        self.plot_results(history_list)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_dir', type=str, required=True, help='Path to directory of tiles for training')
    parser.add_argument('--tissue_type', type=str, required=True, help='Name of the tissue type. i.e., "breast" tissue')
    parser.add_argument('--pathology', type=str, required=True, help='Name of the pathology you want to classify. It will be the positive class for new binary classification model. Every other class will be treated as the negative class.')
    parser.add_argument('--tile_size', type=int, required=False, default=256, help='Resolution of tiles used for neural network input')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='Batch size for training neural networks')
    return parser.parse_args()
    
    
if __name__ == "__main__":

    args = get_args()
    base_dir = args.tile_dir
    tissue_type = args.tissue_type
    pathology = args.pathology
    tile_size = args.tile_size
    batch_size = args.batch_size
    if '/' in tissue_type:
        tissue_type = tissue_type.replace('/', '-')
        
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Available GPUs:", len(physical_devices))
    print("Tissue type:", tissue_type)
    print("Pathology type:", pathology)
        
    HPC = HistoPathology_Classifier(data_dir=base_dir, tissue_type=tissue_type, tile_size=tile_size, pathology=pathology, batch_size=batch_size)
    HPC.test()

