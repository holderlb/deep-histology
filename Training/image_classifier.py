
import tensorflow as tf
if int(tf.__version__.split('.')[1]) >= 8:
    from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M
    efnets = [EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M]
else:
    from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5
    efnets = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5]
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, Callback
from tensorflow.keras import layers, Model, optimizers, models
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from sklearn.metrics import confusion_matrix, f1_score, log_loss
import numpy as np
from tensorflow_addons.metrics import F1Score
from tensorflow_addons.optimizers import MovingAverage
from imgaug import augmenters
import os
import gc


    
class MergeEnsembleF1(Callback):
    def __init__(self, ensemble_size):
        super(MergeEnsembleF1, self).__init__()
        self.input_names = ["input_{}".format(i+1) for i in range(ensemble_size)]
        self.output_names = ["output_{}".format(i+1) for i in range(ensemble_size)]

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        logs['f1'] = np.mean([logs[n+"_f1"] for n in self.output_names])
        logs['val_f1'] = np.mean([logs["val_"+n+"_f1"] for n in self.output_names])


class Classifier:
    """
    General Image Clasification Model
    
    
    Input arguments:
    
    image_rez - Tuple for image resolution. Default: (224, 224).
    
    batch_size - Integer for training batch size.
    
    ensemble_size - Integer for the number of classifiers in the ensemble.
    
    class_independance - Boolean for prediction type. If only one class is present in each image,
        set class_independance=True. If more than one class is present in any image, set
        class_independance=False.
        
    -----------------------------------------------------------------------------------------------
    
    Methods:
    
    train(X, Y, max_epochs=200) - Trains and saves a new model on input data. 
    
    load_model(model_path) - Load a previously trained model.
    
    predict(X) - Returns prediction probabilities on a set of images X.
    
    evaluate(X, Y) - Computes loss, F1-score, and comfusion matrix on input data X and Y.
    
    """ 
    
    def __init__(self, image_rez: tuple, batch_size: int, ensemble_size: int):
        self.image_rez = image_rez
        self.batch_size = batch_size // ensemble_size
        self.ensemble_size = ensemble_size
        self.name = "ImageClassifier"
        self.input_names = ["input_{}".format(i+1) for i in range(self.ensemble_size)]
        self.output_names = ["output_{}".format(i+1) for i in range(self.ensemble_size)]
    
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
    
    def AttentionModule(self, name="AttentionModule"):
        """Attention module"""
        def channel_attention(x):
            squeeze = layers.Dense(x.shape[-1]//8, kernel_initializer="he_normal")
            excitation = layers.Dense(x.shape[-1], kernel_initializer="he_normal")
            av_pool = layers.GlobalAveragePooling2D()(x)
            mx_pool = layers.GlobalMaxPooling2D()(x)
            av_sq = squeeze(av_pool)
            av_sq = layers.BatchNormalization()(av_sq)
            av_sq = layers.Activation("relu")(av_sq)
            mx_sq = squeeze(mx_pool)
            mx_sq = layers.BatchNormalization()(mx_sq)
            mx_sq = layers.Activation("relu")(mx_sq)
            av_ex = excitation(av_sq)
            mx_ex = excitation(mx_sq)
            pool = layers.Add()([av_ex, mx_ex])
            attention_scores = layers.Activation("sigmoid")(pool)
            attention_scores = layers.Reshape((1,1,x.shape[-1]))(attention_scores)
            return layers.multiply([x, attention_scores])
        def spatial_attention(x):
            feature_map = layers.LocallyConnected2D(filters=2, kernel_size=1, kernel_initializer='he_normal', use_bias=False)(x)
            feature_map = layers.BatchNormalization()(feature_map)
            feature_map = layers.Activation("relu")(feature_map)
            attention_scores = layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(feature_map)
            return layers.multiply([x, attention_scores])
        def attention_module(x_in):
            x = inputs = layers.Input(x_in.shape[1:])
            x = channel_attention(x)
            x = spatial_attention(x)
            return Model(inputs, x, name=name)
        return attention_module
    
    def make_model(self):
        self.multi_label = 1
        ensemble_input = [layers.Input(shape=(self.image_rez[1], self.image_rez[0], 3)) for i in range(self.ensemble_size)]
        if np.mean(self.image_rez) >= 480:
            feature_extractor = efnets[5](include_top=False, input_shape=(self.image_rez[1], self.image_rez[0], 3))
        elif np.mean(self.image_rez) >= 384:
            feature_extractor = efnets[4](include_top=False, input_shape=(self.image_rez[1], self.image_rez[0], 3))
        elif np.mean(self.image_rez) >= 300:
            feature_extractor = efnets[3](include_top=False, input_shape=(self.image_rez[1], self.image_rez[0], 3))
        elif np.mean(self.image_rez) >= 260:
            feature_extractor = efnets[2](include_top=False, input_shape=(self.image_rez[1], self.image_rez[0], 3))
        elif np.mean(self.image_rez) >= 240:
            feature_extractor = efnets[1](include_top=False, input_shape=(self.image_rez[1], self.image_rez[0], 3))
        else:
            feature_extractor = efnets[0](include_top=False, input_shape=(self.image_rez[1], self.image_rez[0], 3))
        feature_extractor.drop_connect_rate = 0.4
        self.freeze_model(feature_extractor, 3)
        X = [feature_extractor(x) for x in ensemble_input]
        self.attention_block = self.AttentionModule()(X[0])
        X = [self.attention_block(x) for x in X]
        X = [layers.GlobalAveragePooling2D()(x) for x in X]
        X = [layers.Dropout(0.4)(x) for x in X]
        activation = ['softmax', 'sigmoid']
        loss = [CategoricalCrossentropy(), BinaryCrossentropy()]
        output = [layers.Dense(self.class_num, activation=activation[self.multi_label], kernel_initializer="he_normal",
                               name="output_{}".format(i+1))(x) for i, x in enumerate(X)]
        model = Model(ensemble_input, output)
        self.opt = MovingAverage(optimizers.Adam(1e-4))
        model.compile(optimizer=self.opt, loss=loss[self.multi_label], 
                      metrics=[F1Score(num_classes=self.class_num, average="macro", threshold=0.5, name="f1")])
        #model.summary()
        print("Multi-label type:", activation[self.multi_label])
        self.model = model
    
    def load_model(self, model_path):
        self.model = models.load_model(model_path)
        
    def save_model(self):
        single_input = self.model.input[0]
        feature_extractor = self.model.layers[self.ensemble_size]
        x = feature_extractor(single_input)
        attention_block = self.model.get_layer("AttentionModule")
        x = attention_block(x)
        x = layers.GlobalAveragePooling2D()(x)
        output = [l(x) for l in self.model.layers[-self.ensemble_size:]]
        pruned_model = Model(single_input, output)
        loss = [CategoricalCrossentropy(), BinaryCrossentropy()]
        pruned_model.compile(optimizer=MovingAverage(optimizers.Adam(1e-4)), loss=loss[self.multi_label], 
                             metrics=[F1Score(num_classes=self.class_num, average="macro", threshold=0.5, name="f1")])
        #pruned_model.summary()
        self.model = pruned_model
        if not os.path.exists("./models/"):
            os.makedirs("./models/")
        self.model.save("./models/" + self.name + ".h5")
        #self.model.save("./models/" + self.name)
    
    def setup_bagging_datasets_training(self):
        idxs = np.arange(len(self.images))
        self.train_idxs = [np.random.choice(idxs, len(idxs), replace=True) for _ in range(self.ensemble_size)]
        val_idxs = [np.setdiff1d(idxs, train_idx) for train_idx in self.train_idxs]
        min_val_length = np.min([len(val_idx) for val_idx in val_idxs])
        self.val_idxs = [val_idx[:min_val_length] for val_idx in val_idxs]
        self.compute_selection_probabilities()
        
    def setup_bagging_datasets_testing(self, test_size=0.2):
        test_ids = np.random.choice(len(self.images), int(len(self.images)*test_size))
        idxs = np.delete(np.arange(len(self.images)), test_ids)
        self.train_idxs = [np.random.choice(idxs, len(idxs), replace=True) for _ in range(self.ensemble_size)]
        val_idxs = [np.setdiff1d(idxs, train_idx) for train_idx in self.train_idxs]
        min_val_length = np.min([len(val_idx) for val_idx in val_idxs])
        self.val_idxs = [val_idx[:min_val_length] for val_idx in val_idxs]
        test_data = self.images[test_ids]
        test_labels = self.labels[test_ids]
        self.compute_selection_probabilities()
        return (test_data, test_labels)
        
    def compute_selection_probabilities(self, smoothing=0):
        self.selection_probs = []
        for train_idx in self.train_idxs:
            label_sums = np.zeros(shape=(self.class_num,)) + smoothing
            for label in self.labels[train_idx]:
                for i, l in enumerate(label):
                    if l == 1:
                        label_sums[i] += 1
            weights = [1/(len(label_sums)*s) for s in label_sums]
            selections_probs = [np.mean(weights * l) for l in self.labels[train_idx]]
            selections_probs = selections_probs / np.sum(selections_probs)
            self.selection_probs.append(selections_probs)
    
    def predict(self, X):
        """Prediction funtion for model. 

        Args:
            X - Numpy array of images. Takes the shape (N, W, H, C) where N is the length of the list,
                W is the width of an image, H is the height, and C is the color channels.

        Returns:
            preds - prediction probabilities of the shape (N, S) where N is the length of the list
                and S is the number of classes.
        """
        batch_size = self.batch_size * self.ensemble_size
        print("[Model predictions]")
        preds = []
        i = 0
        while i < len(X):
            batch_pred = self.model(X[i:i+batch_size])
            batch_pred = np.mean(batch_pred, axis=0)
            preds.extend(batch_pred)
            i += batch_size
            if i % (len(X) // 5) < batch_size:
                print("[INFO] Predicted", i, "/", len(X), "images")
        return np.array(preds)

    def evaluate(self, X, Y, class_labels=None):
        """Evaluation funtion for model. 

        Args:
            X - Numpy array of images. Takes the shape (N, W, H, C) where N is the length of the list,
                W is the width of an image, H is the height, and C is the color channels.
                
            Y - Numpy array of sparse labels. Takes the shape (N, S) where N is the length of the list
                and S is the number of classes.

        Returns:
            loss - average entropy loss between Y and Y'
            
            f1 - average macro F1-score between Y and Y'
            
            cm - confusion matrix built from Y and Y'
        """
        print('\n----------------------\n    Evaluating Model\n----------------------')
        probs = self.predict(X)  
        preds = np.where(probs > 0.5, 1, 0)
        y_true = Y#np.argmax(Y, axis=-1)
        loss = log_loss(y_true, probs)
        f1 = f1_score(y_true, preds, average="macro")
        #cm = multilabel_confusion_matrix(y_true, preds, labels=class_labels, samplewise=True)
        results={}
        for label_col in range(len(class_labels)):
            y_true_label = 1-y_true[:, label_col]
            y_pred_label = 1-preds[:, label_col]
            loss = log_loss(y_true_label, y_pred_label)
            f1 = f1_score(y_true_label, y_pred_label, pos_label=0)
            cm = confusion_matrix(y_true_label, y_pred_label)
            results[class_labels[label_col]] = {"loss": loss, "f1-score": f1, "confusion_matrix": cm}

        for label, result in results.items():
            print("\n" + label + ':')
            print("Loss:", result["loss"])
            print("F1-score:", result["f1-score"])
            print("\nConfusion Matrix:\n", result["confusion_matrix"])
        #print("Loss:", loss)
        #print("F1-score:", f1)
        #print("\Confusion Matrix:\n", cm)
        return results
        
    def train_generator(self, augment_data=False):
        x_dict = dict(zip(self.input_names, [None for _ in range(self.ensemble_size)]))
        y_dict = dict(zip(self.output_names, [None for _ in range(self.ensemble_size)]))
        rand_aug = augmenters.RandAugment()
        while True:
            sample_idxs = [np.random.choice(train_idx, self.batch_size, p=None) #self.selection_probs[i]
                           for i, train_idx in enumerate(self.train_idxs)]
            X = [self.images [idxs] for idxs in sample_idxs]
            Y = [self.labels [idxs] for idxs in sample_idxs]
            if augment_data:
                X = [rand_aug(images=x) for x in X]
            x_dict.update(zip(self.input_names, X))
            y_dict.update(zip(self.output_names, Y))
            yield x_dict, y_dict
    
    def val_generator(self):
        x_dict = dict(zip(self.input_names, [None for _ in range(self.ensemble_size)]))
        y_dict = dict(zip(self.output_names, [None for _ in range(self.ensemble_size)]))
        while True:  
            sample_idxs = [np.random.choice(val_idx, self.batch_size) for val_idx in self.val_idxs]
            X = [self.images [idxs] for idxs in sample_idxs]
            Y = [self.labels [idxs] for idxs in sample_idxs]
            x_dict.update(zip(self.input_names, X))
            y_dict.update(zip(self.output_names, Y))
            yield x_dict, y_dict
    
    def lr_scheduler(self, epoch, lr):
        lr_start   = 1e-6
        lr_max     = 1e-4
        lr_ramp_ep = 5
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        if epoch == lr_ramp_ep:
            lr = lr_max
        return lr
    
    def train(self, X: np.ndarray, Y: np.ndarray, max_epochs=400, test=False):
        """Training funtion for model. Automatically train on input images and labels.
        outputs a model to current directory.

        Args:
            X - Numpy array of images. Takes the shape (N, W, H, C) where N is the length of the list,
                W is the width of an image, H is the height, and C is the color channels.
                
            Y - Numpy array of sparse labels. Takes the shape (N, S) where N is the length of the list
                and S is the number of classes.
                
            max_epochs (optional) - Cutoff for model training. Training usually stops
                before this due to EarlyStopping callback.
                
            test (optional) - Indicator for running model test or not. If true, test set will be
                created and used for model evaluation.

        Returns:
            history - training and validation metrics for each epoch.
            
            results - Evaluation on test set if test=True
        """

        print('\n----------------------\n    Training Model\n----------------------')
        
        self.images = X
        self.labels = Y
        self.multi_label = True if any([np.sum(label) > 1 for label in self.labels]) else False
        self.class_num = np.unique([np.argmax(label) for label in self.labels]).shape[0]
        if test:
            (test_data, test_labels) = self.setup_bagging_datasets_testing()
        else:
            self.setup_bagging_datasets_training()

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        #for device in physical_devices:
        #    tf.config.experimental.set_memory_growth(device, True)
        if len(physical_devices) > 1:
            mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:{}".format(i) for i in range(len(physical_devices))])
            with mirrored_strategy.scope():
                self.make_model()
        else:
            self.make_model()

        train_gen = self.train_generator(augment_data=True)
        val_gen = self.val_generator()
        train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, output_types=(dict(zip(self.input_names, [tf.uint8 for _ in range(self.ensemble_size)])),
                                                                                        dict(zip(self.output_names, [tf.uint8 for _ in range(self.ensemble_size)]))))
        val_dataset = tf.data.Dataset.from_generator(lambda: val_gen, output_types=(dict(zip(self.input_names, [tf.uint8 for _ in range(self.ensemble_size)])),
                                                                                    dict(zip(self.output_names, [tf.uint8 for _ in range(self.ensemble_size)]))))
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)
        
        stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)
        reduce = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=10, verbose=1)
        warmup = LearningRateScheduler(self.lr_scheduler)
        history = self.model.fit(train_dataset, steps_per_epoch=100, epochs=max_epochs, 
                                 validation_data=val_dataset, validation_steps=100, 
                                 callbacks=[MergeEnsembleF1(self.ensemble_size), stopping, reduce, warmup], verbose=0).history
        self.save_model()
        if test:
            results = self.evaluate(test_data, test_labels, class_labels=self.class_labels)
        else:
            results = None
        return history, results
    
    def write_results(self, result_data):
        losses = [[results[key]["loss"] for key in results.keys()] for results in result_data]
        f1s = [[results[key]["f1-score"] for key in results.keys()] for results in result_data]
        cms = [[results[key]["confusion_matrix"] for key in results.keys()] for results in result_data]
        classes = result_data[0].keys()
        with open("./results/{}_results.txt".format(self.name), 'w') as f:
            f.write("Results\n\n")
            f.write("Averaged test results across {} bootstrapped models and {} experiment iterations\n".format(self.ensemble_size, len(result_data)))
            for i, cl in enumerate(classes):
                f.write("\n\n" + cl + ':\n')
                f.write("\nAverage Loss: " + str(np.mean(losses, axis=0)[i]))
                f.write("\nAverage F1-Score: " + str(np.mean(f1s, axis=0)[i]))
                f.write("\nAveraged confusion matrix \n".format(self.ensemble_size, len(result_data)))
                f.write(str(np.mean(cms, axis=0)[i].astype(int)))
            f.write("\n\n\n\nIndividual iteration results")
            for i, results in enumerate(result_data):
                f.write("\n\n\nIteration: {}".format(i+1))
                for label, result in results.items():
                    f.write("\n\n" + label + ':')
                    f.write("\nLoss:" + str(result["loss"]))
                    f.write("\nF1-score:" + str(result["f1-score"]))
                    f.write("\nConfusion Matrix:\n" + str(result["confusion_matrix"]))

    def test(self, X: np.ndarray, Y: np.ndarray, k=3, class_labels=None):
        """Runs test variant of the train function k times.

        Args:
            X - Numpy array of images. Takes the shape (N, W, H, C) where N is the length of the list,
                W is the width of an image, H is the height, and C is the color channels.
                
            Y - Numpy array of sparse labels. Takes the shape (N, S) where N is the length of the list
                and S is the number of classes.
                
            k (optional): Number of folds for testing. Defaults to 3.
        """
        self.class_labels=class_labels
        results_list = []
        history_list = []
        for i in range(k):
            print("\nIteration:", i+1, "\n")
            history, results = self.train(X, Y, test=True)
            results_list.append(results)
            history_list.append(history)
            tf.keras.backend.clear_session()
            gc.collect()
        self.write_results(results_list)
