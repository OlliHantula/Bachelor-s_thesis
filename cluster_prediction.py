import scanpy as sc
import squidpy as sq
from squidpy.im import ImageContainer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers,
)
import os
from skimage import io
from skimage.color import rgba2rgb
os.chdir("/lustre/scratch/krolha")

if __name__ == '__main__':

     # Define results folder
    results_folder = './bachelor/results/'

    # Check if paths exist and create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if not (os.path.exists("./bachelor/data/") 
            and os.path.exists("./spaceranger-images/")):
            raise FileNotFoundError("Some of the file-folders is not found.")

    def load_from_pickle(filename):
        import pickle
        with open(filename, 'rb') as handle:
            obj = pickle.load(handle)
        return obj

    def save_to_pickle(obj,filename):
        import pickle
        with open(filename, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sc.logging.print_header()
    print(f"squidpy=={sq.__version__}")
    print(f"tensorflow=={tf.__version__}")

    # Print comments, e.g. what am I testing in this run
    print('''\n
    Hopefully final run for bachelor stuff with sample0.1/0.7.
    Test set predictions are also saved.
    Test sets are not shuffled.

    lr=1e-5, batch_size=64, shuffle=all
    Loading batches separately to GPU from CPU to preserve memory.
    Then all are reconstructed from the train sets as before.
    Layers include: (Random)Flip, Contrast(0.8), Rotation(0.5).
    
    Train/test split done 10-fold + train also split to 10% val

    MAX epochs at 150, but callback stops model fit after epoch 50 if val_accuracy doesn't improve in 10 consecutive epochs.
    \n''')

    def get_ohe(adata: AnnData, cluster_key: str, obs_names: np.ndarray):
        cluster_labels = adata[obs_names, :].obs["joint_leiden_clusters"]
        classes = cluster_labels.unique().shape[0]
        cluster_map = {v: i for i, v in enumerate(cluster_labels.cat.categories.values)}
        labels = np.array([cluster_map[c] for c in cluster_labels], dtype=np.uint8)
        labels_ohe = tf.one_hot(labels, depth=classes, dtype=tf.float32)
        return labels_ohe

    def create_dataset(
        adata: AnnData,
        img: ImageContainer,
        obs_names: np.ndarray,
        cluster_key: str,
        augment: bool,
        shuffle: bool,
    ):

        with tf.device("CPU"):
            # image dataset
            spot_generator = img.generate_spot_crops(
                adata,
                obs_names=obs_names,  # this arguent specified the observations names
                scale=1.5,  # this argument specifies that we will consider some additional context under each spot. Scale=1 would crop the spot with exact coordinates
                as_array="image",  # this line specifies that we will crop from the "image" layer. You can specify multiple layers to obtain crops from multiple pre-processing steps.
                return_obs=False,
            )
            image_dataset = tf.data.Dataset.from_tensor_slices([x for x in spot_generator])

            # label dataset
            lab = get_ohe(adata, cluster_key, obs_names)
            lab_dataset = tf.data.Dataset.from_tensor_slices(lab)

            ds = tf.data.Dataset.zip((image_dataset, lab_dataset))

            if shuffle:  # if you want to shuffle the dataset during training
                ds = ds.shuffle(ds.cardinality(), reshuffle_each_iteration=True)
            ds = ds.batch(64)  # batch
            processing_layers = [
                layers.Resizing(128, 128),
                layers.Rescaling(1.0 / 255),
            ]
            augment_layers = [
                layers.RandomFlip(),
                layers.RandomContrast(0.8),
                #layers.RandomTranslation(0.2, 0.2),
                layers.RandomRotation(0.5), 
            ]
            if augment:  # if you want to augment the image crops during training
                processing_layers.extend(augment_layers)

            data_processing = tf.keras.Sequential(processing_layers)
            
            ds = ds.map(lambda x, y: (data_processing(x), y))  # add processing to dataset
            return ds


    adata = load_from_pickle("./bachelor/data/PC_02_10136_VAS_pw0.1_res0.7.pickle")
    img_rgba = io.imread("spaceranger-images/PC_02_10136_VAS_20x_zoom.tiff", plugin='tifffile')
    img_rgb = rgba2rgb(img_rgba)
    img = sq.im.ImageContainer(img_rgb, layer='image')

    # Num of classes
    classes = adata.obs['joint_leiden_clusters'].unique().shape[0]

    # full_ds for predictions
    full_ds = create_dataset(
        adata, img, adata.obs_names.values, "joint_leiden_clusters", augment=False, shuffle=False
    )

    # Define the number of splits
    n_splits = 10

    # Create a new list to save each of the model predictions
    cluster_predictions = ['99']*len(adata.obs)

    # Create a StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    fold_n = 0

    # Save predictions to preds & histories to histories
    test_preds = []
    preds = []
    histories = []

    # Loop over each fold
    for train_idx, test_idx in skf.split(adata, adata.obs['joint_leiden_clusters']):
        
        tf.keras.backend.clear_session() # Clear model session to avoid memory overload

        # Further split the training data into training and validation sets
        train_idx, val_idx = train_test_split(
            adata[train_idx].obs_names.values,
            test_size=0.1,
            stratify=adata[train_idx].obs["joint_leiden_clusters"],
        )

        train_size = adata[train_idx, :].obs.joint_leiden_clusters.value_counts()
        val_size = adata[val_idx, :].obs.joint_leiden_clusters.value_counts()
        test_size = adata[test_idx, :].obs.joint_leiden_clusters.value_counts()

        print(
            f"Final train set ({fold_n}): \n{train_size} \n\
                \nFinal validation set ({fold_n}): \n{val_size} \n\
                \nFinal test set ({fold_n}: \n{test_size}) \n"
        )

        print("resetting datasets...")
        # reset the datasets
        train_ds = None
        val_ds = None
        test_ds = None

        print("before dataset creation")

        train_ds = create_dataset(adata, img, train_idx, "joint_leiden_clusters", augment=True, shuffle=True)
        val_ds = create_dataset(adata, img, val_idx, "joint_leiden_clusters", augment=False, shuffle=True)
        test_ds = create_dataset(adata, img, test_idx, "joint_leiden_clusters", augment=False, shuffle=False)

        print("after dataset creation")
        
        input_shape = (128, 128, 3)  # input shape
        inputs = tf.keras.layers.Input(shape=input_shape)

        # load Resnet with pre-trained imagenet weights
        x = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
            classes=classes,
            pooling="avg",
        )(inputs)
        outputs = tf.keras.layers.Dense(
            units=classes,  # add output layer
            activation='softmax',
        )(x)

        print("before compiling the model")

        model = tf.keras.Model(inputs, outputs)  # create model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # lr should be under 1e-5
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # add loss, , label_smoothing=0.1 -> no imminent results
            metrics=['accuracy', 'AUC'], # keep track of accuracy and area under ROC curve
        )

        # stop the epochs if the val_accuracy decreases
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=10, 
            restore_best_weights=True,
            start_from_epoch=50,
            )

        model.summary()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=150,
            callbacks=[callback],
            verbose=2,
        )

        model.evaluate(test_ds)
        test_pred = model.predict(test_ds)
        test_preds.append([test_idx, test_pred])

        pred = model.predict(full_ds)
        print(f"Prediction: {pred}") 
        preds.append(pred)
        histories.append(history)

        # Collect all the test set predictions to the original anndata
        n = 0
        for i in adata.obs_names.values:
            if np.any(adata[test_idx].obs_names.values == i):
                cluster_predictions[n] = str(np.argmax(pred[n]))
            n+=1

        fold_n += 1

    print(f"length: {len(cluster_predictions)}, n = {n}")
    adata.obs['cluster_predictions'] = cluster_predictions
    adata.obs['cluster_predictions'] = adata.obs['cluster_predictions'].astype('category')
    print(f"Cluster predictions: {adata.obs['cluster_predictions']}")
    print(f"Actual clusters: {adata.obs['joint_leiden_clusters']}")

    file_name = 'bachelor_pred_7.pickle'
    print(file_name)
    save_to_pickle(adata, results_folder+file_name)
    save_to_pickle(preds, results_folder+'preds_'+file_name)
    save_to_pickle(test_preds, results_folder+'test_preds_'+file_name)
    save_to_pickle(histories, results_folder+'histories_'+file_name)
