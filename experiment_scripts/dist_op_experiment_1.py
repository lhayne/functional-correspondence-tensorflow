import tensorflow as tf
import tensorflow_datasets as tfds
from util import modeling
import gc
import json
import pandas as pd
import numpy as np

def main():
    """
    Experiment to test if we can jointly optimize for the distance between two representations
    in two separate networks AND their in distribution performance.
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    input_shape = (28, 28, 1)

    # baseline_model = tf.keras.Sequential(
    # [
    #     tf.keras.Input(shape=input_shape),
    #     tf.keras.layers.ZeroPadding2D(2),
    #     tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
    #     tf.keras.layers.ZeroPadding2D(2),
    #     tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
    #     tf.keras.layers.ZeroPadding2D(2),
    #     tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation="relu"),
    #     tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(32,activation="relu"),
    #     tf.keras.layers.Dense(32,activation="relu"),
    #     tf.keras.layers.Dense(10),
    # ]
    # )

    ds = tfds.load('mnist_corrupted/stripe', split='test', shuffle_files=False, batch_size=-1)
    corrupted_images = tfds.as_numpy(ds)['image']

    stats = pd.DataFrame([],columns=['beta','rep_loss','y1_acc','y2_acc','distillation_loss','val_distillation_loss','train_agreement','val_agreement','corruption_agreement','epochs'])

    for beta in [-1,0,1e-3,5e-3,1e-2,1]:
        model_A = tf.keras.models.load_model('../data/model_width_4_iteration_0')
        model_A1 = tf.keras.models.Model(model_A.input,model_A.layers[11].output)
        model_A2 = tf.keras.models.Model(model_A.layers[12].input,model_A.output)

        model_B = tf.keras.models.load_model('../data/model_width_4_iteration_0')
        model_B1 = tf.keras.models.Model(model_B.input,model_B.layers[11].output)
        model_B2 = tf.keras.models.Model(model_B.layers[12].input,model_B.output)

        model = modeling.HModel(model_A1=model_A1,model_A2=model_A2,model_B1=model_B1,model_B2=model_B2)

        model_A1.build((None,28,28,1))
        model_B1.build((None,28,28,1))
        model_A2.build((None,32))
        model_B2.build((None,32))
        model.build((None,28,28,1))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            y1_metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            y2_metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            rep_loss_fn=modeling.lin_cka_dist_2,
            y1_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(), 
            y2_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
            distillation_loss_fn=tf.keras.losses.KLDivergence(),
            alpha=0.5,
            beta=100,
        )

        history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=256,epochs=100,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss',
                                        mode='min',
                                        min_delta=0,
                                        patience=5,
                                        restore_best_weights=True,
                )])

        y1_predictions,y2_predictions = model.predict(x_train)
        train_agreement = modeling.prediction_agreement(y1_predictions,y2_predictions)

        y1_predictions,y2_predictions = model.predict(x_test)
        val_agreement = modeling.prediction_agreement(y1_predictions,y2_predictions)

        y1_predictions,y2_predictions = model.predict(corrupted_images)
        corrupted_agreement = modeling.prediction_agreement(y1_predictions,y2_predictions)

        history = history.history
        json.dump(history,open('../experiment_data/mnist_dense_2_pretrained_width_32/histories/beta_'+str(beta)+'.json','w'))

        # model.save('../experiments/mnist_dense_1_width_16/models/hmodel_beta_'+str(beta))

        best_epoch = np.argmin(history['val_loss'])
        stats.loc[len(stats)] = [beta,
                                 history['val_rep_loss'][best_epoch],
                                 history['val_y1_sparse_categorical_accuracy'][best_epoch],
                                 history['val_y2_sparse_categorical_accuracy'][best_epoch],
                                 history['distillation_loss'][best_epoch],
                                 history['val_distillation_loss'][best_epoch],
                                 train_agreement,
                                 val_agreement,
                                 corrupted_agreement,
                                 best_epoch]
        stats.to_csv('../experiment_data/mnist_dense_2_pretrained_width_32/stats.csv')

        del model_A
        del model_A1
        del model_A2
        del model_B
        del model_B1
        del model_B2
        del model
        gc.collect()

if __name__ == '__main__':
    main()