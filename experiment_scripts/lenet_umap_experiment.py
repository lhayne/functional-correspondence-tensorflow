import tensorflow as tf
import tensorflow_datasets as tfds
from util import modeling
import gc
import json
import pandas as pd
import numpy as np
import pickle

def main():
    """
    Experiment to test if we can jointly optimize for the distance between two representations
    in two separate networks AND their in distribution performance.
    """
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Load mnist train, test, corrupted test
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0],-1))
    x_test = x_test.reshape((x_test.shape [0],-1))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    fog = tfds.as_numpy(tfds.load('mnist_corrupted/fog',split='test',data_dir='../data/tensorflow_datasets',batch_size=-1))
    x_fog = fog['image'].astype("float32") / 255
    y_fog = fog['label']
    x_fog = x_fog.reshape((x_fog.shape[0],-1))

    stripe = tfds.as_numpy(tfds.load('mnist_corrupted/stripe',split='test',data_dir='../data/tensorflow_datasets',batch_size=-1))
    x_stripe = stripe['image'].astype("float32") / 255
    y_stripe = stripe['label']
    x_stripe = x_stripe.reshape((x_stripe.shape[0],-1))

    blur = tfds.as_numpy(tfds.load('mnist_corrupted/motion_blur',split='test',data_dir='../data/tensorflow_datasets',batch_size=-1))
    x_blur = blur['image'].astype("float32") / 255
    y_blur = blur['label']
    x_blur = x_blur.reshape((x_blur.shape[0],-1))

    rotate = tfds.as_numpy(tfds.load('mnist_corrupted/rotate',split='test',data_dir='../data/tensorflow_datasets',batch_size=-1))
    x_rotate = rotate['image'].astype("float32") / 255
    y_rotate = rotate['label']
    x_rotate = x_rotate.reshape((x_rotate.shape[0],-1))

    stats = pd.DataFrame([],columns=['model','beta','rep_loss','val_rep_loss','y1_acc','y2_acc',
                                     'distillation_loss','test_distillation_loss',
                                     'train_agreement','test_agreement','fog_agreement',
                                     'stripe_agreement','blur_agreement','rotate_agreement',
                                     'epochs'])

    for m in range(1):
        for beta in list(np.logspace(-10,-1,10)):
            print('BETA',beta)
            model_A = tf.keras.models.load_model('../models/lenet_iteration_'+str(m))
            print(model_A.layers)
            model_A.build((None,784))
            model_A1 = tf.keras.models.Model(model_A.inputs,model_A.layers[1].output)
            model_A2 = tf.keras.models.Model(model_A.layers[2].input,model_A.output)

            model_B = tf.keras.models.load_model('../models/lenet_iteration_'+str(m))
            model_B.build((None,784))
            model_B1 = tf.keras.models.Model(model_B.inputs,model_B.layers[1].output)
            model_B2 = tf.keras.models.Model(model_B.layers[2].input,model_B.output)

            model = modeling.HModel(model_A1=model_A1,model_A2=model_A2,model_B1=model_B1,model_B2=model_B2)

            model_A1.build((None,784))
            model_B1.build((None,784))
            model_A2.build((None,100))
            model_B2.build((None,100))
            model.build((None,784))

            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                y1_metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                y2_metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                rep_loss_fn=modeling.lin_cka_dist_2,
                y1_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(), 
                y2_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
                distillation_loss_fn=tf.keras.losses.KLDivergence(),
                alpha=0.5,
                beta=beta,
            )

            history = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=256,epochs=100,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                                            monitor='val_loss',
                                            mode='min',
                                            min_delta=0,
                                            patience=20,
                                            restore_best_weights=True,
                    )])

            y1_predictions,y2_predictions = model.predict(x_train)
            train_agreement = modeling.prediction_agreement(y1_predictions,y2_predictions).numpy()

            y1_predictions,y2_predictions = model.predict(x_test)
            test_agreement = modeling.prediction_agreement(y1_predictions,y2_predictions).numpy()

            y1_predictions,y2_predictions = model.predict(x_fog)
            fog_agreement = modeling.prediction_agreement(y1_predictions,y2_predictions).numpy()

            y1_predictions,y2_predictions = model.predict(x_stripe)
            stripe_agreement = modeling.prediction_agreement(y1_predictions,y2_predictions).numpy()

            y1_predictions,y2_predictions = model.predict(x_blur)
            blur_agreement = modeling.prediction_agreement(y1_predictions,y2_predictions).numpy()

            y1_predictions,y2_predictions = model.predict(x_rotate)
            rotate_agreement = modeling.prediction_agreement(y1_predictions,y2_predictions).numpy()

            history = history.history
            json.dump(history,open('../histories/lenet_procrustes_iteration_'+str(m)+'_beta_'+str(beta)+'.json','w'))

            best_epoch = np.argmin(history['val_loss'])
            stats.loc[len(stats)] = [m,beta,
                                    history['rep_loss'][best_epoch],
                                    history['val_rep_loss'][best_epoch],
                                    history['val_y1_sparse_categorical_accuracy'][best_epoch],
                                    history['val_y2_sparse_categorical_accuracy'][best_epoch],
                                    history['distillation_loss'][best_epoch],
                                    history['val_distillation_loss'][best_epoch],
                                    train_agreement,
                                    test_agreement,
                                    fog_agreement,
                                    stripe_agreement,
                                    blur_agreement,
                                    rotate_agreement,
                                    best_epoch]
            stats.to_csv('../summary_stats/lenet_cka_stats.csv')

            A_train_activations = model_A1.predict(x_train)
            A_test_activations = model_A1.predict(x_test)
            A_fog_activations = model_A1.predict(x_fog)
            A_stripe_activations = model_A1.predict(x_stripe)
            A_blur_activations = model_A1.predict(x_blur)
            A_rotate_activations = model_A1.predict(x_rotate)

            B_train_activations = model_B1.predict(x_train)
            B_test_activations = model_B1.predict(x_test)
            B_fog_activations = model_B1.predict(x_fog)
            B_stripe_activations = model_B1.predict(x_stripe)
            B_blur_activations = model_B1.predict(x_blur)
            B_rotate_activations = model_B1.predict(x_rotate)

            # Save activations
            pickle.dump({'activations':A_train_activations,'labels':y_train},
                       open('../activations/cka/A_train_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))
            pickle.dump({'activations':A_test_activations,'labels':y_test},
                       open('../activations/cka/A_test_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))
            pickle.dump({'activations':A_fog_activations,'labels':y_fog},
                       open('../activations/cka/A_fog_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))
            pickle.dump({'activations':A_stripe_activations,'labels':y_stripe},
                       open('../activations/cka/A_stripe_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))
            pickle.dump({'activations':A_blur_activations,'labels':y_blur},
                       open('../activations/cka/A_blur_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))
            pickle.dump({'activations':A_rotate_activations,'labels':y_rotate},
                       open('../activations/cka/A_rotate_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))

            pickle.dump({'activations':B_train_activations,'labels':y_train},
                       open('../activations/cka/B_train_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))
            pickle.dump({'activations':B_test_activations,'labels':y_test},
                       open('../activations/cka/B_test_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))
            pickle.dump({'activations':B_fog_activations,'labels':y_fog},
                       open('../activations/cka/B_fog_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))
            pickle.dump({'activations':B_stripe_activations,'labels':y_stripe},
                       open('../activations/cka/B_stripe_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))
            pickle.dump({'activations':B_blur_activations,'labels':y_blur},
                       open('../activations/cka/B_blur_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))
            pickle.dump({'activations':B_rotate_activations,'labels':y_rotate},
                       open('../activations/cka/B_rotate_iteration_'+str(m)+'_beta_'+str(beta)+'.pkl','wb'))

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