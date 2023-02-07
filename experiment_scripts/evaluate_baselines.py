import tensorflow as tf
import tensorflow_datasets as tfds
from util import modeling
import pandas as pd

def main():
    # Load mnist train, test, corrupted test
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0],-1))
    x_test = x_test.reshape((x_test.shape [0],-1))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    fog = tfds.as_numpy(tfds.load('mnist_corrupted/fog',split='test',data_dir='../data/tensorflow_datasets',batch_size=-1))
    x_fog = fog['image'].astype("float32") / 255
    x_fog = x_fog.reshape((x_fog.shape[0],-1))
    y_fog = fog['label']

    stripe = tfds.as_numpy(tfds.load('mnist_corrupted/stripe',split='test',data_dir='../data/tensorflow_datasets',batch_size=-1))
    x_stripe = stripe['image'].astype("float32") / 255
    x_stripe = x_stripe.reshape((x_stripe.shape[0],-1))
    y_stripe = stripe['label']

    blur = tfds.as_numpy(tfds.load('mnist_corrupted/motion_blur',split='test',data_dir='../data/tensorflow_datasets',batch_size=-1))
    x_blur = blur['image'].astype("float32") / 255
    x_blur = x_blur.reshape((x_blur.shape[0],-1))
    y_blur = blur['label']

    rotate = tfds.as_numpy(tfds.load('mnist_corrupted/rotate',split='test',data_dir='../data/tensorflow_datasets',batch_size=-1))
    x_rotate = rotate['image'].astype("float32") / 255
    x_rotate = x_rotate.reshape((x_rotate.shape[0],-1))
    y_rotate = rotate['label']

    stats = pd.DataFrame([],columns=['model_1','model_2','train_accuracy_1','train_accuracy_2',
                                     'test_accuracy_1','test_accuracy_2','fog_accuracy_1',
                                     'fog_accuracy_2','stripe_accuracy_1','stripe_accuracy_2',
                                     'motion_blur_accuracy_1','motion_blur_accuracy_1',
                                     'rotate_accuracy_1','rotate_accuracy_1','train_agreement',
                                     'test_agreement','fog_agreement','stripe_agreement',
                                     'motion_blur_agreement','rotate_agreement'])

    model_predictions = {}
    # For every model
    for i in range(20):
        model_predictions[i] = {}
        model = tf.keras.models.load_model('../models/lenet_iteration_'+str(i))

        # Calculate train, test, and corrupted test accuracy
        model_predictions[i]['train'] = model.predict(x_train)
        model_predictions[i]['test'] = model.predict(x_test)
        model_predictions[i]['fog'] = model.predict(x_fog)
        model_predictions[i]['stripe'] = model.predict(x_stripe)
        model_predictions[i]['motion_blur'] = model.predict(x_blur)
        model_predictions[i]['rotate'] = model.predict(x_rotate)

    m = tf.keras.metrics.SparseCategoricalAccuracy()

    # For every pair of models
    for mi in range(20):
        m.update_state(y_train, model_predictions[mi]['train'])
        mi_train_accuracy = m.result().numpy()
        m.reset_states()

        m.update_state(y_test, model_predictions[mi]['test'])
        mi_test_accuracy = m.result().numpy()
        m.reset_states()

        m.update_state(y_fog, model_predictions[mi]['fog'])
        mi_fog_accuracy = m.result().numpy()
        m.reset_states()

        m.update_state(y_stripe, model_predictions[mi]['stripe'])
        mi_stripe_accuracy = m.result().numpy()
        m.reset_states()

        m.update_state(y_blur, model_predictions[mi]['motion_blur'])
        mi_blur_accuracy = m.result().numpy()
        m.reset_states()

        m.update_state(y_rotate, model_predictions[mi]['rotate'])
        mi_rotate_accuracy = m.result().numpy()
        m.reset_states()

        for mj in range(mi+1,20):
            print("MODELS",mi,mj)
            m.update_state(y_train, model_predictions[mj]['train'])
            mj_train_accuracy = m.result().numpy()
            m.reset_states()

            m.update_state(y_test, model_predictions[mj]['test'])
            mj_test_accuracy = m.result().numpy()
            m.reset_states()

            m.update_state(y_fog, model_predictions[mj]['fog'])
            mj_fog_accuracy = m.result().numpy()
            m.reset_states()

            m.update_state(y_stripe, model_predictions[mj]['stripe'])
            mj_stripe_accuracy = m.result().numpy()
            m.reset_states()

            m.update_state(y_blur, model_predictions[mj]['motion_blur'])
            mj_blur_accuracy = m.result().numpy()
            m.reset_states()

            m.update_state(y_rotate, model_predictions[mj]['rotate'])
            mj_rotate_accuracy = m.result().numpy()
            m.reset_states()

            train_agreement = modeling.prediction_agreement(model_predictions[mi]['train'],
                                                            model_predictions[mj]['train']).numpy()

            test_agreement = modeling.prediction_agreement(model_predictions[mi]['test'],
                                                            model_predictions[mj]['test']).numpy()

            fog_agreement = modeling.prediction_agreement(model_predictions[mi]['fog'],
                                                            model_predictions[mj]['fog']).numpy()

            stripe_agreement = modeling.prediction_agreement(model_predictions[mi]['stripe'],
                                                            model_predictions[mj]['stripe']).numpy()

            blur_agreement = modeling.prediction_agreement(model_predictions[mi]['motion_blur'],
                                                            model_predictions[mj]['motion_blur']).numpy()

            rotate_agreement = modeling.prediction_agreement(model_predictions[mi]['rotate'],
                                                            model_predictions[mj]['rotate']).numpy()

            stats.loc[len(stats)] = [mi,mj,mi_train_accuracy,mj_train_accuracy,
                                     mi_test_accuracy,mj_test_accuracy,mi_fog_accuracy,
                                     mj_fog_accuracy,mi_stripe_accuracy,mj_stripe_accuracy,
                                     mi_blur_accuracy,mj_blur_accuracy,mi_rotate_accuracy,
                                     mj_rotate_accuracy,train_agreement,test_agreement,
                                     fog_agreement,stripe_agreement,blur_agreement,rotate_agreement]

            stats.to_csv('../summary_stats/lenet_baseline_stats.csv')


if __name__=="__main__":
    main()