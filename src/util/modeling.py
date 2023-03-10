import tensorflow as tf
from inspect import signature

class LinearCenteredKernelAlignment(tf.keras.metrics.Metric):

    def __init__(self, name='linear_centered_kernel_alignment', **kwargs):
        super(LinearCenteredKernelAlignment,self).__init__(name=name, **kwargs)
        self.distance = self.add_weight(name='dist', initializer='zeros')

    def update_state(self, A, B):
        similarity = tf.linalg.trace(tf.matmul(tf.matmul(A, tf.transpose(A)), tf.matmul(B, tf.transpose(B))))
        normalization = tf.multiply(tf.norm(tf.matmul(A, tf.transpose(A)), ord='fro'), 
                                    tf.norm(tf.matmul(B, tf.transpose(B)), ord='fro'))
        dist = tf.subtract(1, tf.divide(similarity, normalization))

        self.distance.assign_add(dist)

    def result(self):
        return self.distance

def prediction_agreement(y1_predictions,y2_predictions):
    """
    Calculates the agreement between two network predictions.
    """
    y1_predictions,y2_predictions = tf.argmax(y1_predictions,axis=1),tf.argmax(y2_predictions,axis=1)
    num_agreement = tf.math.count_nonzero(y1_predictions==y2_predictions,dtype=tf.dtypes.float32)
    return num_agreement/tf.size(y1_predictions, out_type=tf.dtypes.float32)

def lin_cka_dist_2(A, B, center=True):
    """
    Computes Linear CKA distance bewteen representations A and B
    based on the reformulation of the Frobenius norm term from Kornblith et al. (2018)
    np.linalg.norm(A.T @ B, ord="fro") ** 2 == np.trace((A @ A.T) @ (B @ B.T))
    
    Parameters
    ----------
    A : examples x neurons
    B : examples x neurons

    Original Code from Ding et al. (2021)
    -------------
    similarity = np.linalg.norm(B @ A.T, ord="fro") ** 2
    normalization = np.linalg.norm(A @ A.T, ord="fro") * np.linalg.norm(B @ B.T, ord="fro")
    """
    if center:
        A = tf.subtract(A, tf.reduce_mean(A,axis=0))
        B = tf.subtract(B, tf.reduce_mean(B,axis=0))
    similarity = tf.linalg.trace(tf.matmul(tf.matmul(A, tf.transpose(A)), tf.matmul(B, tf.transpose(B))))
    normalization = tf.multiply(tf.norm(tf.matmul(A, tf.transpose(A)), ord='fro',axis=(0,1)), 
                                tf.norm(tf.matmul(B, tf.transpose(B)), ord='fro',axis=(0,1)))
    distance = tf.subtract(1.0, tf.divide(similarity, normalization + 1e-10))

    return distance


def procrustes_2(A, B):
    """
    Computes Procrustes distance bewteen representations A and B
    for when |neurons| >> |examples| and A.T @ B too large to fit in memory.
    Based on:
         np.linalg.norm(A.T @ B, ord="nuc") == np.sum(np.sqrt(np.linalg.eig(((A @ A.T) @ (B @ B.T)))[0]))
    
    Parameters
    ----------
    A : examples x neurons
    B : examples x neurons

    Original Code
    -------------    
    nuc = np.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    """
    A_centered = tf.subtract(A, tf.reduce_mean(A,axis=0,keepdims=True))
    A_normalized = tf.divide(A_centered, tf.norm(A, ord='fro',axis=(0,1))+1e-10)

    B_centered = tf.subtract(B, tf.reduce_mean(B,axis=0))
    B_normalized = tf.divide(B_centered, tf.norm(B, ord='fro',axis=(0,1))+1e-10)

    A_sq_frob = tf.math.reduce_sum(tf.pow(A_normalized, 2))
    B_sq_frob = tf.math.reduce_sum(tf.pow(B_normalized, 2))
    # eig = tf.linalg.eig(tf.matmul(
    #     tf.matmul(A_normalized, tf.transpose(A_normalized)), 
    #     tf.matmul(B_normalized, tf.transpose(B_normalized)))
    #     )[0]
    # nuc = tf.math.reduce_sum(tf.math.sqrt(tf.math.abs(eig)))
    nuc = tf.math.reduce_sum(tf.linalg.svd(tf.matmul(A_normalized, 
                                                     tf.transpose(B_normalized)), compute_uv=False))
    return A_sq_frob + B_sq_frob - 2 * nuc


class HModel(tf.keras.Model):
    def __init__(self, model_A1, model_A2, model_B1, model_B2):
        super().__init__()
        self.model_A1 = model_A1
        self.model_A2 = model_A2
        self.model_B1 = model_B1
        self.model_B2 = model_B2

    def call(self, x, training=None):
        rep_1 = self.model_A1(x, training=training)  # Forward pass
        rep_2 = self.model_B1(x, training=training)
        y_pred_1 = self.model_A2(rep_1, training=training)
        y_pred_2 = self.model_B2(rep_2, training=training)
        
        return [y_pred_1, y_pred_2]

    def compile(self, 
                optimizer,
                y1_metrics,
                y2_metrics,
                rep_loss_fn, 
                y1_loss_fn, 
                y2_loss_fn,
                distillation_loss_fn,
                alpha=0.1,
                temperature=3,
                beta=0.1):
        super(HModel, self).compile(optimizer=optimizer)
        self.y1_metrics = y1_metrics
        self.y2_metrics = y2_metrics
        self.rep_loss_fn = rep_loss_fn
        self.y1_loss_fn = y1_loss_fn
        self.y2_loss_fn = y2_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.beta = beta

    def train_step(self,data):
        x, y = data
        batch_size = tf.shape(x)[0]

        with tf.GradientTape() as tape:
            rep_1 = self.model_A1(x, training=True)  # Forward pass
            rep_1_flat = tf.reshape(rep_1, (batch_size,-1))
            rep_2 = self.model_B1(x, training=True)
            rep_2_flat = tf.reshape(rep_2, (batch_size,-1))

            y_pred_1 = self.model_A2(rep_1, training=True)
            y_pred_2 = self.model_B2(rep_2, training=True)
            
            if len(signature(self.rep_loss_fn).parameters)==2:
                rep_loss = self.rep_loss_fn(rep_1_flat,rep_2_flat)
            else:
                rep_loss = self.rep_loss_fn(rep_1_flat,rep_2_flat,y)

            y1_loss = self.y1_loss_fn(y, y_pred_1)
            y2_loss = self.y2_loss_fn(y, y_pred_2)
            
            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            # See keras tutorial: https://keras.io/examples/vision/knowledge_distillation/
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(y_pred_1 / self.temperature, axis=1),
                    tf.nn.softmax(y_pred_2 / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = (self.alpha * ((y1_loss + y2_loss) / 2.0) + (1 - self.alpha) * distillation_loss) - self.beta * tf.math.log(rep_loss + 1e-10)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics
        for metric in self.y1_metrics:
            metric.update_state(y, y_pred_1)
        for metric in self.y2_metrics:
            metric.update_state(y, y_pred_2)

        # Return a dict of performance
        results = {"y1_"+m.name: m.result() for m in self.y1_metrics}
        results.update({"y2_"+m.name: m.result() for m in self.y2_metrics})

        results.update({
            "loss": loss,
            "rep_loss": rep_loss, 
            "y1_loss": y1_loss,
            "y2_loss": y2_loss,
            "distillation_loss": distillation_loss
                })

        return results

    def test_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]

        rep_1 = self.model_A1(x, training=False)
        rep_1_flat = tf.reshape(rep_1, (batch_size,-1))
        rep_2 = self.model_B1(x, training=False)
        rep_2_flat = tf.reshape(rep_2, (batch_size,-1))

        y_pred_1 = self.model_A2(rep_1, training=False)
        y_pred_2 = self.model_B2(rep_2, training=False)

        if len(signature(self.rep_loss_fn).parameters)==2:
            rep_loss = self.rep_loss_fn(rep_1_flat,rep_2_flat)
        else:
            rep_loss = self.rep_loss_fn(rep_1_flat,rep_2_flat,y)

        y1_loss = self.y1_loss_fn(y, y_pred_1)
        y2_loss = self.y2_loss_fn(y, y_pred_2)

        distillation_loss = (
            self.distillation_loss_fn(
                tf.nn.softmax(y_pred_1 / self.temperature, axis=1),
                tf.nn.softmax(y_pred_2 / self.temperature, axis=1),
            )
            * self.temperature**2
        )

        loss = (self.alpha * ((y1_loss + y2_loss) / 2.0) + (1 - self.alpha) * distillation_loss) - self.beta * rep_loss

        # Update the metrics
        for metric in self.y1_metrics:
            metric.update_state(y, y_pred_1)
        for metric in self.y2_metrics:
            metric.update_state(y, y_pred_2)

        # Return a dict of performance
        results = {"y1_"+m.name: m.result() for m in self.y1_metrics}
        results.update({"y2_"+m.name: m.result() for m in self.y2_metrics})
        results.update({
            "loss": loss,
            "rep_loss": rep_loss, 
            "y1_loss": y1_loss,
            "y2_loss": y2_loss,
            "distillation_loss": distillation_loss
                })

        return results


def selective_magnitude(activations,labels):
    """
    Calculates average selectivity (selective magnitude)
    for each unit given its activations.

    Parameters
    ----------
    activations (N x M array):
        matrix of activations for N training examples and 
        M units
    labels (N x C array):
        matrix of one-hot vectors for every training example
        indicating class label

    Returns
    -------
    selective_magnitudes (C x M array):
        average activity of each of M units for C classes
    """
    if len(labels.shape)==1: # if sparse, convert to one-hot
        labels = tf.one_hot(labels,tf.cast(tf.math.reduce_max(labels)+1,tf.int32),axis=-1)

    return tf.math.divide(tf.linalg.matmul(tf.transpose(labels),activations),
                          tf.reshape(tf.math.reduce_sum(labels,axis=0),(-1,1)))


def selectivity(activations,labels):
    """
    Calculates average normalized selectivity
    for each unit given its activations.

    Parameters
    ----------
    activations (N x M array):
        matrix of activations for N training examples and 
        M units
    labels (N x C array):
        matrix of one-hot vectors for every training example
        indicating class label

    Returns
    -------
    selectivities (C x M array):
        average normalized selectivity of each of M units for C classes
    """
    selective_magnitudes = selective_magnitude(activations,labels)
    return tf.math.divide_no_nan(selective_magnitudes, magnitude(selective_magnitudes))


def magnitude(activations):
    return tf.math.reduce_mean(activations,axis=0)


def maximum_selectivity(activations,labels):
    return tf.math.reduce_max(selectivity(activations,labels),axis=0)


def mean_maximum_selectivity(activations,labels):
    return tf.math.reduce_mean(maximum_selectivity(activations,labels))


def mean_maximum_selectivity_difference(A,B,labels):
    return tf.math.subtract(mean_maximum_selectivity(A,labels),
                            mean_maximum_selectivity(B,labels))


def selectivity_index(activations,labels):
    class_conditional_mean_selectivity = tf.sort(selective_magnitude(activations,labels),
                                                 direction='DESCENDING',axis=0)
    mu_max = class_conditional_mean_selectivity[0]
    mu_not_max = tf.math.reduce_mean(class_conditional_mean_selectivity[1:],axis=0)
    SI = tf.math.divide_no_nan(tf.math.subtract(mu_max,mu_not_max),tf.math.add(mu_max,mu_not_max))
    return tf.math.reduce_mean(SI)


def selectivity_index_difference(A,B,labels):
    return tf.math.subtract(selectivity_index(A,labels),
                            selectivity_index(B,labels))