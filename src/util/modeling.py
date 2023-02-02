import tensorflow as tf

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
    num_agreement = tf.math.count_nonzero(y1_predictions==y2_predictions)
    return num_agreement/tf.size(y1_predictions, out_type=tf.dtypes.float32)

def lin_cka_dist_2(A, B):
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
    similarity = tf.linalg.trace(tf.matmul(tf.matmul(A, tf.transpose(A)), tf.matmul(B, tf.transpose(B))))
    normalization = tf.multiply(tf.norm(tf.matmul(A, tf.transpose(A)), ord='fro',axis=(0,1)), 
                                tf.norm(tf.matmul(B, tf.transpose(B)), ord='fro',axis=(0,1)))
    distance = tf.subtract(1.0, tf.divide(similarity, normalization + 1e-10))

    return distance


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
            
            rep_loss = self.rep_loss_fn(rep_1_flat,rep_2_flat)
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

            loss = (self.alpha * ((y1_loss + y2_loss) / 2.0) + (1 - self.alpha) * distillation_loss) - self.beta * rep_loss

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

        rep_loss = self.rep_loss_fn(rep_1_flat,rep_2_flat)
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