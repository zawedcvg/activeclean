import numpy as np
import tensorflow as tf
"""
Seeks to find a vector of model parameters by minimising a loss function over all training examples
ideal_model_param_vector = min(sum(loss) over N training examples) + get_regularisation_term(model_param_vector)
model_param refers to e.g. in a linear regression, the value of m and c in y = mx + c
"""
class Updater:
    def __init__(self, clf, batch_size, step_size, calculate_loss):
        """
        Initialises an Updater object
        :param clf: A model from user with regularisation set in place
        :param batch_size:  size of dirty data sample
        :type batch_size: int
        :param step_size: learning rate in SGD algorithm
        :param model_param: A vector of model parameters
        :type model_param: A vector of integers
        :param sample: A tuple of (X_sampled, Y_sampled) where the sampled data has been cleaned and each row of
        X_sampled is a featurised training data point
        :type sample: tuple
        :param cleaned: A tuple of (X_cleaned, Y_cleaned) where each row of X_cleaned is a featurised training data
        point
        """
        self.clf = clf
        self.opt = tf.keras.optimizers.SGD(learning_rate=step_size)
        self.calculate_loss = calculate_loss # e.g. tensorflow's hinge(y_true, y_pred)
        self.batch_size = batch_size
        self.step_size = step_size

    def get_gradients(self, data):
        X, Y = data[0], data[1]
        for x, y in zip(X, Y):
            # Open a GradientTape.
            with tf.GradientTape() as tape:
                # Forward pass.
                logits = self.clf.fit(x,y)
                # Loss value for this batch.
                loss_value = self.calculate_loss(y, logits)

            # Get gradients of loss wrt the weights.
            gradients = tape.gradient(loss_value, self.clf.trainable_weights)

        return gradients

    def evaluate_sample_gradient(self, sample_data, proportion_dirty, sampling_prob):
        gradients = self.get_gradients(sample_data)
        processed_sample_grads = [proportion_dirty * sampling_prob * gradient for (sampling_prob, gradient) in zip(sampling_prob, gradients)]
        return processed_sample_grads

    def evaluate_cleaned_gradient(self, cleaned_data, proportion_cleaned):
        gradients = self.get_gradients(cleaned_data)
        processed_cleaned_grads = [proportion_cleaned * gradient for gradient in gradients]
        return processed_cleaned_grads

    def get_regularisation_term(self):
        pass

    def update(self, sample_data, sampling_prob, cleaned_data, proportions):
        proportion_cleaned, proportion_dirty = proportions
        sample_gradients = self.evaluate_sample_gradient(sample_data, proportion_dirty, sampling_prob)
        clean_gradients = self.evaluate_cleaned_gradient(cleaned_data, proportion_cleaned)
        final_gradients = [sum(gradients) for gradients in zip(sample_gradients, clean_gradients)]

        # Update the weights of the model
        self.opt = self.opt.apply_gradients(zip(final_gradients, self.clf.trainable_weights))
        return self.clf.compile(loss=self.calculate_loss, optimizer=self.opt)
