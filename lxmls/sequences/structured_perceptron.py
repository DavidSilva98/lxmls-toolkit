from __future__ import division
import sys
import numpy as np
import lxmls.sequences.discriminative_sequence_classifier as dsc
import pdb


class StructuredPerceptron(dsc.DiscriminativeSequenceClassifier):
    """ Implements Structured Perceptron"""

    def __init__(self, observation_labels, state_labels, feature_mapper,
                 num_epochs=10, learning_rate=1.0, averaged=True):
        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_epoch = []

    def train_supervised(self, dataset):
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        num_examples = dataset.size()
        for epoch in range(self.num_epochs):
            num_labels_total = 0
            num_mistakes_total = 0
            for i in range(num_examples):
                sequence = dataset.seq_list[i]
                num_labels, num_mistakes = self.perceptron_update(sequence)
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
            self.params_per_epoch.append(self.parameters.copy())
            acc = 1.0 - num_mistakes_total / num_labels_total
            print("Epoch: %i Accuracy: %f" % (epoch, acc))
        self.trained = True

        if self.averaged:
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w /= len(self.params_per_epoch)
            self.parameters = new_w

    def perceptron_update(self, sequence):
        """Applies one round of the perceptron algorithm, updating the 
        weights for a given sequences, and returning the number of
        predicted labels (which equals the sequence length) and the 
        number of mistaken labels.
        """
        # ----------
        # Solution to Exercise 3
        num_labels = len(sequence)

        # Predict the y labels through the viterbi algorithm using the 
        # current parameter values - 13.5
        pred_seq, _ = self.viterbi_decode(sequence)

        # Update the parameters - 13.6
        y_true, y_hat = sequence.y, pred_seq.y
        error_idx = [i for i in range(num_labels) if y_hat[i] != y_true[i]]
        num_mistakes = len(error_idx)

        for i in error_idx:
            if i == 0:
                # Update initial features
                true_initial_features = self.feature_mapper.get_initial_features(sequence, y_true[i])
                self.parameters[true_initial_features] += self.learning_rate
                hat_initial_features = self.feature_mapper.get_initial_features(sequence, y_hat[i])
                self.parameters[hat_initial_features] -= self.learning_rate

            elif i == num_labels - 1:
                # Update final features
                true_final_features = self.feature_mapper.get_final_features(sequence, y_true[i])
                self.parameters[true_final_features] += self.learning_rate
                hat_final_features = self.feature_mapper.get_final_features(sequence, y_hat[i])
                self.parameters[hat_final_features] -= self.learning_rate

            else:
                # Update emission features
                true_emission_features = self.feature_mapper.get_emission_features(sequence, i, y_true[i])
                self.parameters[true_emission_features] += self.learning_rate
                hat_emission_features = self.feature_mapper.get_emission_features(sequence, i, y_hat[i])
                self.parameters[hat_emission_features] -= self.learning_rate

                if i > 0:
                    # Update transition features
                    # Update when mismatch at time t
                    true_transition_features = self.feature_mapper.get_transition_features(
                        sequence, i-1, y_true[i], y_true[i-1]
                    )
                    self.parameters[true_transition_features] += self.learning_rate
                    hat_transition_features = self.feature_mapper.get_transition_features(
                        sequence, i-1, y_hat[i], y_hat[i-1]
                    )
                    self.parameters[hat_transition_features] -= self.learning_rate

                    # Update when mismatch at time t-1 (if t isn't a mismatch to avoid duplication)
                    if i + 1 not in error_idx:
                        true_transition_features = self.feature_mapper.get_transition_features(
                        sequence, i, y_true[i+1], y_true[i]
                        )
                        self.parameters[true_transition_features] += self.learning_rate
                        hat_transition_features = self.feature_mapper.get_transition_features(
                            sequence, i, y_hat[i+1], y_hat[i]
                        )
                        self.parameters[hat_transition_features] -= self.learning_rate
        # End of Solution to Exercise 3
        # ----------

        return num_labels, num_mistakes

    def save_model(self, dir):
        fn = open(dir + "parameters.txt", 'w')
        for p_id, p in enumerate(self.parameters):
            fn.write("%i\t%f\n" % (p_id, p))
        fn.close()

    def load_model(self, dir):
        fn = open(dir + "parameters.txt", 'r')
        for line in fn:
            toks = line.strip().split("\t")
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()
