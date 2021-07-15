import numpy as np
from lxmls.deep_learning.rnn import RNN
from lxmls.deep_learning.utils import index2onehot, logsumexp


class NumpyRNN(RNN):

    def __init__(self, **config):
        # This will initialize
        # self.config
        # self.parameters
        RNN.__init__(self, **config)

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        p_y = np.exp(self.log_forward(input)[0])
        return np.argmax(p_y, axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """
        gradients = self.backpropagation(input, output)
        learning_rate = self.config['learning_rate']
        # Update each parameter with SGD rule
        num_parameters = len(self.parameters)
        for m in range(num_parameters):
            # Update weight
            self.parameters[m] -= learning_rate * gradients[m]

    def log_forward(self, input):

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        hidden_size = W_h.shape[0]
        nr_steps = input.shape[0]

        # Embedding layer
        z_e = W_e[input, :]

        # Recurrent layer
        h = np.zeros((nr_steps + 1, hidden_size))
        for t in range(nr_steps):

            # Linear
            z_t = W_x.dot(z_e[t, :]) + W_h.dot(h[t, :])

            # Non-linear
            h[t+1, :] = 1.0 / (1 + np.exp(-z_t))

        # Output layer
        y = h[1:, :].dot(W_y.T)

        # Softmax
        log_p_y = y - logsumexp(y, axis=1, keepdims=True)

        return log_p_y, y, h, z_e, input
    
    def backpropagation(self, input, output):

        '''
        Compute gradientes, with the back-propagation method
        inputs:
            x: vector with the (embedding) indicies of the words of a
                sentence
            outputs: vector with the indicies of the tags for each word of
                        the sentence outputs:
            gradient_parameters: vector with parameters gradientes
        '''

        # Get parameters and sizes
        W_e, W_x, W_h, W_y = self.parameters
        nr_steps = input.shape[0]

        log_p_y, y, h, z_e, x = self.log_forward(input)
        p_y = np.exp(log_p_y)

        # Initialize gradients with zero entrances
        gradient_W_e = np.zeros(W_e.shape)
        gradient_W_x = np.zeros(W_x.shape)
        gradient_W_h = np.zeros(W_h.shape)
        gradient_W_y = np.zeros(W_y.shape)

        # ----------
        # Solution to Exercise 6.1

        # Gradient of the cost with respect to the last linear model
        I = index2onehot(output, W_y.shape[0])
        error = (p_y - I) / nr_steps

        # backward pass, with gradient computation
        error_h_next = np.zeros_like(h[0, :])
        for t in reversed(range(nr_steps)):

            # Output linear
            error_h = np.dot(W_y.T, error[t, :]) + error_h_next

            # Non-linear
            error_raw = h[t+1, :] * (1. - h[t+1, :]) * error_h

            # Hidden-linear
            error_h_next = np.dot(W_h.T, error_raw)

            # Weight gradients
            gradient_W_y += np.outer(error[t, :], h[t+1, :])
            gradient_W_h += np.outer(error_raw, h[t, :])
            gradient_W_x += np.outer(error_raw, z_e[t, :])
            gradient_W_e[x[t], :] += W_x.T.dot(error_raw)

        # End of Solution to Exercise 6.1
        # ----------

        # Normalize over sentence length
        gradient_parameters = [
            gradient_W_e, gradient_W_x, gradient_W_h, gradient_W_y
        ]

        return gradient_parameters

    # def backpropagation(self, input, output):

    #     '''
    #     Compute gradientes, with the back-propagation method
    #     inputs:
    #         x: vector with the (embedding) indicies of the words of a
    #             sentence
    #         outputs: vector with the indicies of the tags for each word of
    #                     the sentence outputs:
    #         gradient_parameters: vector with parameters gradientes
    #     '''

    #     # Get parameters and sizes
    #     W_e, W_x, W_h, W_y = self.parameters
    #     nr_steps = input.shape[0]
    #     nr_labels = W_y.shape[0]

    #     log_p_y, y, h, z_e, x = self.log_forward(input)
    #     p_y = np.exp(log_p_y)

    #     # Initialize gradients with zero entrances
    #     gradient_W_e = np.zeros(W_e.shape)
    #     gradient_W_x = np.zeros(W_x.shape)
    #     gradient_W_h = np.zeros(W_h.shape)
    #     gradient_W_y = np.zeros(W_y.shape)

    #     # ----------
    #     # Solution to Exercise 1
    #     # Note: loops through sequence positions are vectorized

    #     # Initialize the error at last layer for a CE cost and 
    #     # backpropagate it through the output linear layer
    #     ohc_y_true = np.eye(nr_labels)[output]
    #     error = (p_y - ohc_y_true) / nr_steps
    #     e_y = W_y.T @ error.T  # 20x5
        
    #     # Initialize recurrent layer error er to a vector of 
    #     # zeros of size J
    #     e_r = np.zeros((nr_steps, W_h.shape[0]))  # 5X20

    #     # Add the recurrent layer backpropagated error and 
    #     # backpropagate through the sigmoid non-linearity
    #     e_h = (e_r + e_y.T) * h[1:] * (np.ones(h[1:].shape) - h[1:])  # 5x20
    #     # h[1:] removes h0 which is just zeros

    #     # Backpropagate the error through the recurrent linear layer
    #     e_r = W_h.T @ e_h.T  # 20x5

    #     # Backpropagate the error through the input linear layer
    #     e_e = W_x.T @ e_h.T  # 50x5

    #     # Compute the gradients using the backpropagated errors 
    #     # and the inputs from the forward pass
    #     d_W_e = np.zeros_like(gradient_W_e)  # 4786x50
    #     for i, pos in enumerate(x):
    #         d_W_e[pos, :] = e_e.T[i]  # gradient only exists for the words that appear in the input sequence
    #     d_W_x = e_h.T @ z_e  # 20x50
    #     d_W_h = e_h.T @ h[:-1]  # 20x20
    #     d_W_y = error.T @ h[1:]  # 12x20

    #     # Update the parameters
    #     lr = self.config['learning_rate']
    #     gradient_W_e += lr * d_W_e
    #     gradient_W_x += lr * d_W_x
    #     gradient_W_h += lr * d_W_h
    #     gradient_W_y += lr * d_W_y

    #     # End of Solution to Exercise 1
    #     # ----------

    #     # Normalize over sentence length
    #     gradient_parameters = [
    #         gradient_W_e, gradient_W_x, gradient_W_h, gradient_W_y
    #     ]

    #     return gradient_parameters

    def cross_entropy_loss(self, input, output):
        """Cross entropy loss"""
        nr_steps = input.shape[0]
        log_probability = self.log_forward(input)[0]
        return -log_probability[range(nr_steps), output].mean()
