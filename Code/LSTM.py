"""
This entire file (LSTM.py) is a combination of the lstm.py and test.py file that
 can be found here:
https://github.com/nicodjimenez/lstm
Please note that github user nicodjimenez is the original creator of this
piece of code and that we are simply using it.
Most (if not all) of our changes were to replace the example_0() function with
run_lstm() to suit our needs for the project.
"""

import random

import numpy as np
import math
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_derivative(values):
    return values * (1 - values)


def tanh_derivative(values):
    return 1. - values ** 2


# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a


class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len))
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))
        self.bg_diff = np.zeros(mem_cell_ct)
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

    def apply_diff(self, lr=1.0):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)


class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)


class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def bottom_data_is(self, x, s_prev=None, h_prev=None):
        # if this is the first lstm node in the network
        if s_prev is None:
            s_prev = np.zeros_like(self.state.s)
        if h_prev is None:
            h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x, h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.xc = xc

    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]


class LstmNetwork():
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        # input sequence
        self.x_list = []

    def y_list_is(self, y_list, loss_layer):
        """
        Updates diffs by setting target sequence
        with corresponding loss layer.
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        # first node only gets diffs from label ...
        loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h,
                                        y_list[idx])
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h,
                                    y_list[idx])
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h,
                                            y_list[idx])
            diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(self.lstm_param.mem_cell_ct,
                                   self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        # get index of most recent x input
        idx = len(self.x_list) - 1
        if idx == 0:
            # no recurrent inputs yet
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)


# Code from lstm.py ends here
# Code from test.py begins here

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def split_data(data, split=0.9):
    """
    Returns a split version of data (which should be a list
    :param data: list containing the prices of the cryptocurrency in question
    over time (separated by equal periods of time)
    :param split: what fraction of the original dataset should be assigned to
    the training dataset
    :return: the training dataset, training labels, validation dataset, and
    validation labels in that order.
    """
    if split >= 1.0:
        print("Invalid split value")
        return [], [], [], []
    if len(data) % 2 == 1:  # this algorithm works best with even amount of data
        data.pop()
    index = math.floor(split * len(data))
    train = data[:index]
    validation = data[index:]
    train_data = train[:len(train) // 2]
    train_labels = train[len(train) // 2:]
    valid_data = validation[:len(validation) // 2]
    valid_labels = validation[len(validation) // 2:]
    # Modify training and validation data so that it may be used by the
    # algorithm
    for i in range(len(train_data)):
        train_data[i] = np.array([train_data[i]])
    for j in range(len(valid_data)):
        valid_data[j] = np.array([valid_data[j]])
    return train_data, train_labels, valid_data, valid_labels


def extract_data():
    """
    Extract the dataset from the csv file
    :return: the list of closing prices of the cryptocurrency (recorded between
    equal periods of time)
    """
    # TODO: IMPLEMENT
    # TODO: This is where you get the contents of csv file and turn it into
    # TODO: a list of closing prices only

    data = [1.34, 2.43, 0.3, 4.34, 4.3, 2.34, 3.45, 4.6, 8.65, 12.04, 7.34, 4.5, 1.34, 2.43, 0.3, 4.34, 4.3, 2.34, 3.45, 4.6, 8.65, 12.04, 7.34, 4.5, 1.34, 2.43, 0.3, 4.34, 4.3, 2.34, 3.45, 4.6, 8.65, 12.04, 7.34, 4.5]
    return data


def plot_results(train_losses, valid_losses):
    """
    Plots the training and validation losses of the LSTM algorithm over the
    epochs.
    """
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def run_lstm():
    """
    Trains this LSTM model to predict bitcoin prices.
    """
    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100  # cell count
    x_dim = 1  # dimension of input (probably should change to 1)
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    # Set up the training and validation data
    data = extract_data()
    train_data, train_labels, valid_data, valid_labels = split_data(data,
                                                                    split=0.9)

    # Set up other variables
    train_losses = []
    validation_losses = []
    epochs = 100
    for cur_iter in range(epochs):  # 100 epochs
        # First deal with training dataset
        print("iter", "%2s" % str(cur_iter), end=": ")
        for ind in range(len(train_data)):  # load dataset into lstm
            lstm_net.x_list_add(train_data[ind])
        loss = lstm_net.y_list_is(train_labels, ToyLossLayer)
        train_losses.append(loss)
        print("Training loss:", "%.3e" % loss, end=", ")  # our loss
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()  # reset dataset for next iteration

        # Find y_pred and loss for the validation dataset
        for i in range(len(valid_data)):
            lstm_net.x_list_add(valid_data[i])
        loss = lstm_net.y_list_is(valid_labels, ToyLossLayer)
        validation_losses.append(loss)
        print("Validation loss:", "%.3e" % loss, end=", ")
        lstm_net.x_list_clear()  # reset dataset for next iteration
        print("y_pred = [" +
              ", ".join(
                  ["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in
                   range(len(valid_labels))]) +
              "]")
    plot_results(train_losses, validation_losses)  # Plot results


if __name__ == "__main__":
    run_lstm()
