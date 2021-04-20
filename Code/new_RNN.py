"""
***IMPORTANT***
When running this code, change the parameter of the function on line 78/79
to be the string of the path to the file containing our dataset. The file name
is called bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv.

This code was crafted primarily from the code provided on this site:
https://github.com/Nishil07/Simple-Rnn-for-my-first-Medium-blog/blob/master/Simple_RNN.ipynb
These code pieces were made by Nishil07.

Most of the code below was from this site with the exception of extract_data(),
plot_results(), and a decent portion of train_and_run().
"""


import matplotlib
import torch
from jedi.api.refactoring import inline
from torch import nn
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)

        # last, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)

        r_out = r_out.view(-1, self.hidden_dim)

        # get final output
        output = self.fc(r_out)

        return output, hidden


def plot_results(losses):
    """
    Plot the losses over epochs.
    :param losses: losses of the model.
    """
    plt.plot(losses, label='Training loss')
    plt.title('Training loss over epochs')
    plt.xlabel('Epochs', fontsize=40)
    plt.legend()

    plt.show()


def extract_data(size):
    """
    Extract data and return it in the correct format.
    :param size: how large the dataset should be.
    :param num_values_to_predict: number of future values to predict.
    :return: the list of closing prices of the cryptocurrency (recorded between
    equal periods of time) (in 2 lists: dataset and test labels).
    """
    # enter correct path to the dataset to be used
    df = pd.read_csv(
        "/Users/haiderraza/Desktop/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")
    df_np = df.to_numpy()

    # taking just the closing prices
    new_df = df_np[:, 4]
    # print(np.count_nonzero(np.isnan(new_df[4757376:4857376]))))

    # there are 131 NaN values in the last 100k values
    # taking the last 100k values
    new_dff = new_df[4757376:4857376]

    # taking the consecutive 10 NaN values out manually
    new_df2 = new_dff[:90522]
    new_df2 = np.append(new_df2, new_df2[90531:])

    # Set dataset size
    new_df2 = new_df2[len(new_df2) - size:]

    # replace 121 (previously 131) NaN values with their successor's value
    index_NaN = np.argwhere(np.isnan(new_df2))
    index_NaN = index_NaN.reshape((index_NaN.shape[0],))
    # print(index_NaN)

    for i in range(len(index_NaN)):
        j = index_NaN[i]
        if j + 1 in index_NaN:
            new_df2[j] = new_df2[j + 2]
        else:  # if j+1 is not NaN
            new_df2[j] = new_df2[j + 1]
    assert np.count_nonzero(np.isnan(new_df2)) == 0
    # print(new_df2.shape)
    data_ = new_df2.tolist()
    data_.resize((seq_length + 1, 1))

    return data_


# train the RNN
def train_and_test(rnn, n_steps, size):
    # initialize the hidden state
    hidden = None
    data = extract_data(size)
    test_data = data[len(data) - (seq_length + 1):]
    data = data[:len(data) - (seq_length + 1)]
    training_losses = []
    for step in range(n_steps):
        # defining the training data
        batch_data = data[step * (seq_length + 1):(step + 1) * (seq_length + 1)]

        x = batch_data[:-1]
        y = batch_data[1:]

        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(
            0)  # unsqueeze gives a 1, batch_size dimension
        y_tensor = torch.Tensor(y)

        # outputs from the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)
        # zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # display loss and predictions
        print(f'Epoch: {step + 1}, Training Loss: {loss.item()}')
        training_losses.append(loss.item())

    x = test_data[:-1]
    y = test_data[1:]

    # convert data into Tensors
    x_tensor = torch.Tensor(x).unsqueeze(
        0)  # unsqueeze gives a 1, batch_size dimension
    y_tensor = torch.Tensor(y)

    # outputs from the rnn
    prediction, hidden = rnn(x_tensor, hidden)

    ## Representing Memory ##
    # make a new variable for hidden and detach the hidden state from its history
    # this way, we don't backpropagate through the entire history
    hidden = hidden.data

    # calculate the loss
    testing_loss = criterion(prediction, y_tensor)

    return rnn, training_losses, testing_loss

# TODO: Run the code with size_ = 5000 and size_ = 30000 when testing!


size_ = 5000  # Try with 30000 as well!
n_steps_ = 100
# how many time steps/data pts are in one batch of data
seq_length = math.floor(size_ / (n_steps_ + 1)) - 1


# decide on hyperparameters
input_size = 1
output_size = 1
hidden_dim = 32
n_layers = 1

# instantiate an RNN
rnn_ = RNN(input_size, output_size, hidden_dim, n_layers)
# print(rnn)

# MSE loss and Adam optimizer with a learning rate of 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn_.parameters(), lr=0.001)

# train the rnn and monitor results
trained_rnn, training_losses, test_loss = train_and_test(rnn_, n_steps_, size_)

# Display losses
print(f'Testing loss: {test_loss}')

plot_results(training_losses)
