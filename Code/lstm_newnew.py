"""
***IMPORTANT***
When running this code, change the parameter of the function on line 67/68
to be the string of the path to the file containing our dataset. The file name
is called bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv.
This code was crafted primarily from the code provided on this site:
https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
The blog containing these code pieces was made by Usman Malik.
Most of the code below was from this site with the exception of extract_data(),
plot_results(), and a small portion of run_lstm_network().
"""


import matplotlib
import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1,
                                                              -1),
                                               self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


def extract_data(size, num_values_to_predict):
    """
    Extract data and return it in the correct format.
    :param size: how large the dataset should be.
    :param num_values_to_predict: number of future values to predict.
    :return: the list of closing prices of the cryptocurrency (recorded between
    equal periods of time) (in 2 lists: dataset and test labels).
    """
    # enter correct path to the dataset to be used
    df = pd.read_csv(
        "/Users/MustafaImam/Downloads/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")

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
    # np.array(data)

    return new_df2[:len(data_) - num_values_to_predict], \
           new_df2[len(data_) - num_values_to_predict:]


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


def run_lstm_network(size, num_values_to_predict):
    """
    Run the lstm network
    :param size: size of the dataset to deal with
    :param num_values_to_predict: number of values to predict into the future
    """
    train_data, test_labels = extract_data(size, num_values_to_predict)

    # Normalize the training dataset for time series predictions
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

    # Convert into tensors
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

    # Number of future values to predict
    train_window = num_values_to_predict

    # Normalize test labels
    data_normalized = np.append(train_data, test_labels)
    data_normalized = scaler.fit_transform(data_normalized.reshape(-1, 1))
    test_labels_normalized = data_normalized[-train_window:].tolist()

    # Create inout sequences
    train_inout_seq = create_inout_sequences(train_data_normalized,
                                             train_window)

    # Set up the model
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 100
    single_loss = 0  # Initialize just to remove warnings
    train_losses = []
    for i in range(1, epochs + 1):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
        train_losses.append(single_loss.item())


    # Run algorithm on testing set
    fut_pred = num_values_to_predict

    test_inputs = train_data_normalized[-train_window:].tolist()
    # print(test_inputs)

    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    # actual_predictions = scaler.inverse_transform(
    #     np.array(test_inputs[train_window:]).reshape(-1, 1))
    # print(actual_predictions)

    # Calculate and print test loss
    test_loss = 0
    for i in range(num_values_to_predict):
        test_loss += (test_inputs[i] - test_labels_normalized[i][0]) ** 2
    print(f'Test loss: {test_loss:10.10f}')

    # Plot the training losses
    plot_results(train_losses)


# TODO: Run this file in order to run all 4 experiments!


# run_lstm_network(5000, 1)  # Use small dataset and make one prediction
run_lstm_network(5000, 10)  # Use small dataset and make 10 predictions into the future
run_lstm_network(30000, 1)  # Use large dataset and make one prediction
run_lstm_network(30000, 10)  # Use large dataset and make 10 predictions into the future
