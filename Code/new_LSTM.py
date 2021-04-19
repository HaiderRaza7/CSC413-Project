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


def split_data(data, split):
    """
    Split data appropriately and return the split lists.
    :param data: list to be split appropriately.
    :param split: portion of data to be dedicated to the first list.
    :return: the split lists.
    """
    index = math.floor(len(data) * split)
    first = data[:index]
    second = data[index:]
    return first, second


def extract_data(split=0.9, val_test_split=0.5, tr=.9, v=0.9, te=0.9):
    """
    Extract data and return it in the correct format.
    :param split: portion of dataset to be dedicated to the training set.
    :param val_test_split: portion of the remaining dataset to be dedicated to the
    validation set.
    :param tr: portion of training set to be dedicated as data points.
    :param v: portion of validation set to be dedicated as data points.
    :param te: portion of testing set to be dedicated as data points.
    :return: the list of closing prices of the cryptocurrency (recorded between
    equal periods of time) (6 lists).
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

    #TODO: Remove this later! Set dataset to 10k sample for easy running
    new_df2 = new_df2[:10000]

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
    data = new_df2.tolist()

    # Now that we have the data, time to split into the datasets
    train, rest = split_data(data, split)
    validation, test = split_data(rest, val_test_split)

    # Split each dataset into data and labels
    # train_data, train_labels = split_data(train, tr)
    # valid_data, valid_labels = split_data(validation, v)
    # test_data, test_labels = split_data(test, te)
    #
    # return train_data, train_labels, valid_data, valid_labels, test_data, \
    #        train_labels

    train = np.array(train)
    validation = np.array(validation)
    test = np.array(test)

    return train, validation, test


def temp():
    train_data, valid_data, test_data = extract_data()
    # Normalize the training dataset for time series predictions
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))

    # Convert into tensors
    train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

    # Number of future values to predict
    train_window = 12

    # Create inout sequences
    train_inout_seq = create_inout_sequences(train_data_normalized,
                                             train_window)

    # Set up the model
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 150
    single_loss = 0  # Initialize just to remove warnings
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


    # Run algorithm on testing set
    fut_pred = 12

    test_inputs = train_data_normalized[-train_window:].tolist()
    print(test_inputs)

    model.eval()

    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item())

    actual_predictions = scaler.inverse_transform(
        np.array(test_inputs[train_window:]).reshape(-1, 1))
    print(actual_predictions)


temp()
