import csv
import sys
import pandas as pd
import numpy as np

def check_distance_between_nonNan(indexofnan: list):
    """ Take the distances between nonNaN values """
    difference_distance = []
    difference_distance_with_index = []
    for i in range(len(indexofnan) - 1):
        difference = indexofnan[i+1] - indexofnan[i]
        difference_distance.append((difference, (indexofnan[i+1], indexofnan[i+1])))
        difference_distance_with_index.append(
            (difference, (indexofnan[i + 1], indexofnan[i + 1])))

    return difference_distance, difference_distance_with_index

def distances_of_1(distances: list):
    """ """
    counter = 0
    for i in range(len(distances)):
        if distances[i] == 1:
            counter += 1
    return counter

def hundred_k_consecutive(distances: list):
    """ """
    indexes_of_hundred_k_consecutive = []
    for i in range(len(distances) - 100000):
        distance = distances[i:i+100000]
        if 1 not in distance:
            pass
        else:
            indexes_of_hundred_k_consecutive.append((i, i+100000))
            print((i, i+100000), )

    return indexes_of_hundred_k_consecutive


if __name__ == '__main__':
    df = pd.read_csv("/Users/haiderraza/Downloads/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")
    df_np = df.to_numpy()
    # print(df_np[0])
    new_df = df_np[:, 4]
    # print(np.count_nonzero(np.isnan(new_df[4757376:4857376]))))
    # there are 131 NaN values in the last 100k values

    # taking the last 100k values
    new_dff = new_df[4757376:4857376]

    # taking the consecutive 10 NaN values out manually
    new_df2 = new_dff[:90522]
    new_df2 = np.append(new_df2, new_dff[90531:])

    # replace 121 (previously 131) NaN values with their successors value
    index_NaN = np.argwhere(np.isnan(new_df2))
    index_NaN = index_NaN.reshape((index_NaN.shape[0],))
    # print(index_NaN)

    for i in range(len(index_NaN)):
        j = index_NaN[i]
        if j+1 in index_NaN:
            new_df2[j] = new_df2[j+2]
        else: # if j+1 is not NaN
            new_df2[j] = new_df2[j + 1]
    assert np.count_nonzero(np.isnan(new_df2)) == 0
    # print(new_df2.shape)
    print(np.count_nonzero(np.isnan(new_df[4657376:4857376])))











    # # count the number of NaN values
    # # print(np.count_nonzero(np.isnan(new_df)))
    #
    # #### check the difference between each NaN value
    #
    # # indexes of each non-NaN values
    # index_of_nonNan = np.argwhere(np.logical_not(np.isnan(new_df)))
    # # reshaping to make into list
    # index_of_nonNan = index_of_nonNan.reshape((index_of_nonNan.shape[0],))
    # # this gives us the distances between nonNaN values
    # distances, distances_with_index = check_distance_between_nonNan(index_of_nonNan.tolist())
    # # print(hundred_k_consecutive(distances))
    #
    # # check which 100k is the most consistent
    #
    #
    #
    #
    #
    # # removing NaN values
    # new_df = new_df[np.logical_not(np.isnan(new_df))]
    # new_df = new_df.tolist()
