"""
This script implements the k-NN method for classifying boxing punches.
It reads the provided data files in the data directory using the python
pickle module.

The script applies the k-NN method and generates some plots using metric
functions from the scikit-learn library and uses the metrics to generate plots.

Two experiments are run in this script:
    1. Varying the k paramter
    2. Examining the difference between the two coordinate systems provided by the
       mediapipe pose extraction model (world coordinates: centered on the midpoint
       of the hips in meters, relatice coordinates: location on the image centered
       om the top left corner)

"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pickle
label_names = ['cross', 'jab', 'lead hook', 'lead uppercut', 'rear hook',
               'rear uppercut', 'stance']

# Labels of the dataset (not included in the pose_data_1 and pose_data_2 files)
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
          0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6]

# Contains weights by which keypoint distances are adjusted based on importance
weights_1 = [1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
           0.1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1,
           0.1, 0.1, 0.1, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5,
           0.5, 0.5, 0.1, 0.1]



def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    """
    Plot a confusion matrix for a 7-class classification problem.

    Parameters:
    - true_labels (list): List of true labels.
    - predicted_labels (list): List of predicted labels.
    - class_names (list): List of class names.

    Returns:
    - None (Displays the confusion matrix plot).
    """
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plotting the confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)

    plt.title('Confusion Matrix world coordinates')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def my_dist(a, b, weights):
    """
    Calculates the distance between two data-points fot the k-NN method

    Parameters:
    - a (list of thruples): data-point 1 
    - b (list of thruples): data-point 2 

    Returns:
    - distances measure (weighted sum of distances)
    """
    distances = []
    #print(len(weights))
    for i, point_pair in enumerate(zip(a, b)):
        raw_dist = np.linalg.norm(point_pair[0] - point_pair[1])
        distances.append(raw_dist * weights_1[i])
    return np.sum(distances)


def distances_to_set(x, point, weights):
    """
    Finds the distances from the given data-point (point) and returns a list
    of distances from that point to all the data-points in the provided list
    of data-points (x)

    Parameters:
    - x (list of list of keypoints): data-set
    - point (key-point thruple): point to compare to the data-set

    Returns:
    - list of distances
    """
    distances = []
    for keypoint in x:
        distances.append(my_dist(keypoint, point, weights))
    return distances


def mode(lst):
    """
    Finds the most common element of a list

    Parameters:
    lst (list): list to check the mode

    Returns:
    mode of the list
    """
    return int(max(set(lst), key=lst.count))


def knn_multiple(data, point_set, weights, k):
    """
    Executes the k-NN algorithm for every point in the given set of data-points
    and returns a list of predicted labels

    Parameters:
    data (list of list of key-points): Data to be used in k-NN algorithm
    point_set (list of list of key-points): Points for which labels should be predicted
    weights (list): not used
    k (int): k paramameter

    Returns:
    predictions (list): list of predicted labels for datapoints in point_set

    """
    x = data[0]
    y = data[1]
    predictions = []
    for point in point_set:
        distances = distances_to_set(x, point, weights)
        result = (np.array([distances, y]))
        result = pd.DataFrame(np.transpose(result), columns=['Distances', 'Labels'])
        result = result.sort_values('Distances')
        prediction = mode(result['Labels'].to_list()[0:k])
        predictions.append(prediction)
    return predictions


def load_data(num):
    """
    Unpickles the datafiles that contain the points to be used by the k-NN method
    """
    pickle_file = None
    if num == 1:
        pickle_file = open('data\pose_data_1', 'rb')
    elif num == 2:
        pickle_file = open('data\pose_data_2', 'rb')
    return pickle.load(pickle_file)


def main():
    """ Main entry point of the app """
    file_names = ['jab_data', 'cross_data', 'lead_hook_data', 'rear_hook_data',
                  'lead_uppercut_data', 'rear_uppercut_data', 'stance_data']

    db1 = load_data(1)
    db2 = load_data(2)



    world_coords_1 = db1['world_landmarks']
    coords_1 = db1['landmarks']

    world_coords_2 = db2['world_landmarks']
    coords_2 = db2['landmarks']

    coords = np.array(coords_1 + coords_2)
    world_coords = np.array(world_coords_1 + world_coords_2)


    x_train, x_test, y_train, y_test = train_test_split(world_coords, labels, test_size=0.3, random_state=12345)


    ### Experiment on different k parameter values
    f1_scores = []
    for k in range(1, 6):
        predictions = knn_multiple([x_train, y_train], x_test, weights_1, k)
        f1_scores.append(f1_score(y_test, predictions, average=None))

    print(f1_scores)
    plt.figure()
    for i in range(0, 5):
        plt.plot(label_names, f1_scores[i],  'o', alpha=0.5)
    plt.ylim([0, 1.1])
    plt.legend(['k=1', 'k=2', 'k=3', 'k=4', 'k=5'])
    plt.xlabel('Boxing Moves')
    plt.ylabel('F1 scores')
    plt.title('F1 scores for different values of k (world coordinates)')

    plt.show()



    ### Execute k-NN and plot results
    predictions = knn_multiple([x_train, y_train], x_test, weights_1, 3)
    print(predictions)
    print(y_test)
    mse = mean_squared_error(y_test, predictions)
    f1 = f1_score(y_test, predictions, average=None)
    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)

    print("MSE: " + str(mse))
    print('Accuracy: ' + str(acc))
    plt.plot(label_names, f1, 'ko')
    plt.plot(label_names, precision, 'ro')
    plt.plot(label_names, recall, 'bo')

    plt.legend(['F1 Score', 'Precision', 'Recall'])
    plt.xlabel('Moves')
    plt.ylabel('Statistic')
    plt.title('F1 score, precision & recall for each boxing move')
    plt.ylim([0, 1.1])

    plt.xlim()
    plt.show()

    plot_confusion_matrix(y_test, predictions, label_names)

    print('precision')
    print(precision)
    print('recall')
    print(recall)
    print('f1')
    print(f1)
    print(label_names)

if __name__ == "__main__":
    main()
