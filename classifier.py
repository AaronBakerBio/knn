import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as aggy


class KNNClassifier:
    def __init__(self, k, x_train, y_train):
        """
        Initialize the KNN classifier with the number of neighbors, training data, and training labels.
        :param k: int, the number of nearest neighbors to consider.
        :param x_train: list or numpy array of training data points.
        :param y_train: list or numpy array of training labels.
        """
        self.k = k
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    @staticmethod
    def euclidean_distance(x1, x2):
        """
        Calculate the Euclidean distance between two data points.
        :param x1: numpy array, the first data point.
        :param x2: numpy array, the second data point.
        :return: numpy array of distances between x1 and each point in x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def predict(self, x_test):
        """
        Predict the class labels for the provided test data using the KNN classifier.
        :param x_test: list or numpy array of test data points.
        :return: numpy array of predicted class labels for each test point.
        """
        x_test = np.array(x_test)
        predictions = []
        for test_point in x_test:
            distances = self.euclidean_distance(test_point, self.x_train)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            # Majority vote using numpy
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return np.array(predictions)


def parse_data(filename):
    """
    Parse data from a file into training data, labels, and header.
    :param filename: string, the file path to read from.
    :return: tuple of (data as numpy array, labels as numpy array, header as list).
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    header = lines[0].strip().split('\t')[:]
    data = []
    # Process all lines except the last one as data
    for line in lines[1:-1]:
        data.append(list(map(float, line.strip().split('\t')[1:])))
        # The last line contains class labels
    labels = lines[-1].strip().split('\t')[1:]
    labels = np.array(labels)
    data = np.array(data).T
    return data, labels, header


def label_to_int(labels):
    """
    Convert string labels to integer labels.
    :param labels: numpy array of string labels.
    :return: tuple of (integer labels as numpy array, mapping dictionary from string labels to integer labels).
    """
    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = np.array([label_map[label] for label in labels])
    return int_labels, label_map


def save_predictions(filename, patient_ids, predictions):
    """
    Save the predictions to a file.
    :param filename: string, the path to the file where predictions will be saved.
    :param patient_ids: list of patient IDs corresponding to the predictions.
    :param predictions: list of predicted labels.
    """
    with open(filename, 'w') as file:
        for patient_id, pred in zip(patient_ids, predictions):
            file.write(f'{patient_id}\t{pred}\n')


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.
    :param y_true: numpy array of true labels.
    :param y_pred: numpy array of predicted labels.
    :return: accuracy as a float.
    """
    correct = 0
    total = len(y_true)  # Total number of samples
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return float(correct) / float(total)


def cluster_classify(x_train, y_train_int):
    """
    Classify samples using Agglomerative Clustering.
    :param x_train: numpy array of training samples.
    :param y_train_int: numpy array of integer class labels corresponding to x_train.
    :return: numpy array of predicted integer labels after clustering.
    """
    cluster_model = aggy(n_clusters=2, linkage='average')
    y_pred_clusters = cluster_model.fit_predict(x_train)
    cluster0_class = np.argmax(np.bincount(y_train_int[y_pred_clusters == 0]))
    cluster1_class = np.argmax(np.bincount(y_train_int[y_pred_clusters == 1]))
    if cluster0_class == cluster1_class:
        if np.sum(y_pred_clusters == 0) > np.sum(y_pred_clusters == 1):
            cluster1_class = 1 - cluster0_class
        else:
            cluster0_class = 1 - cluster1_class
    y_pred = np.array([cluster0_class if cluster == 0 else cluster1_class for cluster in y_pred_clusters])
    return y_pred


def calc_TP_TN_FP_FN(y_true, y_pred):
    """
    Calculate true positives, true negatives, false positives, and false negatives.
    :param y_true: numpy array of true class labels.
    :param y_pred: numpy array of predicted class labels.
    :return: tuple of (TP, TN, FP, FN).
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN


def main():
    """
    Main method for executing full run of program.
    :return: No return
    """
    # task 1
    x_train, y_train, train_patients = parse_data('GSE994-train.txt')
    x_test, _, test_patients = parse_data('GSE994-test.txt')
    y_train_int, label_map = label_to_int(y_train)
    knn_1 = KNNClassifier(k=1, x_train=x_train, y_train=y_train_int)
    knn_3 = KNNClassifier(k=3, x_train=x_train, y_train=y_train_int)
    y_pred_1_int = knn_1.predict(x_test)
    y_pred_3_int = knn_3.predict(x_test)
    inverse_label_map = {v: k for k, v in label_map.items()}
    y_pred_1 = [inverse_label_map[label] for label in y_pred_1_int]
    y_pred_3 = [inverse_label_map[label] for label in y_pred_3_int]
    save_predictions('Prob5-1NNoutput.txt', test_patients, y_pred_1)
    save_predictions('Prob5-3NNoutput.txt', test_patients, y_pred_3)
    # 5 fold task
    k_vals = [1, 3, 5, 7, 11, 21, 23]
    acc_by_k = {}
    fold_ranges = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30)]
    for k in k_vals:
        fold_accuracies = []
        for fold_idx, (start_idx, end_idx) in enumerate(fold_ranges):
            x_train_fold = np.concatenate([x_train[:start_idx], x_train[end_idx:]])
            y_train_fold = np.concatenate([y_train_int[:start_idx], y_train_int[end_idx:]])
            x_val_fold = x_train[start_idx:end_idx]
            y_val_fold = y_train_int[start_idx:end_idx]
            knn = KNNClassifier(k, x_train_fold, y_train_fold)
            y_pred_fold = knn.predict(x_val_fold)
            acc_fold = accuracy(y_val_fold, y_pred_fold)
            fold_accuracies.append(acc_fold)
        acc_by_k[k] = sum(fold_accuracies) / len(fold_accuracies)
    for k, acc in acc_by_k.items():
        print(f"Average accuracy for k={k}: {acc:.2f}")
    # Plot the graph of accuracies
    plt.plot(k_vals, [acc_by_k[k] for k in k_vals], marker='o')
    plt.title('Average Accuracy vs. Number of Neighbors (k)')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Average Accuracy')
    plt.grid(True)
    plt.savefig('knn_accuracies.png')
    plt.show()
    # task 2
    y_pred_clusters = cluster_classify(x_train, y_train_int)
    cluster_acc = accuracy(y_train_int, y_pred_clusters)
    TP, TN, FP, FN = calc_TP_TN_FP_FN(y_train_int, y_pred_clusters)
    print(f'Agglomerative Clustering - TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, Accuracy: {cluster_acc:.2f}')
    # this step grabs the best k.
    best_k = max(acc_by_k, key=acc_by_k.get)
    best_knn = KNNClassifier(best_k, x_train, y_train_int)
    y_pred_knn = best_knn.predict(x_train)
    knn_acc = accuracy(y_train_int, y_pred_knn)
    TP_knn, TN_knn, FP_knn, FN_knn = calc_TP_TN_FP_FN(y_train_int, y_pred_knn)
    print(f'Best KNN (k={best_k}) - TP: {TP_knn}, TN: {TN_knn}, FP: {FP_knn}, FN: {FN_knn}, Accuracy: {knn_acc:.2f}')


if __name__ == "__main__":
    main()
