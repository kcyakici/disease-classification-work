import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def custom_flatten(matrix):
    flat_list = list()
    for row in matrix:
        flat_list.extend(row)
    return flat_list


def custom_train_test_split(num_classes, samples_per_class, test_samples_per_class):

    number_of_iterations = samples_per_class // test_samples_per_class
    total_samples = num_classes * samples_per_class

    # Create an array of indices corresponding to the samples
    indices = np.arange(total_samples)

    # Reshape the indices array to match the structure of dataset
    indices = indices.reshape((num_classes, samples_per_class))

    # Initialize empty arrays to store indices for training and test sets
    train_indices = list()
    test_indices = list()

    # Iterate over the desired number of iterations
    for iteration in range(number_of_iterations):

        # Calculate the starting index for the test set
        test_start = iteration * test_samples_per_class
        temp_set_test_indices = list()
        temp_set_train_indices = list()

        # Iterate over each class
        for class_idx in range(num_classes):

            # Calculate the indices for the test set
            test_class_indices = indices[class_idx,
                                         test_start:test_start + test_samples_per_class]

            # Calculate the indices for the training set
            train_class_indices = np.setdiff1d(
                indices[class_idx], test_class_indices)

            # Append the class indices to the overall training and test sets
            temp_set_test_indices.append(test_class_indices)
            temp_set_train_indices.append(train_class_indices)

        test_indices.append(custom_flatten(temp_set_test_indices))
        train_indices.append(custom_flatten(temp_set_train_indices))

        # test_indices = np.concatenate((test_indices, test_class_indices))
        # train_indices = np.concatenate(
        #     (train_indices, train_class_indices))

    return train_indices, test_indices


def custom_evaluate(classifier, X, y, train_indices, test_indices):
    num_splits = len(train_indices)
    max_accuracy_score = -1
    split_with_max_accuracy_score = None

    # Iterate over the splits
    for iteration in range(num_splits):
        print(f"{iteration + 1}. split")
        train_set = train_indices[iteration]
        test_set = test_indices[iteration]

        X_train, X_test = X[train_set], X[test_set]
        y_train, y_test = y[train_set], y[test_set]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > max_accuracy_score:
            max_accuracy_score = accuracy
            split_with_max_accuracy_score = iteration + 1

        report = classification_report(y_test, y_pred)
        y_check = classifier.predict(X_train)
        accuracy_check = accuracy_score(y_train, y_check)
        print('Confusion matrix: ' + "\n" +
              str(confusion_matrix(y_test, y_pred)))
        print('Accuracy Score: '+str(accuracy))
        print('Accuracy Score on Train Data: '+str(accuracy_check))
        print('Classification Report:\n'+report)
        print("----")

    print(
        f"Split with the highest accuracy: {split_with_max_accuracy_score} " + "\n" +
        f"Accuracy achieved with this split: {max_accuracy_score}")


def process_features(feature_array):
    processed_features = []
    for j in range(len(feature_array)):
        features = feature_array[j].values

        for sentence in range(0, len(features)):
            # Remove all the special characters
            processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

            # remove all single characters
            processed_feature = re.sub(
                r'\s+[a-zA-Z]\s+', ' ', processed_feature)

            # Remove single characters from the start
            processed_feature = re.sub(
                r'\^[a-zA-Z]\s+', ' ', processed_feature)

            # Substituting multiple spaces with single space
            processed_feature = re.sub(
                r'\s+', ' ', processed_feature, flags=re.I)

            # Removing prefixed 'b'
            processed_feature = re.sub(r'^b\s+', '', processed_feature)

            # Converting to Lowercase
            processed_feature = processed_feature.lower()

            processed_features.append(processed_feature)

    return processed_features


hypothyroidism = pd.read_csv('hypothyroidism.csv')
hashimoto = pd.read_csv('hashimoto.csv')
rheumatism = pd.read_csv('rheumatism.csv')

hypothyroidism['target'] = 0
hashimoto['target'] = 1
rheumatism['target'] = 2

# to concatenate dataframes
data = pd.concat([hypothyroidism, hashimoto, rheumatism])
# print(f"Data: \n{data}")

feature_array = [data[['Title', 'Abstract', 'Keywords']]]
# print(f"Feature array: \n{feature_array}")

processed_features = process_features(feature_array)
# print(f"Features after processed:\n{processed_features}")

# Feature extraction using TF-IDF
nltk.download('stopwords')
vectorizer = TfidfVectorizer(
    max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(processed_features).toarray()
y = data['target'].to_numpy()

# print(type(X))
# print(type(y))
train_indices, test_indices = custom_train_test_split(3, 500, 100)

# print(len(train_indices))
# print(len(test_indices))

# Initializing classifiers
rf = RandomForestClassifier(random_state=42)
linear_svm = LinearSVC()
knn = KNeighborsClassifier()
lr = LogisticRegression()

classifiers = [rf, linear_svm, knn, lr]

custom_evaluate(rf, X, y, train_indices, test_indices)
