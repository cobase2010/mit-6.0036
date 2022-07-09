from enum import auto
import pdb
import numpy as np
import math
import numpy.linalg as linalg
import code_for_hw3_part2 as hw3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are hw3.standard and hw3.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]

features2 = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

# Construct the standard data and label arrays
auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

if False:                               # set to True to see histograms
    import matplotlib.pyplot as plt
    for feat in range(auto_data.shape[0]):
        print('Feature', feat, features[feat][0])
        # Plot histograms in one window, different colors
        plt.hist(auto_data[feat,auto_labels[0,:] > 0])
        plt.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
        # Plot histograms in two windows, different colors
        fig,(a1,a2) = plt.subplots(nrows=2)
        a1.hist(auto_data[feat,auto_labels[0,:] > 0])
        a2.hist(auto_data[feat,auto_labels[0,:] < 0])
        plt.show()
def get_learner(learner_name, t):
    def perceptron2(*args):
        return hw3.perceptron(*args, {"T":t})
    def average_perceptron2(*args):
        return hw3.averaged_perceptron(*args, {"T":t})
    if learner_name == "perceptron":
        return perceptron2
    if learner_name == "averaged_perceptron":
        return average_perceptron2

def evaluate_data(learner, feature_set):
    auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, feature_set)
    print(hw3.xval_learning_alg(learner, auto_data, auto_labels, 10))

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Your code here to process the auto data
# # 4.1C 1)
# evaluate_data(get_learner("perceptron", 1), features)
# evaluate_data(get_learner("averaged_perceptron", 1), features)

# # 4.1C 2)
# evaluate_data(get_learner("perceptron", 1), features2)
# evaluate_data(get_learner("averaged_perceptron", 1), features2)

# # 4.1C 3)
# evaluate_data(get_learner("perceptron", 10), features)
# evaluate_data(get_learner("averaged_perceptron", 10), features)

# # 4.1C 4)
# evaluate_data(get_learner("perceptron", 10), features2)
# evaluate_data(get_learner("averaged_perceptron", 10), features2)
# auto_data, auto_labels = hw3.auto_data_and_labels(auto_data_all, features2)
# print("thetas: ", hw3.averaged_perceptron(auto_data, auto_labels, {"T":10}))

# # 4.1C 5)
# evaluate_data(get_learner("perceptron", 50), features)
# evaluate_data(get_learner("averaged_perceptron", 50), features)

# # 4.1C 6)
# evaluate_data(get_learner("perceptron", 50), features2)
# evaluate_data(get_learner("averaged_perceptron", 50), features2)
#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data
# learners = ["perceptron", "averaged_perceptron"]
# iterations = [1, 10, 50]
# for iter in iterations:
#     for learner in learners:
#         print("Iteration", iter, "Evaluating", learner)
#         print(hw3.xval_learning_alg(get_learner(learner, iter), review_bow_data, review_labels, 10))

# th, th0 =  hw3.averaged_perceptron(review_bow_data, review_labels, {"T":10})
# print(th.shape)
# reverse_d = hw3.reverse_dict(dictionary)
# ind = np.argpartition(th, -10, axis=0)[-10:,0]
# print(ind)
# # ind2 = ind[np.argsort(th[ind:0], reversed=True)]

# res = "["
# for i in ind:
#     res += "\"" + reverse_d[i] + "\","
# res += "]"
# print(res)

# # Negative words
# ind = np.argpartition(th, 10, axis=0)[0:10,0]
# print(ind.shape)
# # ind2 = ind[np.argsort(th[ind:0], reversed=True)]
# res = "["
# for i in ind:
#     # print(i)
#     res += "\"" + reverse_d[i] + "\","
# res += "]"
# print(res)

# 5.2C optional: find the most postive and negative reviews

# d, n = review_bow_data.shape
# pos_margin = 0
# neg_margin = 1000
# most_positive_ind = -1
# most_negative_ind = -1
# for i in range(n):
#     y_i = review_labels[0, i]
#     x_i = review_bow_data[:,i]
#     # print("x",i, linalg.norm(x_i))
#     if y_i > 0:
#         gamma = (th.T @ x_i + th0) / linalg.norm(th)
#         if gamma > pos_margin:
#             pos_margin = gamma
#             most_positive_ind = i
#     if y_i <= 0:
#         gamma = (th.T @ x_i + th0) / linalg.norm(th)
#         if gamma < neg_margin:
#             neg_margin = gamma
#             most_negative_ind = i

# print("pos", pos_margin, most_positive_ind)
# print("Most Positive:", review_texts[most_positive_ind])
# print("neg", neg_margin, most_negative_ind)
# print("Most Negative:", review_texts[most_negative_ind])
#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

"""
Returns a dictionary formatted as follows:
{
    0: {
        "images": [(m by n image), (m by n image), ...],
        "labels": [0, 0, ..., 0]
    },
    1: {...},
    ...
    9
}
Where labels range from 0 to 9 and (m, n) images are represented
by arrays of floats from 0 to 1
"""
mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
d2 = mnist_data_all[2]["images"]
d3 = mnist_data_all[3]["images"]
d4 = mnist_data_all[4]["images"]
d5 = mnist_data_all[5]["images"]
d6 = mnist_data_all[6]["images"]
d7 = mnist_data_all[7]["images"]
d8 = mnist_data_all[8]["images"]
d9 = mnist_data_all[9]["images"]

y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)
y2 = np.repeat(-1, len(d2)).reshape(1,-1)
y3 = np.repeat(1, len(d3)).reshape(1,-1)
y4 = np.repeat(1, len(d4)).reshape(1,-1)
y5 = np.repeat(1, len(d5)).reshape(1,-1)
y6 = np.repeat(-1, len(d6)).reshape(1,-1)
y7 = np.repeat(1, len(d7)).reshape(1,-1)
y8 = np.repeat(1, len(d8)).reshape(1,-1)
y9 = np.repeat(1, len(d9)).reshape(1,-1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
data24 = np.vstack((d2, d4))
data68 = np.vstack((d6, d8))
data90 = np.vstack((d9, d0))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T
labels24 = np.vstack((y2.T, y4.T)).T
labels68 = np.vstack((y6.T, y8.T)).T
labels90 = np.vstack((y9.T, y0.T)).T

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    n, m1, m2 = x.shape
    y = x.reshape(n, m1*m2)
    y = np.transpose(y)
    return y

    # data_points, m, n = x.shape
    # return x.reshape((data_points, m*n)).T


def row_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    # print("row_average_features", x.shape)
    n, m1, m2 = x.shape

    y = np.average(x, axis = 2).reshape(n, m1)
    return y.T

    # n_samples, m, n = x.shape
    # return np.mean(x, axis=2,keepdims=True).T.reshape((m,n_samples))

def col_average_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (n,n_samples) array where each entry is the average of a column
    """
    n, m1, m2 = x.shape

    y = np.average(x, axis = 1).reshape(n, m2)
    # print(y.shape)
    return y.T

    # n_samples, m, n = x.shape
    # return np.mean(x, axis=1,keepdims=True).T.reshape((n,n_samples))


def top_bottom_features(x):
    """
    This should either use or modify your code from the tutor questions.

    @param x (n_samples,m,n) array with values in (0,1)
    @return (2,n_samples) array where the first entry of each column is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    k, m, n = x.shape
    x1 = x[:,0:math.floor(m/2), :]
    x2 = x[:,math.floor(m/2):, :]
    y1 = np.average(x1, axis=(1, 2)).reshape(k, 1)
    y2 = np.average(x2, axis=(1, 2)).reshape(k, 1)
    y = np.concatenate((y1, y2), axis=1)
    y = np.transpose(y).reshape(2, k)
    # print(y.shape)
    return y

    # n_samples, m, n = x.shape
    # return np.array([np.mean(x[:,:m // 2,],axis=(1,2)), np.mean(x[:,m // 2:,],axis=(1,2))]).reshape((2,n_samples))

# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(raw_mnist_features(data), labels)
print("accuracy 0v1", acc)
acc = hw3.get_classification_accuracy(raw_mnist_features(data24), labels24)
print("accuracy 2v4", acc)
acc = hw3.get_classification_accuracy(raw_mnist_features(data68), labels68)
print("accuracy 6v8", acc)
acc = hw3.get_classification_accuracy(raw_mnist_features(data90), labels90)
print("accuracy 9v0", acc)
# raw_mnist_features(data)
# row_average_features(data)
# top_bottom_features(data)
acc = hw3.get_classification_accuracy(row_average_features(data), labels)
print("row average accuracy 0v1", acc)
acc = hw3.get_classification_accuracy(col_average_features(data), labels)
print("col average accuracy 0v1", acc)
acc = hw3.get_classification_accuracy(top_bottom_features(data), labels)
print("top bottom accuracy 0v1", acc)

acc = hw3.get_classification_accuracy(row_average_features(data24), labels)
print("row average accuracy 2v4", acc)
acc = hw3.get_classification_accuracy(col_average_features(data24), labels)
print("col average accuracy 2v4", acc)
acc = hw3.get_classification_accuracy(top_bottom_features(data24), labels)
print("top bottom accuracy 2v4", acc)

acc = hw3.get_classification_accuracy(row_average_features(data68), labels)
print("row average accuracy 6v8", acc)
acc = hw3.get_classification_accuracy(col_average_features(data68), labels)
print("col average accuracy 6v8", acc)
acc = hw3.get_classification_accuracy(top_bottom_features(data68), labels)
print("top bottom accuracy 6v8", acc)

acc = hw3.get_classification_accuracy(row_average_features(data90), labels)
print("row average accuracy 9v0", acc)
acc = hw3.get_classification_accuracy(col_average_features(data90), labels)
print("col average accuracy 9v0", acc)
acc = hw3.get_classification_accuracy(top_bottom_features(data90), labels)
print("top bottom accuracy 9v0", acc)
#-------------------------------------------------------------------------------
# Analyze MNIST data
#-------------------------------------------------------------------------------

# Your code here to process the MNIST data

