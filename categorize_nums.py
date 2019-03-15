from sklearn import svm, neural_network, tree
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import random
import numpy as np
import matplotlib.pyplot as plt


def data_analysis(X, Y, model, model_name):

    """
    Generate plots of model results
    """

    score = model.score(X, Y)
    results = model.predict(X)

    cv = StratifiedKFold()

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, len(X))

    i = 0
    for train, test in cv.split(X, Y):
        probas_ = model.predict_proba(X)
        fpr, tpr, thresholds = roc_curve(Y, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1] , [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for {0}'.format(model_name))
    plt.legend(loc='lower right')
    plt.show()

def plot_features(x, y):

    """
    Plot features to be viewed
    """

    plt.scatter(x, y)
    plt.title('Training Features')
    plt.xlabel('Number')
    plt.ylabel('Classification')
    plt.show()


def SVM(X, Y):

    """
    Support Vector Machine model
    """

    clf = svm.SVC(gamma='scale', probability=True)
    clf.fit(X['train'], np.ravel(Y['train']))

    return clf


def MLP(X, Y):

    """
    Multi Layer Perceptron
    """

    clf = neural_network.MLPClassifier()
    clf.fit(X['train'], np.ravel(Y['train']))
    return clf

def DTC(X, Y):

    """
    Decision Tree Classifier
    """

    clf = tree.DecisionTreeClassifier()
    clf.fit(X['train'], np.ravel(Y['train']))

    return clf



def separate_data(data):

    """
    separates x and y
    """

    X = []
    Y = []
    for each_data in data:
        X.append(each_data[0])
        Y.append(each_data[1])

    return np.array(X), np.array(Y)


def encoder(data):

    """
    Encodes data into binary
    """

    new_data = []
    for each_data in data:
        binary = '{0:b}'.format(each_data[0])[-1]
        new_data.append(int(binary))

    return np.array(new_data).reshape(-1, 1)


def generate_training_data(max_num):

    """
    Generates training data
    """

    x_evens = np.array(list(range(0, max_num, 2))).reshape(-1, 1)
    y_evens = np.ones((len(x_evens), 1), dtype=int).reshape(-1,1)
    x_odds = np.array(list(map(lambda x: x+1, x_evens))).reshape(-1, 1)
    y_odds = np.ones((len(x_odds), 1), dtype=int)*0
    y_odds = y_odds.reshape(-1, 1)

    evens = list(zip(x_evens, y_evens))
    odds = list(zip(x_odds, y_odds))

    all_numbers = np.array(evens+odds)
    np.random.shuffle(all_numbers)

    X, Y = separate_data(all_numbers)

    X_encoded = encoder(X)

    return X_encoded, Y, X


def split_data(X, Y):

    """
    Splits data into train, test, and validation
    """

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

    x_data = {"train": x_train, "test": x_test, "val": x_val}
    y_data = {"train": y_train, "test": y_test, "val": y_val}

    return x_data, y_data


def main():

    X, Y, X_true = generate_training_data(100)
    plot_features(X_true, Y)
    X, Y = split_data(X, Y)
    x_real, y_real, x_true = generate_training_data(10000)


    svm_model = SVM(X, Y)
    data_analysis(x_real, y_real, svm_model, 'SVM')

    mlp_model = MLP(X, Y)
    data_analysis(x_real, y_real, mlp_model, 'MLP')

    dtc_model = DTC(X, Y)
    data_analysis(x_real, y_real, dtc_model, 'DTC')


if __name__ == '__main__':
    main()
