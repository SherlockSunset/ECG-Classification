from sklearn.model_selection import train_test_split     # data splitting
from sklearn.svm import SVC                              # SVM
from sklearn.linear_model import LogisticRegression      # logistic regression

from Solution import *

names = ["LinearSVM", "RBFSVM", "LogisticRegression"]
classifiers = [
    SVC(kernel="linear", C=0.025),                                  # linear SVM high C, may overfitting, low C, may underfitting
    SVC(C=5, kernel='rbf',gamma=2, decision_function_shape='ovo'),  # RBF SVM
    LogisticRegression(C=100.0, random_state=0),]                   # Logistic Regression: smaller C, stonger normalization


def loadDataset(filename):  # load .txt file data
    xdataMat = []
    yMat = []
    fr = open(filename)
    for line in fr.readlines():
        line = line.strip().split(" ")
        xdataMat.append([int(line[0]),float(line[1])])
        yMat.append(int(line[2]))
    return xdataMat, yMat
data_list_dir = obtain_data_list(path = './database/train/')
x = []
y = []
for data_dir in data_list_dir:
    x_single, y_single = loadDataset('test_'+data_dir + '.txt')
    x = x + x_single
    y = y + y_single

train_data, test_data, train_label, test_label =train_test_split(x, y, random_state=1, train_size=0.6, test_size=0.4)  #sklearn.model_selection.


# iterate through classifiers
for name, clf in zip(names, classifiers):
    # ax = mp.subplot(len(datasets), len(classifiers) + 1, i)
    clf.fit(train_data, train_label)  # train
    score_train = clf.score(train_data, train_label)  # train accuracy
    score_test = clf.score(test_data, test_label)  # test accuracy
    print("Classifier: ", name)
    print("Train dataset: ", score_train)
    print("Test dataset: ", score_test)

    # predict on final testing data
    test_case_list = obtain_data_list(path='./database/test/')
    for ele1 in test_case_list:
        testing_file_path = './database/test/' + ele1
        ecg_sig_test, ecg_type_test, ecg_peak_test, signals = read_ecg(testing_file_path)

        index_gap_test = calculate_gap(ecg_peak_test)
        amplitude_test = calculate_amplitude(ecg_sig_test, ecg_peak_test)
        # label_list_test = assign_label(ecg_type_test)
        combined_data = combine(index_gap_test, amplitude_test)

        test_prediction = clf.predict(combined_data)
        result = []
        for ele in test_prediction:
            if ele == 1:
                result.append('V')
            else:
                result.append('?')
        filename = name +'_'+ ele1
        wfdb.wrann(ele1, 'test', ecg_peak_test, result, write_dir='./database/test_result/')  # write the prediction results to WFDB format
        textdatadir = './database/test_result/'
        np.savetxt(textdatadir+filename + '.txt', np.array(result), fmt='%s')                 # save the prediction results to .txt
