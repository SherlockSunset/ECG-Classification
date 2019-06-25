1. The final prediction results are show in three folders under database/test: linear SVM, RBF SVM, Logistic Regression (representing predicted by corresponding method), 
    each folder contains b1.test and b2.test, the '.txt' file of the predicted results are also provided.

2. To solve the PVC classification problem, I selected two features: the gap between two successive impulses as feature 1, the amplitude of the impulse as feature 2.

3. I tested three classifiers in sklearn:  linear SVM, RBF SVM, Logistic Regression

Final performance:

Linear SVM:
Training accuracy:  0.968
Testing accuracy:   0.965

RBF SVM:
Training accuracy:  0.984
Testing accuracy:   0.972

Logistic Regression:
Training accuracy:  0.967
Testing accuracy:   0.965

For the codes, one should first excute the Preprocessing.py file to extract the features from provided ECG data, 
then by excuting the Classifier.py, one can achieve prediction.


