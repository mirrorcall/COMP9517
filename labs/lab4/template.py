## BEGIN Setup ##
# Step 1 - import relevant packages
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import numpy as np

print('COMP9517 Week 5 Lab - z5077914\n')

# Step 2 - load images using sklearn's load_digits()
digits = load_digits()
#digits = ( data, target )

# Step 2a - familiarize yourself with the dataset.
# There are 1797 8*8 images about digits.
#print(len(digits.target))

# Step 2b - Display some of the images and their labels.
# for i in range(10):
    # plt.imshow(np.reshape(digits.data[i], (8,8)), cmap='gray')
    # plt.title('Label: %i\n' % digits.target[i], fontsize=25)
    # plt.show()

# Step 3 - split the images using sklearn's train_test_split() with a test
# size anywhere between 20% and 30% (inclusive)
print('Test size =', 0.25)
X_train, X_test, Y_train, Y_test = train_test_split(
        digits.data, 
        digits.target, 
        test_size=0.25
)
## END Setup ##

## BEGIN Classification ##
# Step 4 - initialise the model (KNN, SGD, DT)
KNN = KNeighborsClassifier()    # n_neighbors=5
SGD = SGDClassifier()           # max_iter=1000, tol=1e-3
DT  = DecisionTreeClassifier()  # random_state=0

# Step 5 - fit the model to the training data
KNN.fit(X_train, Y_train)
SGD.fit(X_train, Y_train)
DT.fit(X_train, Y_train)

# Step 6 - Use the trained/fitted model to evaluate the test data
kscore = KNN.score(X_test, Y_test)
krecal = metrics.recall_score(Y_test, KNN.predict(X_test), average='macro')

sscore = SGD.score(X_test, Y_test)
srecal = metrics.recall_score(Y_test, SGD.predict(X_test), average='macro')

dscore = DT.score(X_test, Y_test)
drecal = metrics.recall_score(Y_test, DT.predict(X_test), average='macro')
## END Classification ##

## BEGIN Evaluation ##
# Step 7 - For each of the classifiers, evaluate the digit classification performance
# by calculating the accuracy, recall and generating the confusion matrix
print('KNN Acuracy:{:10f} Recall:{:10f}'.format(kscore, krecal))
print('SGD Acuracy:{:10f} Recall:{:10f}'.format(sscore, srecal))
print('DT Acuracy:{:10f} Recall:{:10f}'.format(dscore, drecal))
kcm = metrics.confusion_matrix(Y_test, KNN.predict(X_test))
print('KNN Confusion Matrix:\n{}'.format(kcm))
## END Evaluation ##
