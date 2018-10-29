"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics

from matplotlib.font_manager import FontProperties
######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        num_survived = float(np.sum(y == 1))
        total_passengers = len(y)
        self.probabilities_ = { 
            'survived' : num_survived / total_passengers,
            'not_survived' : (total_passengers - num_survived) / total_passengers
        }  
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        y = np.random.choice([0,1], size=len(X), p=[self.probabilities_['not_survived'],self.probabilities_['survived']])        
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2, train_size=None) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0    
    for i in range(0, ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=i)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        train_error += 1- metrics.accuracy_score(y_train, y_pred, normalize=True)
        y_pred = clf.predict(X_test)
        test_error += 1- metrics.accuracy_score(y_test, y_pred, normalize=True)

    train_error /= ntrials
    test_error /= ntrials   
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier()       # create Random classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1- metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1- metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- 3-NN training error: %.3f' % train_error)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1- metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- 5-NN training error: %.3f' % train_error)
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = 1- metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- 7-NN training error: %.3f' % train_error)

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    clf = MajorityVoteClassifier()
    train_error, test_error = error(clf, X, y)
    print('\t-- MajorityVote training error: %.3f \t testing error: %.3f' % (train_error, test_error))
    clf = RandomClassifier()
    train_error, test_error = error(clf, X, y)
    print('\t-- Random training error: %.3f \t testing error: %.3f' % (train_error, test_error))
    clf = DecisionTreeClassifier(criterion='entropy')
    train_error, test_error = error(clf, X, y)
    print('\t-- DecisionTree training error: %.3f \t testing error: %.3f' % (train_error, test_error))
    clf = KNeighborsClassifier(n_neighbors=5)
    train_error, test_error = error(clf, X, y)
    print('\t-- 5-NN training error: %.3f \t testing error: %.3f' % (train_error, test_error))
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    xPlot=[]
    yPlot=[]
    for i in range(1,50,2):
        clf = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
        print('\t-- %d-NN 10-fold cross validation error: %.3f' % (i, 1-scores.mean()))
        xPlot.append(i)
        yPlot.append(1-scores.mean())
    plt.plot(xPlot, yPlot, '-')
    plt.axis('auto')
    plt.xlabel('K')
    plt.ylabel('10-Fold Cross Validation Average Error')
    plt.title('K-NN Classifier 10-Fold Cross Validation\nFor Titanic Data')
    plt.savefig("Problem4.2-f.pdf")
    plt.clf()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    xPlot=[]
    trainErrorPlot=[]
    testErrorPlot=[]
    for i in range(1,21):
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        trainError, testError= error(clf, X, y)
        print('\t-- Max-Depth %d Decision Tree 80/20 cross validation training error: %.3f \t testing error: %.3f' % (i, trainError, testError))
        xPlot.append(i)
        trainErrorPlot.append(trainError)
        testErrorPlot.append(testError)
    line1, =plt.plot(xPlot, trainErrorPlot, '-', label='Training Error')
    line2, =plt.plot(xPlot, testErrorPlot, '-', label='Test Error')
    plt.axis('auto')
    plt.xlabel('Max Depth')
    plt.ylabel('Average Error')
    plt.legend(loc='lower left')
    plt.title('Max-Depth Decision Tree Classifier 80/20 Cross Validation\nFor Titanic Data')
    plt.savefig("Problem4.2-g.pdf")
    plt.clf()
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    xPlot=[]
    decisionTreeTrainErrorPlot=[]
    decisionTreeTestErrorPlot=[]
    kNeighborsTrainErrorPlot=[]
    kNeighborsTestErrorPlot=[]
    clfDecisionTree = DecisionTreeClassifier(criterion='entropy', max_depth=6)
    clfKNeighbors = KNeighborsClassifier(n_neighbors=7)
    for trainingPercentage in range(1,11):
        trainError, testError= error(clfDecisionTree, X, y, test_size=0.1, train_size=trainingPercentage*0.09)
        print('\t-- Max-Depth 6 Decision Tree %d%% learning training error: %.3f \t testing error: %.3f' % (trainingPercentage*10, trainError, testError))
        xPlot.append(trainingPercentage*10)
        decisionTreeTrainErrorPlot.append(trainError)
        decisionTreeTestErrorPlot.append(testError)
    for trainingPercentage in range(1,11):
        trainError, testError= error(clfKNeighbors, X, y, test_size=0.1, train_size=trainingPercentage*0.09)
        print('\t-- 7-NN %d%% learning training error: %.3f \t testing error: %.3f' % (trainingPercentage*10, trainError, testError))
        kNeighborsTrainErrorPlot.append(trainError)
        kNeighborsTestErrorPlot.append(testError)
    line1, =plt.plot(xPlot, decisionTreeTrainErrorPlot, '-', label='Max Depth 6 Decision Tree Training Error')
    line2, =plt.plot(xPlot, decisionTreeTestErrorPlot, '-', label='Max Depth 6 Decision Tree Test Error')
    line3, =plt.plot(xPlot, kNeighborsTrainErrorPlot, '-', label='7-NN Training Error')
    line4, =plt.plot(xPlot, kNeighborsTestErrorPlot, '-', label='7-NN Test Error')
    plt.axis('auto')
    plt.xlabel('Training Percentage')
    plt.ylabel('Average Error')
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(loc='best', prop=fontP)
    plt.title('Learning Rate of 7-NN and Max Depth 6 Decision Tree\nFor Titanic Data')
    plt.savefig("Problem4.2-h.pdf")
    plt.clf()

    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
