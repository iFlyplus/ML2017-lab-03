#
# Authors: Ziwei Zhang
# Create Date: Dec. 15, 2017
# Final Update Date: Dec. 17, 2017
# 
# Machine Learning 2017 Experiments No.3
#    Face Classification Based on AdaBoost Algorithm
#    Part 2: AdaBoostClassifier implementation
#
import pickle
import copy
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifiers = []
        self.weak_classifier_weights = np.zeros(self.n_weakers_limit, dtype=np.float32)
        self.weak_classifier_errors = np.ones(self.n_weakers_limit, dtype=np.float32)

    def __str__(self):
        _clf = str(type(self.weak_classifier))[8:-2]
        _n = self.n_weakers_limit
        _str = 'AdaBoostClassifier(weak_classifier=%s, n_weakers_limit=%d)' % (_clf, _n)
        return _str

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y,verbose=True):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        
        num_train = X.shape[0]

        # initialize sample weight to 1.0/num_sample
        sample_weight = np.ones((num_train,)) / num_train

        # Clear any previous fit results
        self.weak_classifiers = []
        self.weak_classifier_weights = np.zeros(self.n_weakers_limit, dtype=np.float32)
        self. weak_classifier_errors = np.ones(self.n_weakers_limit, dtype=np.float32)

        for iboost in range(self.n_weakers_limit):
            # boost step
            sample_weight, classifier_weight, classifier_error = \
                                            self._boost(iboost, X, y, sample_weight)
            
            # early terminate
            if sample_weight is None:
                if verbose:
                    print('the error of current classifier as bad as random guessing(or worse), stop boost')
                break
            
            self.weak_classifier_weights[iboost] = classifier_weight
            self.weak_classifier_errors[iboost] = classifier_error
            
            # stop is error is 0
            if classifier_weight == 0:
                if verbose:
                    print('the error of current classifier is 0, stop boost')
                break

            if verbose:
                print('boost step %d / %d: classifier weight: %.3f, classification error: %.3f'
                        % (iboost+1, self.n_weakers_limit, classifier_weight, classifier_error))
            
    def _boost(self, iboost, X, y, sample_weight):
        '''
        Perform a single boost.
        Notice: the boost implement here is only suitable for 
                BINARY classification problem.

        Inputs:
        - iboost: (int) The index of the current boost iteration
        - X: A numpy array of shape (N, D), 
             containing training data;
             there are N training samples each of dimension D.
        - y: A numpy array of shape (N,) 
             containing training labels
        - sample_weight: A numpy array of shape (N,),
                         containing the current sample weights. 

        Returns:
        - sample_weight: A numpy array of shape (N,) 
                         containing the reweighted sample weights.
                         None indicates the boosting has terminated early.
        - classifier_weight: (float) The weight of the current boost.
                              None indicates the boosting has terminated early.
        - classifier_error: (float) The classification error of the current boost.
                            None indicates the boosting has terminated early.
        
        '''
        # create and train a new weak classifier
        clf = self._create_classifier()
        clf.fit(X, y, sample_weight=sample_weight)
        
        y_pred = clf.predict(X)
        
        incorrect = (y_pred != y)

        classifier_error = np.average(incorrect, weights=sample_weight)

        # stop if classifier is perfect
        if classifier_error == 0:
            return sample_weight, 1.0, 0.0
        
        # stop if classifier perform badly
        if classifier_error >= 0.5:
            self.classifiers.pop(-1)
            if len(self.classifiers) == 0:
                raise ValueError('weak_classifier provided can not used to create an emsemble classifier')
            return None, None, None

        classifier_weight = 0.5 * np.log((1.0 - classifier_error) / classifier_error)

        zm = np.sum(sample_weight * np.exp(-classifier_weight * y * y_pred))
        sample_weight *= np.exp(-classifier_weight * y * y_pred)
        sample_weight /= zm

        return sample_weight, classifier_weight, classifier_error
    
    def _create_classifier(self):
        '''create a copy of self.weak_classifier'''
        clf = copy.deepcopy(self.weak_classifier)
        self.weak_classifiers.append(clf)
        return clf

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        scores = sum((clf.predict(X) * w for clf, w in zip(self.weak_classifiers , self.weak_classifier_weights)))
        return scores

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
            shape (N,)
        '''
        scores = sum((clf.predict(X) * w for clf, w in zip(self.weak_classifiers, self.weak_classifier_weights)))
        # y_pred /= X.shape[0]
        y_pred = np.ones((X.shape[0],))
        y_pred[scores < threshold] = -1
        return y_pred
    
    def staged_predict(self, X, threshold=0):
        '''
        Yeild the ensemble prediction after after each boost.

        Inputs:
        - X: A numpy array of shape (N, D), 
             containing training data;
             there are N training samples each of dimension D.
        - threshold: The demarcation number of deviding the samples into two parts.

        Returns:
        - y: A generator of array of shape(N,), 
             containing prediction of each sample in X.
        '''
        scores = None
        for clf, w in zip(self.weak_classifiers, self.weak_classifier_weights):
            current_score = clf.predict(X) * w # (N,)

            if scores is None:
                scores = current_score
            else:
                scores += current_score

            current_pred = np.ones((X.shape[0],))
            current_pred[scores < threshold] = -1
            yield current_pred


    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
