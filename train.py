#
# Authors: Ziwei Zhang
# Create Date: Dec. 15, 2017
# Final Update Date: Dec. 19, 2017
# 
# Machine Learning 2017 Experiments No.3
#    Face Classification Based on AdaBoost Algorithm
#    Part 1: train script
#
import os
import numpy as np
from scipy.misc import imread, imsave, imresize
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from feature import NPDFeature
from sklearn.tree import DecisionTreeClassifier
from ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

def get_data(num_train=600):
    data_folder = './datasets/original/'
    if os.path.exists(os.path.join(data_folder, 'dataset.pickle')):
        print('Use pickled dataset.')
        dataset = load(data_folder + 'dataset.pickle')
    else:
        print('Load raw image.')
        dataset = {}
        faceset = load_data(os.path.join(data_folder, 'face'))
        nonfaceset = load_data(os.path.join(data_folder, 'nonface'))
        dataset['face'] = preprocess_data(faceset)
        dataset['nonface'] = preprocess_data(nonfaceset)
        # pickle dataset for later use
        save(dataset, data_folder + 'dataset.pickle')
    
    # num_train + num_val = total sample number
    # num_train_face + num_train_nonface = num_train
    # num_train_face + num_val_face = total face sample number
    # num_val_nonface = num_val - num_val_face
    #                 = total - num_train - (total_face - num_train_face)
    #                 = total_non_face - num_train_nonface
    # ==> num_train_nonface + num_val_nonface = total non face sample number
    num_feature = dataset['face'].shape[1]
    num_val = dataset['face'].shape[0] + dataset['nonface'].shape[0] - num_train
    num_train_face = num_train // 2
    num_train_nonface = num_train - num_train_face
    num_val_face = dataset['face'].shape[0] - num_train_face
    num_val_nonface = num_val - num_train_face

    X_train = np.ndarray(shape=(num_train, num_feature), dtype=np.float32)
    y_train = np.ones((num_train,))
    X_test = np.ndarray(shape=(num_val, num_feature), dtype=np.float32)
    y_test = np.ones((num_val,))

    X_train[:num_train_face] = dataset['face'][:num_train_face]
    X_train[num_train_face:] = dataset['nonface'][:num_train_nonface]
    y_train[num_train_face:] = -1
    X_test[:num_val_face] = dataset['face'][num_train_face:]
    X_test[num_val_face:] = dataset['nonface'][num_train_nonface:]
    y_test[num_val_face:] = -1

    return {
      'X_train': X_train, 'y_train': y_train,
      'X_test': X_test, 'y_test': y_test
    }

def load_data(data_folder):
    '''Load image data from data_folder'''
    image_files = os.listdir(data_folder)
    imageset = None
    image_index = 0
    for image in image_files:
        image_file = os.path.join(data_folder, image)
        try:
            img = imread(image_file).astype(float)
            '''
            if img.shape != (image_size, image_size, 3):
                raise Exception('Unexpected image shape: %s' % str(img.shape))
            '''
            if imageset is None:
                image_size = img.shape[0]
                imageset = np.ndarray(shape=(len(image_files), image_size,image_size, 3), 
                                    dtype=np.float32)
            imageset[image_index] = img
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e)
    num_image = image_index
    imageset = imageset[:num_image, :, :, :]

    return imageset

def preprocess_data(dataset):
    '''convert image to greyscale,resize to (24,24) and extract NPD feature'''
    # feature size: 165600
    num_feature = 165600
    num_sample = dataset.shape[0]
    dataset_processed = np.ndarray(shape=(num_sample, num_feature), 
                                    dtype=np.float32)
    for i in range(num_sample):
        img_grey = convert_to_grey(dataset[i])
        img_grey_resize = imresize(img_grey, (24,24))
        npdfeature = NPDFeature(img_grey_resize).extract()
        dataset_processed[i] = npdfeature
    return dataset_processed

def convert_to_grey(img):
    return img[:,:,0] * 0.3 + img[:,:,1] * 0.59 + img[:,:,2] * 0.11


def save(file, filename):
    with open(filename, 'wb') as f:
        pickle.dump(file, f)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def compute_accuracy(y_pred, y_groundture):
    return np.mean(np.array(y_pred == y_groundture), dtype = np.float32)

def report(y_groundture, y_pred, filename=None):
    pass

if __name__ == "__main__":
    #
    # Notice: dataset dir: './datasets/original/face', './datasets/original/nonface'
    # 		pickle op will store the serialized data under  './datasets/original/'
	
    # load dataset. (total 1000 samples, use num_train samples to fit adaboost model)
    num_train = 600
    dataset = get_data(num_train=num_train)
    X_train, y_train, X_test, y_test = dataset['X_train'], dataset['y_train'], dataset['X_test'], dataset['y_test']
    print('datasize: 1000, use %d samples to fit adaboost model.' % num_train)
    # setup model parameter.
    #       DecisionTreeClassifier
    #           max_depth: The maximum depth of the tree.
    #           min_samples_leaf:
    #           min_samples_split: The minimum number of samples required to split an internal node
    #       AdaBoostClassifier:
    #           weak_classifier: default use DecisionTreeClassifier
    #           n_weakers_limit: maximum number of weak classifier to boost an adaboost model.
    max_depth = 2
    min_samples_leaf = 1
    min_samples_split = None
    n_weakers_limit = 50

    print('\nUse DecisionTreeClassifier as weak classifiers.')
    print('\nTree settings:\n- max_depth = %d\n- min_samples_leaf = %s\n- min_samples_split = %s'
           % (max_depth, str(min_samples_leaf), str(min_samples_split)))
    print('\nAdaboost model settings:\n- n_weakers_limit = %d' % n_weakers_limit)
    print('\nStart boosting the model:')

    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    ada_model = AdaBoostClassifier(weak_classifier=dt, n_weakers_limit=n_weakers_limit)
    ada_model.fit(X_train, y_train,verbose=True)
    
    # predict labels for X_train and X_test and compute accuracy.
    y_pred_train = ada_model.predict(X_train)
    y_pred_test = ada_model.predict(X_test)
    train_accuracy = compute_accuracy(y_pred_train, y_train)
    val_accuracy = compute_accuracy(y_pred_test, y_test)

    print('\ntraining accuracy: %.3f.\nvaluation accuracy: %.3f.' % (train_accuracy, val_accuracy))
    
    # Classification Report
    target_names = ['nonface', 'face']
    print('\nClassification report for training data:\n\n- Classifier: %s\n\n- Report:\n%s' 
            % (ada_model, classification_report(y_train, y_pred_train, target_names=target_names)))
    print('\n\nClassification report for testing data:\n\n- Classifier: %s\n\n- Report:\n%s' 
            % (ada_model, classification_report(y_test, y_pred_test, target_names=target_names)))
    # wirte report into report.txt
    filename = 'report.txt'
    with open(filename, 'w') as f:
        target_names = ['nonface', 'face']
        f.write('Classification report for training data:\n\n- Classifier: %s\n\n- Report:\n%s' 
                % (ada_model, classification_report(y_train, y_pred_train, target_names=target_names)))
        f.write('\n\nClassification report for testing data:\n\n- Classifier: %s\n\n- Report:\n%s' 
                % (ada_model, classification_report(y_test, y_pred_test, target_names=target_names)))

    # fit a decision tree, use as a baseline
    print('\nFit a decision tree...')
    d = DecisionTreeClassifier()
    d.fit(X_train, y_train)
    dt_accuracy = d.score(X_test, y_test)
    print('training accuracy: %.5f' % d.score(X_train, y_train))
    print('test accuracy: %.5f' % dt_accuracy)

    # visualize result.
    # 
    # plot the ensemble prediction accuracy after each boost
    #
    print('\nVisualize result.')
    ada_accuracy_train = np.zeros((n_weakers_limit,))
    for i, y_pred in enumerate(ada_model.staged_predict(X_train)):
        ada_accuracy_train[i] = compute_accuracy(y_pred, y_train)

    ada_accuracy_test = np.zeros((n_weakers_limit,))
    for i, y_pred in enumerate(ada_model.staged_predict(X_test)):
        ada_accuracy_test[i] = compute_accuracy(y_pred, y_test)
    
    # study the behavior of AdaBoost Classifier
    # change the power of weak classifier (change tree depth)
    ada_models_accuracys = [] # store different AdaBoost model behaviors
    n_weakers = 50
    for dt_depth in range(1, 6):
        print('Fits a AdaBoost model with tree depth %d' %dt_depth)
        dt = DecisionTreeClassifier(max_depth=dt_depth)
        ada = AdaBoostClassifier(weak_classifier=dt, n_weakers_limit=n_weakers)
        ada.fit(X_train, y_train,verbose=False)
        accuracys = np.zeros((n_weakers,))
        for i, y_pred in enumerate(ada.staged_predict(X_test)):
            accuracys[i] = compute_accuracy(y_pred, y_test)
        print('accuracy: %.3f' % accuracys[-1])
        ada_models_accuracys.append(accuracys)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('training accuracy and test accuracy')

    ax.plot(np.arange(n_weakers_limit) + 1, ada_accuracy_train,
            label='AdaBoost Train',
            color='blue')

    ax.plot(np.arange(n_weakers_limit) + 1, ada_accuracy_test,
            label='AdaBoost Test',
            color='red')
    
    ax.set_xlabel('n_classifiers')
    ax.set_ylabel('accuracy')

    leg = ax.legend(loc='lower right', fancybox=True)
    leg.get_frame().set_alpha(0.7)

    
    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111)
    ax.set_title('AdaBoost Test accuracy with different weak decision tree classifier')

    ax.plot(np.arange(n_weakers) + 1, ada_models_accuracys[0],
            label='tree depth=1',
            color='blue')
    ax.plot(np.arange(n_weakers) + 1, ada_models_accuracys[1],
            label='tree depth=2',
            color='red')
    ax.plot(np.arange(n_weakers) + 1, ada_models_accuracys[2],
            label='tree depth=3',
            color='green')
    ax.plot(np.arange(n_weakers) + 1, ada_models_accuracys[3],
            label='tree depth=4',
            color='gold')
    ax.plot(np.arange(n_weakers) + 1, ada_models_accuracys[4],
            label='tree depth=5',
            color='orange')

    ax.plot(np.arange(n_weakers_limit) + 1, [dt_accuracy] * n_weakers_limit, 'k--', 
            label='Decision Tree')

    ax.set_xlabel('n_classifiers')
    ax.set_ylabel('accuracy')
    leg = ax.legend(loc='lower right', fancybox=True)
    leg.get_frame().set_alpha(0.7)

    plt.show()