# -*- coding: utf-8 -*-

#!/usr/bin/env python3

from sklearn.model_selection import StratifiedKFold
import numpy as np
import argparse

def read_corpus(path):
	""" Read a dataset file in the format *class text* in each line and 
	returns an array of documents corpus_x and an array of classes corpus_y
	Input: A path to a file 
	Output: array of documents, array of classes
	"""
	with open(path, encoding="utf8", errors='ignore') as fin:
		lines = fin.readlines()
		lines = [ x.strip().split() for x in lines ]
	lines = [ x for x in lines if x ]
	corpus_x = [ x[1:] for x in lines ]
	corpus_y = [x[0] for x in lines ]
	return np.array(corpus_x), np.array(corpus_y)

def save_corpus(filename,index, y,x):
	""" Writes a new file in the format *index class text* in each line
	E.g.: 1515 10 from dackt about linux
	Input: filename, array of indexes from dataset, array of classes and array of documents
	Output: writes a file to disk
	"""
	for instance in zip(index,y,x):
		with open(filename, 'a') as f:           
		    f.write(str(instance[0]))
		    f.write(" ")
		    f.write(str(instance[1]))
		    f.write(" ")
		    f.write(' '.join(instance[2]))
		    f.write("\n")

def save_corpus1(filename,index, y,x):
	""" Writes a new file in the format *index class text* in each line
	E.g.: 1515 10 from dackt about linux
	Input: filename, array of indexes from dataset, array of classes and array of documents
	Output: writes a file to disk
	"""
	for instance in zip(index,y,x):
		with open(filename, 'a') as f:           
		    #f.write(str(instance[0]))
		    #f.write(" ")
                    f.write(' '.join(instance[2]))
                    f.write(",")
                    f.write(str(instance[1]))
                    f.write(",")
                    f.write(str(instance[0]))
                    f.write("\n")

#split datasets in k-fold cross validation and save each fold
def split_data(n_folds,output, path_train=False, path_test=False):
    """ Receives a path to dataset file (train and test file) or files (train file and test file)
    saves k-fold files in a stratified CV process to output path
    Input: number of folds to split, path to save file folds, path to a training file, path to a testing file (if so)
    Output: k-training files and k-testing files written to disk to the output path
    """
    
    #define stratified K-fold
    kf = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=57)
    
    #loads files
    if path_train:
        X_train,Y_train = read_corpus(path_train)
    if path_test:
        X_test,Y_test = read_corpus(path_test)

    #cross-validation process with train and test in different input files
    if path_train and path_test:
        i=0
        for train_index, test_index in kf.split(X_train,Y_train):
            X_train = X_train[test_index]
            y_train =  Y_train[test_index]

            print(i)
            print(X_train.shape, y_train.shape)
            print(" ")

            save_corpus(output+"train"+str(i), y_train, X_train)
            
            i=i+1

        i=0
        for train_index, test_index in kf.split(X_test,Y_test):

            x_test = X_test[test_index]
            y_test =  Y_test[test_index]

            print(i)
            print(x_test.shape, y_test.shape)
            print(" ")

            save_corpus(output+"test"+str(i), y_test,x_test)
            i=i+1
    
    #cross-validation process with train and test in the same file        
    if path_train and not path_test:
        i=0
        for train_index, test_index in kf.split(X_train,Y_train):
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = Y_train[train_index], Y_train[test_index]
            
            print(i)
            print(X_train_fold.shape, y_train_fold.shape)
            print(X_test_fold.shape, y_test_fold.shape)
            print(" ")
            
            save_corpus(output+"train"+str(i), train_index, y_train_fold,X_train_fold)
            save_corpus(output+"test"+str(i), test_index, y_test_fold, X_test_fold)
            save_corpus1(output+"train"+str(i)+'.csv', train_index, y_train_fold,X_train_fold)
            save_corpus1(output+"test"+str(i)+'.csv', test_index, y_test_fold, X_test_fold)
            i=i+1

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Split input dataset into k folds of cross-validation.')
	parser.add_argument('-itr', '--inptrain', required=True, help='Dataset train file. If dataset is not split in train/test, use this parameter to inform the whole corpus file')
	parser.add_argument('-ite', '--inptest', required=False, help='Dataset test file.')
	parser.add_argument('-o', '--outputpath', required=True, help='Directory output path to folds.')
	parser.add_argument('-k', '--folds', default=5, required=True, help='Number of folds for K fold cross-validation.', type=int)


	args = parser.parse_args()	
	split_data(args.folds, args.outputpath ,args.inptrain, args.inptest)
