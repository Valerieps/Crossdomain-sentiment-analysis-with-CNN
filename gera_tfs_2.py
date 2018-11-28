# -*- coding: utf-8 -*-

#!/usr/bin/env python3

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import argparse
import itertools
import operator
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
import csv
	
def read_corpus_y(path):
	""" Read a dataset file in the format *class text* in each line and 
	returns an array of documents corpus_x and an array of classes corpus_y
	Input: A path to a file 
	Output: array of documents, array of classes
	"""
	with open(path, encoding="utf8", errors='ignore') as fin:
		lines = fin.readlines()
	lines = [ x.strip().split() for x in lines ]
	#lines = [ x for x in lines if x ]
	#corpus_x = [ x[1:] for x in lines ]
	#corpus_x = [ " ".join(x[1:]) for x in lines ]
	corpus_y = [x[0] for x in lines ]
	return np.array(corpus_y)

def read_corpus1(path):
	""" Read a dataset file in the format *class text* in each line and 
	returns an array of documents corpus_x and an array of classes corpus_y
	Input: A path to a file 
	Output: array of documents, array of classes
	"""
	with open(path, encoding="utf8", errors='ignore') as fin:
	    for line in fin:
                l = line.strip().split()
                #print(l)
                corpus_x =  " ".join(l[1:])
                #print('1')
                #corpus_y = [l[0]]
                #print(str(corpus_x))
                yield corpus_x
                
def read_folds1(path):
	"""Read a fold dataset file from CV process in the format * index class text* and 
	returns and array of documents corpus_x and an array of classes corpus_y
	E.g.: 1515 10 from dackt about linux
	Input: A path to a file
	"""
		
	with open(path, encoding="utf8", errors='ignore') as fin:
	    for line in fin:
                l = line.strip().split()
                #print(l)
                corpus_x =  " ".join(l[2:])
                #print('1')
                #corpus_y = [l[1]]
                #print(str(corpus_x))
                yield corpus_x
                
def read_folds_y(path):
    """Read a fold dataset file from CV process in the format * index class text* and 
    returns and array of documents corpus_x and an array of classes corpus_y
    E.g.: 1515 10 from dackt about linux
    Input: A path to a file"""
    
    with open(path, encoding="utf8", errors='ignore') as fin:
        lines = fin.readlines()
    lines = [ x.strip().split() for x in lines ]
    corpus_y = [x[1] for x in lines ]
    return np.array(corpus_y)

def dump_svmlight(filepath, iterable):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(iterable)
    
def generate_tfidf(dir_corpus, dir_folds, dirSave, k, mindf):
	""" Generates TFIDF from input file
	Input: corpus and files from cross-validation folds
	Output: returns vocabulary and saves TFIDF files to disk
	""" 

	#transforms each k-fold to document-term matrix and saves to svmlight format 
	if dir_folds:
		for i in range(k):
			fold_x_train = read_folds1(dir_folds+'/train'+str(i))
			fold_y_train = read_folds_y(dir_folds+'/train'+str(i))
			tfidfvectorizer = TfidfVectorizer(sublinear_tf=True, min_df=mindf)
			tfidfvectorizer.fit(fold_x_train)
			fold_x_train = read_folds1(dir_folds+'/train'+str(i))
			vectortfidf = tfidfvectorizer.transform(fold_x_train)
			print("Shape train fold:"+str(i), vectortfidf.shape)
			#saving TFIDF into svmlight format
			dump_svmlight_file(vectortfidf, fold_y_train.astype(int), dirSave+'/train'+str(i)+'.tfidf')

			fold_x_test = read_folds1(dir_folds+'/test'+str(i))
			fold_y_test = read_folds_y(dir_folds+'/test'+str(i))
			vectortfidf = tfidfvectorizer.transform(fold_x_test)
			print("Shape test fold:"+str(i),vectortfidf.shape)
			#print(vectortfidf[0].toarray())
			#saving TFIDF into svmlight format
			dump_svmlight_file(vectortfidf, fold_y_test.astype(int), dirSave+'/test'+str(i)+'.tfidf')
			
			#saving fold vocabulary
			save_vocabulary(dirSave+'/vocab'+str(i)+'.csv',tfidfvectorizer.vocabulary_,0,1)
			print("Saving vocabulary tfidf "+ str(i))

	return(tfidfvectorizer.vocabulary_)

def generate_tf(dir_corpus, dir_folds, dirSave, k, mindf):
	""" Generates TF from input file
	Input: corpus and files from cross-validation folds
	Output: returtns vocabulary and saves TF files to disk
	""" 
	#learns vocabulary and idf from corpus
	corpus_gen = read_corpus1(dir_corpus)
	#print(corpus_gen)
	corpus_y = read_corpus_y(dir_corpus)
	#corpus_x, corpus_y = [i,j for i, j in corpus_gen]
	print('bla')	
	tfVectorizer = CountVectorizer(min_df=mindf)
	tfVectorizer.fit(corpus_gen)
	#print(tfVectorizer.vocabulary_)
	
	#fits corpus to document-term matrix and saves to svmlight format
	corpus_gen = read_corpus1(dir_corpus)
	vectortf = tfVectorizer.transform(corpus_gen)
	print("Shape corpus:", vectortf.shape)
	dump_svmlight_file(vectortf, corpus_y.astype(int), dirSave+'corpus.tf')
	print(type(vectortf), type(corpus_y.astype(int)))

	#transforms each cross-validation fold to document-term matrix and saves to svmlight format 
	if dir_folds:
		for i in range(k):
			fold_x = read_folds1(dir_folds+'/train'+str(i))
			fold_y = read_folds_y(dir_folds+'/train'+str(i))
			vectortf = tfVectorizer.transform(fold_x)
			#saving TFIDF into svmlight format
			dump_svmlight_file(vectortf, fold_y.astype(int), dirSave+'/train'+str(i)+'.tf')
			print("Shape train fold:"+str(i), vectortf.shape)

			fold_x = read_folds1(dir_folds+'/test'+str(i))
			fold_y = read_folds_y(dir_folds+'/test'+str(i))
			vectortf = tfVectorizer.transform(fold_x)
			#saving TFIDF into svmlight format
			dump_svmlight_file(vectortf, fold_y.astype(int), dirSave+'/test'+str(i) +'.tf')
			print("Shape test fold:"+str(i),vectortf.shape)		
						

	return(tfVectorizer.vocabulary_)

def save_vocabulary(dirSave, vocab,tf,tfidf):
	"""Saves vocabulary to disk
	"""
	if tf:
		w = csv.writer(open(dirSave+"/vocabulary.csv", "w"))
		for key, val in sorted(vocab.items(),key=operator.itemgetter(1)):
			w.writerow([key, val])
	if tfidf:
		#print("Saving vocabulary "+ dirSave)
		w = csv.writer(open(dirSave, "w"))
		for key, val in sorted(vocab.items(),key=operator.itemgetter(1)):
			w.writerow([key, val])
		


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Creates TFIDF and TF files in svmlight format and learns vocabulary saving to csv format.')
	parser.add_argument('-ic', '--inputCorpus', required=True, help='Path to preprocessed corpus of data. Format *class text* 1 instance per line.')
	parser.add_argument('-if', '--inputFolds', required=True, help='Path to pre-defined cross-validation folds.')
	parser.add_argument('-o', '--output', required=True, help='Output path to files in TF or TFIDF in svmlight format.')
	parser.add_argument('-k', '--folds', default=5, required=False, help='[Optional] (default = 5) Number of folds.', type=int)
	parser.add_argument('-tf', '--tf', default=0, required=False, help='[Optional] (default = 0) Generates tf.', type=int)
	parser.add_argument('-tfidf', '--tfidf', default=0, required=False, help='[Optional] (default = 0) Generates tdidf.', type=int)  
	parser.add_argument('-v', '--vocab', default=0, required=False, help='[Optional] (default = 0) Generates vocabulary. If TF, from corpus. If TFIDF from each fold', type=int)
	parser.add_argument('-mindf', '--mindf', default=1, required=False, help='[Optional] (default = 1) Removes terms that appear in less than mind_df documents.', type=int)  


	args = parser.parse_args()

	if args.tf:
		print("Generating TF...")
		vocabulary = generate_tf(args.inputCorpus, args.inputFolds, args.output, args.folds, args.mindf)
	
	if args.tfidf:
		print("Generating TFIDF...")
		vocabulary = generate_tfidf(args.inputCorpus, args.inputFolds, args.output, args.folds, args.mindf)

	if args.vocab:
		print("Saving vocabulary...")
		if args.tf:
			save_vocabulary(args.output, vocabulary,1, 0)



































