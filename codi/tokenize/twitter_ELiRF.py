#! /usr/bin/python3
#!encoding:utf8

__version__ = "0.33"
__date__ = "20150608:1807"

import numpy
import os
import random
import re
import sys

from gensim.models import word2vec


try:
    import freeling
except ImportError:
    from pyELiRF.nlp import freeling
try:
    import twokenize_ES
except ImportError:
    from pyELiRF.twitter import twokenize_ES
try:
    import emoticons_ES
except ImportError:
    from pyELiRF.twitter import emoticons_ES

import scipy.sparse
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC



import numpy as np
import array
from sklearn.base import is_classifier# BaseEstimator, ClassifierMixin, clone,
from sklearn.utils.validation import _num_samples
from sklearn.multiclass import _predict_binary
import scipy.sparse as sp


regex_elongated2 = re.compile(r'([A-Z])\1{2,}', re.UNICODE + re.IGNORECASE) #repetitions of 3 or more equal caracters in a item. Ex. 'holaaaa'
regex_elongated1 = re.compile(r'([A-Z])\1{1,}', re.UNICODE + re.IGNORECASE) #repetitions of 2 or more equal caracters in a item. Ex. 'holaa'
def reduce_elongate(s, n=2):
    if n == 1:
        return regex_elongated1.sub('\g<1>', s)
    else:
        return regex_elongated2.sub('\g<1>', s)

def extract_chars(sentence, fill='<SPACE>'):
    spt = sentence.split()
    chars = []
    for word in spt:
        chars += [c for c in word] + [fill]
    return chars[:-1]


def extract_nchars(sentence, n=3, uni=False, fill='<SPACE>'):
    nchars = []
    spt = sentence.split()
    for word in spt:
        L = len(word)
        i=0
        if uni:
            j=1
        else:
            j=min(L,n)
        while j <= L:
            ph = word[i:j]
            nchars.append(ph)
            j += 1
            if j-i-1 == n:
                i += 1
        nchars.append(fill)
    return nchars[:-1]
            




def read_tweets(file_name, columns = 4, label_column=None, multilabel=False, multisplit='+'):
    tweets = {}
    if label_column != None:
        labels = {}
    for line in open(file_name):
        spt = line.strip().split(maxsplit=columns-1)
        tid = spt[0]
        tweets[tid] = spt[-1]
        if label_column != None:
            if multilabel:
                labels[tid] = spt[label_column-1].replace(',', '|').split('|')
            else:
                labels[tid] = spt[label_column-1]
    if label_column != None:
        return tweets, labels
    return tweets





def read_tweets_list_ORIGINAL(file_name, with_reference=True, clean_mode=2):
    # READ TWEET FILE
    ltweets = []
    for line in open(file_name):
        tweet_info = {}
        tid, tuser, tlabel, raw_tweet = line.split(maxsplit=3)
        tweet_info['id'] = tid
        tweet_info['user'] = tuser
        if with_reference:
            tweet_info['labels'] = tlabel
        ctweet = clean_tweet(raw_tweet, mode=clean_mode)
        tweet_info['text'] = ctweet
        ltweets.append(tweet_info)
    return ltweets




def read_tweets_list(file_name, with_reference=True, clean_mode=2, fields=None, label_column=None):
    # READ TWEET FILE
    if fields == None:
        fields = ["id", "user", "labels", "text"]
    columns = len(fields)
    ltweets = []
    for line in open(file_name):
        tweet_info = {}
        #tid, tuser, tlabel, raw_tweet
        finfo = line.split(maxsplit=columns-1)
        for c in range(columns):
            tweet_info[fields[c]] = finfo[c]
        if clean_mode != None:
            tweet_info[fields[columns-1]] = clean_tweet(tweet_info[fields[columns-1]], mode=clean_mode)
        ltweets.append(tweet_info)
    return ltweets



def read_labels(file_name, label_column=None, extra=False, multilabel=False):
    tlabels = {}
    textra = {}
    if label_column == None: label_column = 0
    for line in open(file_name):
        spt = line.split()
        tid = spt[0]
        label = spt[label_column-1]
        if multilabel:
            labs = label.replace(',', '|').replace('+', '|').split('|')
            tlabels.setdefault(tid,[]).extend(labs)
        else:
            tlabels[tid] = label
        if extra:
            info = spt[1:label_column] + spt[label_column:]
            textra.setdefault(tid,[]).extend(labs)
    return tlabels

def __read_labels(file_name, coln=3,):
    labels = {}
    for line in open(file_name):
        spt = line.split()
        tid = spt[0]
        labels[tid] = spt[coln-1]
    return labels

def read_file(file_name, columns=3, extra=False, uniq=False):
    tweets = []
    labels = []
    extra_info = []
    extra = extra and columns > 3
    for line in open(file_name):
        spt = line.strip().split(maxsplit=columns-1)
        #tid = spt[0]
        tweet = spt[-1]
        if not uniq or tweet not in tweets:
            tweets.append(tweet)
            labels.append(spt[-2].replace(',', '|').split('|'))
            if extra:
                extra_info.append(spt[1:-2])
    #print (len(tweets),len(labels))
    if extra:
        return tweets, labels, extra_info
    return tweets, labels


def add_labels(ltweets, labels):
    i = 0
    while (i < len(ltweets)):
        tid = ltweets[i]['id']
        label = labels.get(tid, None)
        if label:
            ltweets[i]['labels'] = label
            i += 1
        else:
            ltweets.pop(i)

def replace_labels(tweets, old_label, new_label):
    for tweet in tweets:
        if tweet['labels'] == old_label:
            tweet['labels'] = new_label


def read_corpora(train_file, test_file, test_lab_file=None, experiment_type='evaluate', nfolds=1):
    train_tweets = read_tweets(train_file, with_reference=True)
    #print (len(train_tweets))
    if experiment_type == 'predict':
        test_tweets = read_tweets(test_file, with_reference=False)
    if experiment_type == 'evaluate':
        test_tweets = read_tweets(test_file, with_reference=False)
        test_labs = read_labels(test_lab_file, coln = 3)
        add_labels(test_tweets, test_labs)
    if experiment_type == 'validation':
        ntrain, ntest = choice_nfold(train_tweets, nfolds)
    else:
        ntrain, ntest = [train_tweets], [test_tweets]
    return ntrain, ntest


def voting_monolabel_dic(hypos):
    result = {}
    for k in hypos[0].keys():
        all1 = [hypo.get(k, None) for hypo in hypos] #+ [hypos[2][k]]
        all2 = sorted([(all1.count(x), x) for x in set(all1)], reverse=True)
        m = all2[0][1]
        result[k] = m
    return result


def voting_monolabel_dic_weight(hypos, w):
    result = {}
    for k in hypos[0].keys():
        t = {}
        for i in range(len(hypos)):
            t[hypos[i][k]] = t.get(hypos[i][k], 0.0) + w[i]
        result[k] = sorted([(v,k) for k, v in t.items()], reverse=True)[0][1]
    return  result








def choice_nfold(samples, n, random_sets=False):
    nsamples = len(samples)
    test_size = nsamples // n
    train = []
    test = []
    if random_sets == False:
        extra = nsamples - test_size * n
        i = 0
        for fold in range(n):
            j = i + test_size + (1 if fold < extra else 0)
            test.append(samples[i:j])
            train.append(samples[0:i]+samples[j:])
            i = j
        
    else:
        remaining = set(range(nsamples))
        for fold in range(n):
            if fold < n - 1:            
                j_test = random.sample(remaining, test_size)
            else:
                j_test = list(remaining)
            #j_test.sort() # innecessari
            test.append([samples[j] for j in j_test])
            train.append([samples[i] for i in range(len(samples))
                                                        if i not in j_test])
            remaining.difference_update(j_test)
        
    return train, test


def evaluate_just_microf1(lreal, lhypo):
    e = MicroMacroEvaluator()
    micro, macro, detail = e.evaluate_micro_and_macro(lreal, lhypo)
    return micro[-1]
    


# ####################################
# ### ORIGINAL DEL TWEETLID_2014 #####
# ####################################
# ########### B E G I N ##############
# ####################################


def do_experiment(ntrain, ntest, experiment_type, **parm):
    return new_do_experiment(ntrain, ntest, experiment_type, **parm)


def new_do_experiment(ntrain, ntest, experiment_type, nfolds=1,
        vectorizers = None, 
        kernel='linearsvc', cs=None,
        force_monolabel = False, one_vs_rest=False, use_auxi=False,
        max_multilabel=1000,
        with_prob=False,
        is_multilabel=True,
        thrs=None,
        what_max="macro_F1",
        verbose = 0):

    feval = evaluate_results_multilabel if is_multilabel else evaluate_results_monolabel
    if cs == None:
        cs = [2]
    if vectorizers == None:
        vectorizers = ['tfcb5']
    result = []
    all_results = []
    best_measure_avg = 0.0

    if verbose == 2:
        if experiment_type == 'validation':
            print ('C:\tmacro_F1 avg:\tmicro_F1 avg:')
        elif experiment_type == 'evaluate':
            print ('C:\tmacro_F1:\tmicro_F1:')
    if verbose == 3:
        if experiment_type in ('validation', 'evaluate'):
            print ('C\tmacro_P\tmacro_R\tmacro_F1\tmicro_P\tmicro_R\tmicro_F1')

    for c in cs:
        # BEGIN FOLDS
        measures = {}
        hypos = []
        f1_acum = 0.0
        f1_micro_acum = 0.0
        fold_num = nfolds if experiment_type == 'validation' else 1
        for fold in range(fold_num):
            #GET TRAIN and TEST PARTITION
            train = ntrain[fold]
            test = ntest[fold]
            #PREPARE TRAIN
            # l'usuari en realitat no s'utilitza, fixat en li = LanguageIdentificator(.....
            x_train, x2_train, y_train = prepare_train_samples_whole_tweet(train, user = True, force_monolabel=force_monolabel, is_multilabel=is_multilabel)
            x_train_clean = []
            x2_train_clean = []
            y_train_clean = []
            for i in range(len(x_train)):
                #if 'und' not in y_train[i] and 'other' not in y_train[i]:
                if len(y_train[i]) <= max_multilabel:
                    x_train_clean.append(x_train[i])
                    y_train_clean.append(y_train[i])
                    x2_train_clean.append(x2_train[i])   
            x_train, x2_train, y_train = x_train_clean, x2_train_clean, y_train_clean
            #PREPARE TEST
            if experiment_type in ('validation', 'evaluate'):
                x_test, x2_test, y_test = prepare_test(test, gold_standard=True)
            else:
                x_test, x2_test = prepare_test(test, gold_standard=False)
            #LEARN MODEL
            li = LanguageIdentificator(clf_type=kernel, multilabel = one_vs_rest, vectorizers = vectorizers, C = c, use_auxi = use_auxi)
            li.learn_model(x_train, y_train)
            #PREDICT
            if with_prob:
                rp = li.predict_with_prob(x_test)
                #pred = [x for x,y in pred2]
                pred = li.predict(x_test)
                result.append(rp)
            else:
                pred = li.predict(x_test, thrs)
                result.append(pred)
    
            # EVALUATION
            if experiment_type in ('validation', 'evaluate'):
                global_r, detailed_r, accuracy, matrix1, matrix2, matrix3 = feval(y_test, pred)
                for key in global_r['<GLOBAL>'].keys():
                    measures[key] = measures.get(key, 0) + global_r['<GLOBAL>'][key]
                f1_acum += global_r['<GLOBAL>']['macro_F1']
                if verbose > 3:
                    show_results_multilabel(global_r, detailed_r, accuracy, matrix1, matrix2, matrix3)
            hypos.append('%s\t%s' % (tid, langs) for tid, langs in zip([a['id'] for a in test], pred))
        if experiment_type == 'validation':
            for key in measures:
                measures[key] /= nfolds
            if measures[what_max] > best_measure_avg:
                comment = '+'
                best_measure_avg = measures[what_max]
            else:
                comment = ''
            if verbose == 2:
                print ('%f\t%f\t%f\t%s' % (c, measures['macro_F1'], measures['micro_F1'], comment ))
            elif verbose == 3:
                print ('%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s' % (c, measures['macro_P'], measures['macro_R'], measures['macro_F1'], measures['micro_P'], measures['micro_R'], measures['micro_F1'], comment ))
        elif experiment_type == 'evaluate':
            if verbose == 2:
                print ('%f\t%f\t%f' % (c, measures['macro_F1'],  measures['micro_F1']))
            elif verbose == 3:
                print ('%f\t%f\t%f\t%f\t%f\t%f\t%f' % (c, measures['macro_P'], measures['macro_R'], measures['macro_F1'], measures['micro_P'], measures['micro_R'], measures['micro_F1']))
        all_results.append((measures, hypos, result))
    return all_results




def do_experiment_notebook(ntrain, ntest, experiment_type, kernel='linearsvc', vectorizers = None, cs=[3], nfolds=1, force_monolabel = False, one_vs_rest=False, use_auxi=False, verbose = 0, max_multilabel=1000, with_prob=False):
    best_f1_avg = 0.0
    if vectorizers == None:
        #vectorizers = ['tfcb5']
        vectorizers = ['tfcb5']#, 'tfwd1']
    result = []
    for c in cs:
        # BEGIN FOLDS
        f1_acum = 0.0
        f1_micro_acum = 0.0
        fold_num = nfolds if experiment_type == 'validation' else 1
        measures = {}
        for fold in range(fold_num):
            #GET TRAIN and TEST PARTITION
            train = ntrain[fold]
            test = ntest[fold]
            #PREPARE TRAIN
            # l'usuari en realitat no s'utilitza, fixat en li = LanguageIdentificator(.....
            x_train, x2_train, y_train = prepare_train_samples_whole_tweet(train, user = True, force_monolabel=force_monolabel)
            x_train_clean = []
            x2_train_clean = []
            y_train_clean = []
            for i in range(len(x_train)):
                #if 'und' not in y_train[i] and 'other' not in y_train[i]:
                if len(y_train[i]) <= max_multilabel:
                    x_train_clean.append(x_train[i])
                    y_train_clean.append(y_train[i])
                    x2_train_clean.append(x2_train[i])   
            x_train, x2_train, y_train = x_train_clean, x2_train_clean, y_train_clean
            #PREPARE TEST
            if experiment_type in ('validation', 'evaluate'):
                x_test, x2_test, y_test = prepare_test(test, gold_standard=True)
            else:
                x_test, x2_test = prepare_test(test, gold_standard=False)
            #LEARN MODEL
            li = LanguageIdentificator(clf_type=kernel, multilabel = one_vs_rest, vectorizers = vectorizers, C = c, use_auxi = use_auxi)
            li.learn_model(x_train, y_train)
            #PREDICT
            if with_prob:
                pred2 = li.predict_with_prob(x_test)
                pred = [x for x,y in pred2]
                result.append(pred2)
            else:
                pred = li.predict(x_test)
                result.append(pred)
            #EVALUATION
            if experiment_type in ('validation', 'evaluate'):
                global_r, detailed_r, accuracy, matrix1, matrix2, matrix3 = evaluate_results_multilabel(y_test, pred)
                if verbose > 1: 
                    show_results_multilabel(global_r, detailed_r, accuracy, matrix1, matrix2, matrix3)                
                f1_acum += global_r['<GLOBAL>']['macro_F1']
            #else:
             #   for tid, langs in zip([a['id'] for a in test], pred):
              #      print (tid+'\t'+langs)
        if experiment_type == 'validation':
            #print ("----")
            f1_avg = (f1_acum / nfolds)
            if f1_avg > best_f1_avg:
                comment = '+'
                best_f1_avg = f1_avg
            else:
                comment = ''
            print ('C:%f\tF1 avg:\t%f\t%s' % (c, f1_avg, comment ))
        elif experiment_type == 'evaluate':
            #print ("----")
            print ('C:%f\tF1:\t%f' % (c, f1_acum))
        #print ("----")
    return result




def do_experiment_antic(ntrain, ntest, experiment_type, vct_type = 'tfwd4',cs=[3], nfolds=1,
                                    force_monolabel = False, one_vs_rest=False, use_auxi=False, max_multilabel=1000, 
                                    name=None, verbose=3):    
    """validation
    evaluate
    predict
    """
    all_results = []
    best_macro_f1_avg = 0.0
    for c in cs:
        # BEGIN FOLDS
        #f1_macro_acum = 0.0
        #f1_micro_acum = 0.0
        fold_num = nfolds if experiment_type == 'validation' else 1
        measures = {}
        hypos = []
        #print (type(fold_num), fold_num)
        for fold in range(fold_num):
            #GET TRAIN and TEST PARTITION
            train = ntrain[fold]
            test = ntest[fold]
            #PREPARE TRAIN
            x_train, y_train = prepare_train_samples_whole_tweet(train, user = False, force_monolabel=force_monolabel)# 0
            # 1
            x_train_clean = []
            x2_train_clean = []
            y_train_clean = []
            for i in range(len(x_train)):
                #if 'und' not in y_train[i] and 'other' not in y_train[i]:
                if len(y_train[i]) <= max_multilabel:
                    x_train_clean.append(x_train[i])
                    y_train_clean.append(y_train[i])
                    #x2_train_clean.append(x2_train[i])
            #x_train, x2_train, y_train = x_train_clean, x2_train_clean, y_train_clean
            x_train, y_train = x_train_clean, y_train_clean
            #PREPARE TEST
            if experiment_type in ('validation', 'evaluate'):
                x_test, _, y_test = prepare_test(test, gold_standard=True, user=False)
            else:
                x_test, _ = prepare_test(test, gold_standard=False, user=False)
            #LEARN MODEL
            li = LanguageIdentificator(multilabel = one_vs_rest, vct_type = vct_type, C = c, use_auxi = use_auxi)
            li.learn_model(x_train, y_train)
            #PREDICT
            pred = li.predict(x_test)
            #EVALUATION
            if experiment_type in ('validation', 'evaluate'):
                global_r, detailed_r, accuracy, matrix1, matrix2, matrix3 = evaluate_results_multilabel(y_test, pred)
                for key in global_r['<GLOBAL>'].keys():
                    measures[key] = measures.get(key, 0) + global_r['<GLOBAL>'][key]
                #show_results_multilabel(global_r, detailed_r, accuracy, matrix1, matrix2, matrix3)                
            else:
                for tid, langs in zip([a['id'] for a in test], pred):
                    print (tid+'\t'+langs)
            hypos.append('%s\t%s' % (tid, langs) for tid, langs in zip([a['id'] for a in test], pred))
        if experiment_type == 'validation':
            for key in measures:
                measures[key] /= nfolds
            if measures['macro_F1'] > best_macro_f1_avg:
                comment = '+'
                best_macro_f1_avg = measures['macro_F1']
            else:
                comment = ''
            if verbose > 2:
                print ('C:%f\tmacro_F1 avg:\t%f\tmicro_F1 avg:\t%f\t%s' % (c, measures['macro_F1'], measures['micro_F1'], comment ))
        elif experiment_type == 'evaluate':
            #print ("----")
            if verbose > 2:
                print ('C:%f\tmacro_F1:\t%f\tmicro_F1:\t%f' % (c, measures['macro_F1'],  measures['micro_F1']))
        #print ("----")
        all_results.append((measures, hypos))
    return all_results



def format_results(results, prefix=None, verbose=2):
    n = len(results)
    if prefix == None:
        prefix = [''] * n
    elif hasattr(prefix, '__iter__') and (len(prefix)) == n:
        prefix = [str(x)+'\t' for x in prefix]
    else:
        prefix = [str(prefix) + '\t']*n
    str_list = []
    for i in range(len(results)):
        if verbose == 1:
            w = ("macro_F1", "micro_F1")
        elif verbose == 2:
            w = ("macro_P", "macro_R", "macro_F1", "micro_P", "micro_R", "micro_F1")
        elif verbose == 3:
            w = ("tp", "fp", "fn", "macro_P", "macro_R", "macro_F1", "micro_P", "micro_R", "micro_F1")
        s = prefix[i] + '\t'.join('%f'%results[i][0][x] for x in w)
        str_list.append(s)
    return str_list



def do_tuning(experiment_opts, niter=4, epsilon = 0.0001, nsteps=8, argmax = 'macro_F1', verbose=3):    
    experiment_opts['experiment_type'] = 'validation'
    experiment_opts['verbose'] = verbose
    best_result = 0.0
    iter = 0
    final = False
    half_steps = int(nsteps/2)
    csqr = [x for x in range(-half_steps+1, half_steps+2)]
    while not final:
        #print ([2**x for x in csqr])
        experiment_opts['cs'] = (2**x for x in csqr)
        results = do_experiment(**experiment_opts)
        measure = [m[argmax] for m, h in results]
        maxm = max(measure)
        pos = measure.index(maxm)
        bvalue = csqr[pos]
        step = 2 * (bvalue - csqr[pos-1] if pos > 0 else csqr[pos+1] - bvalue) / (nsteps +2)
        csqr = [bvalue + x*step for x in range(-half_steps, half_steps+1)]
        if verbose > 1:
            print ("millor valor: %f, millora: %f" % (2**bvalue, abs(best_result - maxm)))
        iter += 1
        if iter == niter or abs(best_result - maxm) < epsilon:
            final = True
        best_result = maxm
    return 2**bvalue, maxm





def prepare_train_samples_whole_tweet(ltweets, user = False, force_monolabel = False, is_multilabel=True):

    if not is_multilabel:
        force_monolabel = False
    y, x,  x2 = [], [], []
    for i in range(len(ltweets)):
        all_info = ltweets[i]
        tlab, tweet = all_info['labels'], all_info['text']
        if user and "user" in all_info:
            tuser = all_info['user']
        else:
            tuser = 'twitteruser'
        #TODO:nyap!!

        if is_multilabel:
            if '/' in tlab:
                tlab = tlab.replace('+','/')
                langs = tlab.split('/')
            else:
                langs = tlab.split('+')
        else:
            langs = tlab

        if force_monolabel:
            x += [tweet] * len(langs)
            x2 += [tuser] * len(langs)
            y += [a.split('+') for a in langs]
        else:
            x += [tweet]
            x2 += [tuser]
            y += [langs]
    #print ('ltweets',len(ltweets))
    #print ('x', len(x))
    if user:
        return x, x2, y
    else:
        return x, y






def prepare_test(test, gold_standard=False,  user=False):
    x_test = []
    x2_test = []
    y_test = []
    for tweet_info in test:
        if user:
            tuser = tweet_info['user']
        else:
            tuser = 'twitteruser'
        tweet = tweet_info['text']
        if gold_standard:
            tlab = tweet_info['labels']
        x_test.append(tweet)
        x2_test.append(tuser)
        if gold_standard:
            y_test.append(tlab)
    if not gold_standard:
        return x_test, x2_test
    else:
        return x_test, x2_test, y_test


class PolarityPredictor():
    def __init__(self, vector, cls, extra_vector=None):
        self.set_vectorizer(vector)
        self.set_extra_vectorizer(extra_vector)
        self.set_classifier(cls)
    
    def set_vectorizer(self, vector):
        self.vector = vector

    def set_extra_vectorizer(self, vector):
        self.extra_vector = vector
        
    def set_classifier(self, cls):
        self.cls = cls

    def fit(self, train):
        x_train = []
        y_train = []
        for sample, label in train:
            x_train.append(sample)
            y_train.append(label)
        X_train = self.vector.fit_transform(x_train)
        if self.extra_vector != None:
            X2_train = self.extra_vector.extract_features(x_train)
            X_train = scipy.sparse.hstack([X_train, X2_train])
        self.cls.fit(X_train, y_train)
    
    def predict(self, x):
        X = self.vector.transform(x)
        if self.extra_vector != None:
            X2 = self.extra_vector.extract_features(x)
            X = scipy.sparse.hstack([X, X2])
        pred = self.cls.predict(X)
        return pred



# ####################################
# ########## C L A S S E S ###########
# ####################################



class MyOneVsRestClassifier(OneVsRestClassifier):

    def __init__(self, estimator, n_jobs=1):
        super(MyOneVsRestClassifier, self).__init__(estimator, n_jobs)


    def predict(self, X, thrs=None):
        """Predict multi-class targets using underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        Returns
        -------
        y : (sparse) array-like, shape = [n_samples] or [n_samples, n_classes].
            Predicted multi-class targets.
        """

        #check_is_fitted(self, 'estimators_')
        if thrs == None:
            thrs = []
        n_samples = _num_samples(X)
        n_classes = len(self.estimators_)
        if (hasattr(self.estimators_[0], "decision_function") and
                is_classifier(self.estimators_[0])):
            #print ("aqui3")
            #thresh = [0]
            #thresh = -0.05
            thrs = thrs + [0] * (n_classes-len(thrs))
            #print (len(thrs))
        else:
            thrs = thrs + [0.5] * (n_classes-len(thrs))
            #thresh = .5

        #print (len(thrs), n_classes)

        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                print (self.estimators_)
                pred = _predict_binary(e, X)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.label_binarizer_.classes_[np.array(argmaxima.T)]
        else:
            indices = array.array('i')
            indptr = array.array('i', [0])
            #print (type(self.estimators_))
            for i in range(n_classes):
            #for e in self.estimators_:
                e = self.estimators_[i]
                #print (_predict_binary(e, X))

                #print (_predict_binary(e, X).shape)
                #indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                indices.extend(np.where(_predict_binary(e, X) > thrs[i])[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)



    def predictORIGUNAL(self, X):
        """Predict multi-class targets using underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        Returns
        -------
        y : (sparse) array-like, shape = [n_samples] or [n_samples, n_classes].
            Predicted multi-class targets.
        """

        #check_is_fitted(self, 'estimators_')
        if (hasattr(self.estimators_[0], "decision_function") and
                is_classifier(self.estimators_[0])):
            thresh = 0
        else:
            thresh = .5

        n_samples = _num_samples(X)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                pred = _predict_binary(e, X)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.label_binarizer_.classes_[np.array(argmaxima.T)]
        else:
            indices = array.array('i')
            indptr = array.array('i', [0])
            for e in self.estimators_:
                indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)




class MyVectorizer():

    def __init__(self, types):
        #print ("-",types)
        if types == None:
            types = ['tfcb5']
        self.__vectorizers = [self.create_vectorizer(t) for t in types]
        self.__types = types


    def create_vectorizer(self, vct_type):#, atype, nrange)
        # nomes si es una cadena
        if isinstance(vct_type, str):
            if vct_type[:2] == 'tf':
                vtype = 'tfidf'
                vtype_auxi = 'tfidf'
            elif vct_type[:2] == 'ct':
                vtype = 'count'
                vtype_auxi = 'count'
            else:
                print ('MAL', vct_type, vct_type[:2])
                raise
            if vct_type[2:4] == 'cb':
                atype = 'char_wb'
            elif vct_type[2:4] == 'ch':
                atype = 'char'
            elif vct_type[2:4] == 'wd':
                atype = 'word'
            else:
                raise
            nrange = (1,int(vct_type[4:]))
            if vtype == 'tfidf':
                v = TfidfVectorizer(analyzer=atype, ngram_range=nrange)
            elif vtype == 'count':
                v = CountVectorizer(analyzer=atype, ngram_range=nrange)
        else:
            v = vct_type
        return v

    def __str__(self):
        return '+'.join(str(t) for t in self.__types)

    def fit_transform(self, samples):
        Xs = [v.fit_transform(samples) for v in self.__vectorizers] #]self.__vct.fit_transform(samples)
        if len(Xs) > 1:
            X = scipy.sparse.hstack(Xs)
        else:
            X = Xs[0]
        #print (dir(self.__vectorizers[0]))
        return X

    def transform(self, samples):
        Xs = [v.transform(samples) for v in self.__vectorizers] #]self.__vct.fit_transform(samples)
        if len(Xs) > 1:
            X = scipy.sparse.hstack(Xs)
        else:
            X = Xs[0]
        return X

class LanguageIdentificator:
    def __init__(self, clf_type = 'linearsvc', multilabel = False, vectorizers = None, vct_type = None, C = 3,  use_auxi = False):
        # "vct_type" parameter only for Backward compatibility
        if vct_type != None:
            vectorizers = vct_type
        self.__clf_extra = None
        self.__multilabel = multilabel
        #create vectorizers
        if isinstance(vectorizers, MyVectorizer):
            self.__vcts = vectorizers
        else:
            self.__vcts = MyVectorizer(vectorizers)
#        if multilabel and use_auxi:
#            self.__vct_extra = self.create_vectorizer(vtype_auxi, anal_type, (1, max_n))
        # create classifiers
        self.__clf = self.create_classifier(clf_type, C)
        if multilabel:
            self.__clf = MyOneVsRestClassifier(self.__clf, n_jobs=-2)
            if use_auxi:
                self.__clf_extra = self.create_classifier(clf_type, C)
        # 
        if multilabel:
            self.__mlb = MultiLabelBinarizer()
        else:
            self.__mlb = None

    def create_classifier(self, ctype, c=1):
        if ctype == 'linearsvc':
            c = LinearSVC(C=c)
            #c = LinearSVC(C=c,  class_weight ='auto')
            #c = LinearSVC(C=c,  class_weight = {'en':1, 'ca': 1, 'gl': 4, 'es':1, 'other':1})
        elif ctype == 'lsvc':
            c = SVC(kernel='linear', C=c, probability=True)
        return c
    
    def learn_model(self, samples, labels):
        if self.__multilabel:
            y = self.__mlb.fit_transform(labels)
        else:
            y = numpy.ravel(labels, order='C')
        X = self.__vcts.fit_transform(samples)
        self.__clf.fit(X, y)
        if self.__clf_extra != None:
            #X_auxi = self.__vct_extra.fit_transform(samples)
            y_auxi = numpy.ravel(labels, order='C')
            #self.__clf_extra.fit(X_auxi, y_auxi)
            self.__clf_extra.fit(X, y_auxi)


    def predict_with_prob(self, samples):
        X = self.__vcts.transform(samples)
        #return self.__clf.predict_proba(X)
        pb = self.__clf.decision_function(X)
        if self.__mlb != None:
            class_names = self.__mlb.classes_
        else:
            #print (self.__clf)
            #print (dir(self.__clf))
            class_names = self.__clf.classes_
            class_names = ['+'.join(i) for i in class_names]
        result = []
        for i in range(len(samples)):
            m = max(pb[i])
            pos = numpy.where(pb[i] == m)[0]
            result.append(('+'.join(class_names[ind] for ind in pos), m))
        return result, pb, class_names


    def predict(self, samples, thrs=None):
        X = self.__vcts.transform(samples)
        if thrs != None:
            pr = self.__clf.predict(X, thrs)
        else:
            pr = self.__clf.predict(X)
        if self.__clf_extra != None:
            pr_extra = self.__clf_extra.predict(X)
        
        if self.__multilabel:
            pr = self.__mlb.inverse_transform(pr)
        
        pred = [x if isinstance(x, (list, tuple)) else [x] for x in pr]
        if self.__clf_extra != None:
            pred_auxi = [x if isinstance(x, (list, tuple)) else [x] for x in pr_extra]
            # if prediction of clf is empty for a sample use de prediction of clf_extra
            pred = [pred[i] if len(pred[i]) > 0 else pred_auxi[i] for i in range(len(pred))]
        
        if self.__multilabel:
            pred = [a if 'und' not in a else [b for b in a if b != 'und'] for a in pred]
            pred = [a if 'other' not in a else [b for b in a if b != 'other'] for a in pred]
            pred = [a if len(a) > 0 else ['und'] for a in pred]
    #        print (set([a for a in pred if len(a) != 1]))
        pred = ['+'.join(sorted(a)) for a in pred]
        return pred



def show_results_multilabel(global_result, detailed_result, accuracy, matrix, matrix2, matrix_amb):
    
    
    #l1 = ["es", "pt", "ca", "en", "gl", "eu", "en+es", "es+eu", "ca+es", "en+pt", "ca+en", "es+gl", "en+eu", "en+gl", "es+pt", 'eu+es', 'gl+es', 'gl+pt', 'pt+en', "ca+pt", 'gl+pt+es', "es+gl+pt", "ca+es+eu", "en+es+eu", "ca+en+es", "en+es+gl", 'en+und', 'pt+und', 'es+ca', 'pt+gl', 'pt+gl+es', "und"]
    #l2 = ["es", "pt", "ca", "en", "gl", "eu", "en+es", "es+eu", "ca+es", "en+pt", "ca+en", "es+gl", "en+eu", "en+gl", "es+pt", 'eu+es', 'gl+es', 'gl+pt', 'pt+en', "ca+pt", 'gl+pt+es', "es+gl+pt", "ca+es+eu", "en+es+eu", "ca+en+es", "en+es+gl", 'en+und', 'pt+und', 'es+ca', 'pt+gl', 'pt+gl+es', "und"]


#    if False:
    #s1 = set(matrix.keys())
    #s2 = set(matrix2.keys())


    s2 = set()
    for i in matrix:
        for j in matrix[i].keys():
            s2.add(j)
    for i in matrix2:
        for j in matrix2[i].keys():
            s2.add(j)
    #print ([b for a,b in sorted((x.count('+'), x) for x in s4)])

    #l1 = sorted(matrix.keys())
    l1 = [a for _,_,a in sorted([(-accuracy[i]['hits']-accuracy[i]['errors'], i.count('+'), i) for i in matrix.keys()])]
    l1.remove('und') # put "und" the last
    l1.append('und')

    l2 = [a for _,_,a in sorted([(-accuracy[i]['hits']-accuracy[i]['errors'], i.count('+'), i) for i in matrix2.keys()])]
    #s1 = s2.difference(l1+l2)
    l3 = l1 + l2 + [a for _,a in sorted((i.count('+'), i) for i in s2.difference(l1+l2))]

    #l1 = [b for a,b in sorted((x.count('+'), x) for x in s1)]
    #l2 = [b for a,b in sorted((x.count('+'), x) for x in s2)]
        #print (len(l1), len(set(l1)))


    if len(matrix2) > 0:
        print ('\nCONFUSION MATRIX MONOLABEL')    
        print ('==========================')
    else:
        print ('\nCONFUSION MATRIX')    
        print ('================')

    print ('\t\t%s' % ('\t'.join(j for j in l3)))
    #for i in sorted(matrix):
    for i in l1:
        st = '\t'.join('%d' % matrix[i].get(j, 0) for j in l3)
        print ('%s\t%f\t%s' % (i, accuracy[i]['accuracy'], st))

    if len(matrix2) > 0:
        print ('\nCONFUSION MATRIX MULTILABEL')    
        print ('===========================')
        for i in l2:
            st = '\t'.join('%d' % matrix2[i].get(j, 0) for j in l3)
            print ('%s\t%f\t%s' % (i, accuracy[i]['accuracy'], st))

    if len(matrix_amb) > 0:
        print ('\nCONFUSION MATRIX AMBIGOUS')    
        print ('==========================')
        for i in sorted(matrix_amb):        
            st = '\t'.join('%d' % matrix_amb[i].get(j, 0) for j in l3)
            print ('%s\t%f\t%s' % (i, accuracy[i]['accuracy'], st))

    
    for i in sorted(detailed_result):
        print ('='*30)
        print ('GROUP:%s'% i)
        print (('\t'*2) + ('\t'.join(x for x in ('tp', 'fp', 'fn', 'P', 'R', 'F1'))))
        print ('\t' + '\t--'*6)
        for j in (sorted(detailed_result[i])):
            print ('\t%s\t%s\t%s' % (j,
                                    '\t'.join(('%6d'%detailed_result[i][j][x]) for x in ('tp', 'fp', 'fn')),
                                    '\t'.join(('%5f'%detailed_result[i][j][x]) for x in ('P', 'R', 'F1'))))
        print ('-'*10)
        print ('Macro-Precision:\t%f' % global_result[i]['macro_P'])
        print ('Macro-Recall:\t%f' % global_result[i]['macro_R'])
        print ('Macro-F1:\t%f' % global_result[i]['macro_F1'])
        print ('F1 of Macro:\t%f' % global_result[i]['F1_of_macro'])
        print ('-'*10)
        print ('tp:\t%d' % global_result[i]['tp'])
        print ('fp:\t%d' % global_result[i]['fp'])
        print ('fn:\t%d' % global_result[i]['fn'])
        print ('Micro-Precision:\t%f' % global_result[i]['micro_P'])
        print ('Micro-Recall:\t%f' % global_result[i]['micro_R'])
        print ('Micro-F1:\t%f' % global_result[i]['micro_F1'])


def evaluate_results_monolabel(ref, hypo):
    return evaluate_results_monolabel_lists(ref, hypo)


def evaluate_results_monolabel_dics(ref, hypo):
    keys = sorted(ref.keys())
    lref = [ref[k] for k in keys]
    lhypo = [hypo[k] for k in keys]
    return evaluate_results_monolabel_lists(lref, lhypo)


def evaluate_results_monolabel_lists(lref, lhypo):
    """lref and lhypo are lists having the same size
        
    labels 'other' and 'und' are considered equivalent
    """
    
    #TODO: some features could be configured using parameters
    

    "to record macro-averaging"
    macros = {}
    def add_count(label, group, t):        
        m0 = macros.setdefault(str(group), {}).setdefault(label, {'tp':0, 'fp':0, 'fn':0}) # tp, fp, fn        
        m0[t] += 1
        m1 = macros.setdefault('<GLOBAL>', {}).setdefault(label, {'tp':0, 'fp':0, 'fn':0}) # tp, fp, fn        
        m1[t] += 1

    
    hits = {} # hits
    errors = {} # errors
    matrix = {} # confusion matrix for non ambigous samples with one label in the reference
    "count hits, true positive, false positive, and false negative"
    "compute agglutinative matrix of confusion"
    for i in range(len(lref)):

        ref_str = lref[i].replace('other', 'und')
        hypo_str = lhypo[i].replace('other', 'und')

        "count hit and errors"
        if ref_str == hypo_str:
            hits[ref_str] = hits.get(ref_str, 0) + 1
        else:
            errors[ref_str] = errors.get(ref_str, 0) + 1

        "add to confusion matrix"
        e = matrix.setdefault(ref_str,{})
        e[hypo_str] = e.get(hypo_str, 0) + 1
        
        "true positive and false negative (considering reference)"
        t = 'tp' if ref_str == hypo_str else 'fn'
        add_count(ref_str, 1, t)
        
        "false positive (considering hypos)"
        if hypo_str != ref_str:
            add_count(hypo_str, 1, 'fp')
    
    "compute statistics (macro and micro)" 
    detailed_result = {}
    global_result = {}
    for group in macros:
        detailed_result[group] = {}
        #global_result[group] = {}
        acum_tp = acum_fp = acum_fn = 0
        macro_R = macro_F1 = macro_P = macro_R = macro_F1 = F1_of_macro= 0.0
        for label in macros[group]:
            tp, fp, fn = macros[group][label]['tp'], macros[group][label]['fp'], macros[group][label]['fn']
            acum_tp += tp
            acum_fp += fp
            acum_fn += fn
            P = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
            R = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
            F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
            macro_P += P
            macro_R += R
            macro_F1 += ((2*P*R) / (P+R) if (P+R) > 0 else 0.0)
            detailed_result[group][label] = {'tp':tp, 'fp':fp, 'fn':fn, 'P':P, 'R':R, 'F1':F1}            
        micro_P = float(acum_tp) / (acum_tp + acum_fp) if (acum_tp + acum_fp) > 0 else 0.0
        micro_R = float(acum_tp) / (acum_tp + acum_fn) if (acum_tp + acum_fn) > 0 else 0.0
        micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R) if micro_P + micro_R > 0 else 0.0
        macro_P /= len(macros[group])
        macro_R /= len(macros[group])
        macro_F1 /= len(macros[group])
        F1_of_macro = (2 * (macro_P * macro_R) / (macro_P + macro_R)) if macro_P + macro_R > 0 else 0.0
        global_result[group] = {
                                'tp':acum_tp, 'fp':acum_fp, 'fn':acum_fn,
                                'micro_P':micro_P, 'micro_R':micro_R, 'micro_F1': micro_F1,
                                'macro_P':macro_P, 'macro_R':macro_R, 'macro_F1': macro_F1, 'F1_of_macro':F1_of_macro
                                }
   
    "compute accuracy'"
    accuracy = {}    
    for label in set(list(hits.keys())+list(errors.keys())):
        accuracy[label] = {'hits':hits.get(label, 0),
                           'errors':errors.get(label, 0),
                           'accuracy':float(hits.get(label,0))/(hits.get(label,0)+errors.get(label,0))
                           }
        
    return (global_result, detailed_result, accuracy, matrix, {}, {})





def evaluate_results_multilabel(lref, lhypo):
    """lref and lhypo are lists having the same size

    Each position is a sample.
    ',' and '+' separate mandatory classes
    '/' separates optional classes

    Samples with '/' in the reference are evaluated in a in a special way.
        They are considered to have just one label 'amb' (ambiguous)

    labels 'other' and 'und' are considered equivalent
    """

    #TODO: some features could be configured using parameters


    AMB_LABEL = '<amb>'

    "to record macro-averaging"
    macros = {}
    def add_count(label, group, t):
        m0 = macros.setdefault(str(group), {}).setdefault(label, {'tp':0, 'fp':0, 'fn':0}) # tp, fp, fn
        m0[t] += 1
        m1 = macros.setdefault('<GLOBAL>', {}).setdefault(label, {'tp':0, 'fp':0, 'fn':0}) # tp, fp, fn
        m1[t] += 1


    hits = {} # hits
    errors = {} # errors
    matrix = {} # confusion matrix for non ambigous samples with one label in the reference
    matrix2 = {} # confusion matrix for non ambigous samples with more than one label in the reference
    matrix_amb = {} # confusion matrix for ambigous samples
    "count hits, true positive, false positive, and false negative"
    "compute agglutinative matrix of confusion"
    for i in range(len(lref)):
        amb = '/' in lref[i]
        ref_labs = frozenset(lref[i].replace('other', 'und').replace(',', '+').replace('/','+').split('+'))
        hypo_labs = frozenset(lhypo[i].replace('other', 'und').replace(',', '+').split('+'))
        ref_str = ('/' if amb else '+').join(sorted(ref_labs))
        hypo_str = '+'.join(x for x in hypo_labs)

        "count hit and errors"
        if not amb:
            if ref_labs == hypo_labs:
                hits[ref_str] = hits.get(ref_str, 0) + 1
            else:
                errors[ref_str] = errors.get(ref_str, 0) + 1
        elif hypo_labs.issubset(ref_labs):
            hits[ref_str] = hits.get(AMB_LABEL, 0) + 1
        else:
            errors[ref_str] = errors.get(AMB_LABEL, 0) + 1

        "add to confusion matrix"
        if amb:
            e = matrix_amb.setdefault(ref_str,{})
        elif len(ref_labs) == 1:
            e = matrix.setdefault(ref_str,{})
        else:
            e = matrix2.setdefault(ref_str,{})
        e[hypo_str] = e.get(hypo_str, 0) + 1

        "true positive and false negative (considering reference)"
        if amb:
            "ambigous sample, any label in ref_labs is valid"
            t = 'tp' if len(ref_labs.intersection(hypo_labs)) > 0 else 'fn'
            add_count(AMB_LABEL, '<AMBIGOUS>', t)
        else:
            for label in ref_labs:
                t = 'tp' if label in hypo_labs else 'fn'
                add_count(label, len(ref_labs), t)

        "false positive (considering hypos)"
        grp = len(ref_labs) if not amb else '<AMBIGOUS>'
        for label in hypo_labs:
            if label not in ref_labs:
                add_count(label, grp, 'fp')

    "compute statistics (macro and micro)"
    detailed_result = {}
    global_result = {}
    for group in macros:
        detailed_result[group] = {}
        #global_result[group] = {}
        acum_tp = acum_fp = acum_fn = 0
        macro_R = macro_F1 = macro_P = macro_R = macro_F1 = F1_of_macro= 0.0
        for label in macros[group]:
            tp, fp, fn = macros[group][label]['tp'], macros[group][label]['fp'], macros[group][label]['fn']
            acum_tp += tp
            acum_fp += fp
            acum_fn += fn
            P = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
            R = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
            F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
            macro_P += P
            macro_R += R
            macro_F1 += ((2*P*R) / (P+R) if (P+R) > 0 else 0.0)
            detailed_result[group][label] = {'tp':tp, 'fp':fp, 'fn':fn, 'P':P, 'R':R, 'F1':F1}
        micro_P = float(acum_tp) / (acum_tp + acum_fp) if (acum_tp + acum_fp) > 0 else 0.0
        micro_R = float(acum_tp) / (acum_tp + acum_fn) if (acum_tp + acum_fn) > 0 else 0.0
        micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R) if micro_P + micro_R > 0 else 0.0
        macro_P /= len(macros[group])
        macro_R /= len(macros[group])
        macro_F1 /= len(macros[group])
        F1_of_macro = (2 * (macro_P * macro_R) / (macro_P + macro_R)) if macro_P + macro_R > 0 else 0.0
        global_result[group] = {
                                'tp':acum_tp, 'fp':acum_fp, 'fn':acum_fn,
                                'micro_P':micro_P, 'micro_R':micro_R, 'micro_F1': micro_F1,
                                'macro_P':macro_P, 'macro_R':macro_R, 'macro_F1': macro_F1, 'F1_of_macro':F1_of_macro
                                }

    "compute accuracy'"
    accuracy = {}
    for label in set(list(hits.keys())+list(errors.keys())):
        accuracy[label] = {'hits':hits.get(label, 0),
                           'errors':errors.get(label, 0),
                           'accuracy':float(hits.get(label,0))/(hits.get(label,0)+errors.get(label,0))
                           }

    return (global_result, detailed_result, accuracy, matrix, matrix2, matrix_amb)



# ####################################
# ### ORIGINAL DEL TWEETLID_2014 #####
# ####################################
# ############# E N D ################
# ####################################



class MicroMacroEvaluator:
    def __init__(self):
        pass

    def evaluate_micro_and_macro(self, lreal, lhypo, allow_repeated=False):
        assert len(lreal) == len(lhypo)
        if isinstance(lreal, dict):
            lid = sorted(lreal.keys())
            lid2 = sorted(lhypo.keys())
            assert lid == lid2
            lreal = [lreal[x] for x in lid]
            lhypo = [lhypo[x] for x in lid]
            
        lreal = [x if isinstance(x, (list, tuple)) else x.replace(',','|').split('|') for x in lreal]
        lhypo = [x if isinstance(x, (list, tuple)) else x.replace(',','|').split('|') for x in lhypo]
        macro = {}
        for i in range(len(lreal)):
            refe = lreal[i]
            hypo = lhypo[i]
            if allow_repeated == False:
                refe = sorted(set(refe))
                hypo = sorted(set(hypo))
            for tag in hypo:
                m = macro.setdefault(tag, {'tp':0, 'fp':0, 'fn':0}) # tp, fp, fn
                if tag in refe:
                    m['tp'] +=1
                    refe.remove(tag) # remove the tag from the reference 
                else:#if tag not in refe:
                    m['fp'] += 1
            for tag in refe:
                m = macro.setdefault(tag, {'tp':0, 'fp':0, 'fn':0}) # tp, fp, fn
                m['fn'] += 1
        tp = fp = fn = 0
        macro_P = macro_R = macro_F1 = 0.0
        macro_detail = {}
        for k in macro:
            tag_tp, tag_fp, tag_fn = macro[k]['tp'], macro[k]['fp'], macro[k]['fn']
            tp += tag_tp
            fp += tag_fp
            fn += tag_fn
            tag_P = float(tag_tp) / (tag_tp + tag_fp) if (tag_tp + tag_fp) > 0 else 0.0
            tag_R = float(tag_tp) / (tag_tp + tag_fn) if (tag_tp + tag_fn) > 0 else 0.0
            tag_F1 = 2 * (tag_P * tag_R) / (tag_P + tag_R) if tag_P + tag_R > 0 else 0.0
            macro_P += tag_P
            macro_R += tag_R
            macro_F1 += 2 * (tag_P * tag_R) / (tag_P + tag_R) if tag_P + tag_R > 0 else 0.0
            macro_detail[k] = (tag_tp, tag_fp, tag_fn, tag_P, tag_R, tag_F1)
        # macro measures
        macro_P /= len(macro)
        macro_R /= len(macro)
        macro_F1 /= len(macro)
        macro_F1_alter = 2 * (macro_P * macro_R) / (macro_P + macro_R) if macro_P + macro_R > 0 else 0.0
        # micro measures
        micro_P = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
        micro_R = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_F1 = 2 * (micro_P * micro_R) / (micro_P + micro_R) if micro_P + micro_R > 0 else 0.0
        # all together
        micro_results = (micro_P, micro_R, micro_F1)
        macro_results = (macro_P, macro_R, macro_F1, macro_F1_alter)
        return (micro_results, macro_results, macro_detail)

    def show_results(self, result):
        micro_result, macro_result, macro_detail = result
        print ("MICRO:")
        print ("------")
        print (micro_result)
        print ("MACRO:")
        print ("------")
        print (macro_result)
        print ("DETAIL:")
        print ("-------")
        for tag in sorted(macro_detail):
            print ("%s\t%s" % (tag, macro_detail[tag]))
        #ev.show_results(result)

class TweetSplitter:

    def __init__(self):
        pass

    def next_stop(self, tweet, ini = 0):
        i = ini
        N = len(tweet)
        while i < N:
            POS = tweet[i]['POS']
            if POS in ('Fp'):
                return i
            i += 1
        return i-1

    def previous_stop(self, tweet, ini = None):
        if ini == None:
            ini = len(tweet)
        i = ini
        while i > 0:
            POS = tweet[i]['POS']
            if POS in ('Fp'):
                return i
            i -= 1
        return i

    ####################################
    ## SPLITs
    ## a partir d'un tweet i una llista d'entitats (inici, long, i nom) determina l'abast de cada entitat
    ##


    def split_tweet(self, tweet, entities_list, stype=0, **argv):
        if not isinstance(tweet, list):
            tweet = [{'word':x, 'POS':x if x != '.' else 'Fp'} for x in tweet.split()]
        if stype == 1:
            r = self.split_run1(tweet, entities_list, **argv)
        elif stype == 2:
            r = self.split_run2(tweet, entities_list, **argv)
        elif stype == 3:
            r = self.split_run3(tweet, entities_list, **argv)
        else:
            r = self.split_run0(tweet, entities_list, **argv)

        return r

    def split_run0(self, tweet, entities_list):
        "pensada de Ferran"
        "number of entities in the tweet"
        "Note: two different entities can reference to the same 'real' entity in the tweet (ie #ppsoe)"
        "all_tweets[id][1] -> list with the lemma of the tweet"
            
      
        #num = len(entities_list)
        tweet_len = len(tweet)
        result = []
        
        #for entity_pos, entity_name in sorted([(tweet[1].index(x), x) for x in set(entities_list)]):
        #for entity_pos, entity_len, entity_name in sorted([(tweet[1].index(x), x) for x in set(entities_list)]):
        for entity_pos, entity_len, entity_name in entities_list:
            #result.append((entity_name, self.previous_stop(tweet, entity_pos),  self.next_stop(tweet, entity_pos) + 1)) # RUN30
            #result.append((entity_name, 0,  self.next_stop(tweet, entity_pos) + 1))  # RUN31
            #result.append((entity_name, self.previous_stop(tweet, entity_pos),  tweet_len)) # RUN32
            result.append((entity_name, 0,  tweet_len)) # RUN33 -> el baseline
        return result

    def split_run1(self, tweet, entities_list):
        "pensada de Ferran per al tass2013(run1)"
        THR_BEGIN = 2
        THR_END = 4
        THR_TOGETHER = 2
        "number of entities in the tweet"
        "Note: two different entities can reference to the same 'real' entity in the tweet (ie #ppsoe)"
        "all_tweets[id][1] -> list with the lemma of the tweet"
       
        num = len(entities_list)
        tweet_len = len(tweet)
        #pos, lng, name = entities_list#([(tweet[1].index(x), x) for x in set(entities_list)])
        #result = []
        if num == 1:
            "tweets with one entity"
            pos, lng, name = entities_list[0]
            fi = pos - lng + 1
            
            if pos >= THR_BEGIN and fi < tweet_len - THR_END:
                """1.1
                entity is in the middle of the tweet"""
                subtweet_ini = pos
                subtweet_end = tweet_len
            elif pos < THR_BEGIN:
                """1.2
                entity at the beginning of the tweet"""
                subtweet_ini = 0
                subtweet_end = self.next_stop(tweet, fi) + 1
            else:
                """1.3
                entity at the end of the tweet"""
                subtweet_ini = 0
                subtweet_end = tweet_len
            result = [(name, subtweet_ini, subtweet_end)]
        else:
            "tweets with more than one entity"
            "a) sort entity list"
            pos_primer, lng_primer, name_primer = entities_list[0]
            pos_ultim, lng_ultim, name_ultim = entities_list[-1]
            fi_primer = pos_primer - lng_primer + 1
            fi_ultim = pos_ultim - lng_ultim + 1
#            primer = pos[0][0]
#            ultim = pos[-1][0]
            diff = pos_ultim - (pos_primer + lng_primer -1)
            
            if num == 2:
                if pos_primer < THR_BEGIN:
                    "la primera al principi"
                    if fi_ultim < tweet_len - THR_END and diff > THR_TOGETHER:
                        """2.1
                        primera al principi
                        l'ultima (segona) al mig"""
                        result = [
                                        (name_primer, 0, pos_ultim),
                                        (name_ultim, pos_ultim, tweet_len)  
                                                        ]
                    else:
                        """2.2, 2.3
                        primera al principi
                        l'ultima (segona) pegada a ella o al final"""
                        result = [
                                        (name_primer, 0, tweet_len),
                                        (name_ultim, 0, tweet_len)                                                        
                                                        ]
                elif diff <= THR_TOGETHER and fi_ultim >= tweet_len - THR_END:
                    """2.4
                    les 2 entitats juntes al final"""
                    result = [
                                    (name_primer, 0, tweet_len),
                                    (name_ultim, 0, tweet_len)
                                                    ]
                else:
                    "la primera pel mig"
                    if fi_ultim >= tweet_len - THR_END:
                        """2.5
                        la primera pel mig
                        l'ultima (segona) al final"""
                        result = [
                                        (name_primer, pos_primer, pos_ultim),
                                        (name_ultim, 0, tweet_len)
                                                        ]
                    else:
                        """2.6, 2.7
                        la primera pel mig
                        l'ultima (segona) pel mig tambe"""
                        result = [
                                        (name_primer, 0, pos_ultim),
                                        (name_ultim, pos_ultim, tweet_len) 
                                                        ]
            else:
                "more of 2 entities in the tweet"
                result = [(entity_name, 0, tweet_len) for entity_name in [z for x, y, z in entities_list]]
        return result

    def split_run2(self, tweet, entities_list):
        "pensada per al tass2013(run2)"
        THR_BEGIN = 2
        THR_END = 4
        THR_TOGETHER = 2
        "number of entities in the tweet"
        "Note: two different entities can reference to the same 'real' entity in the tweet (ie #ppsoe)"
        "all_tweets[id][1] -> list with the lemma of the tweet"
       
        num = len(entities_list)
        #print (num)
        tweet_len = len(tweet)
        #pos = sorted([(tweet[1].index(x), x) for x in set(entities_list)])
        #result = []
        if num == 1:

            entity_pos, entity_lng, entity_name = entities_list[0]
            entity_end = entity_pos - entity_lng

            subtweet_ini = 0
            subtweet_end = tweet_len #RUN20
            #subtweet_end = self.next_stop(tweet, entity_pos) + 1 # RUN21
            result = [(entity_name, subtweet_ini, subtweet_end)]
        else:
            "tweets with more than one entity"
            "a) sort entity list"
#            primer = pos[0][0]
#            ultim = pos[-1][0]

            #print (entities_list)
            pos_primer, lng_primer, name_primer = entities_list[0]
            pos_ultim, lng_ultim, name_ultim = entities_list[-1]
            fi_primer = pos_primer + lng_primer - 1
            fi_ultim = pos_ultim + lng_ultim - 1
            diff = pos_ultim - (pos_primer + lng_primer -1)

            #print (pos_primer, lng_primer, fi_primer, name_primer)
            #print (pos_ultim, lng_ultim, fi_ultim, name_ultim)
            #print (diff)
            #print (tweet_len)
            
            if num == 2:
                if pos_primer < THR_BEGIN:
                    "la primera al principi"
                    if fi_ultim < tweet_len - THR_END and diff > THR_TOGETHER:
                        """2.1
                        primera al principi
                        l'ultima (segona) al mig"""
                        result = [
                                        (name_primer, 0, pos_ultim),
                                        (name_ultim, pos_ultim, tweet_len)  
                                                        ]
                    else:
                        """2.2, 2.3
                        primera al principi
                        l'ultima (segona) pegada a ella o al final"""
                        result = [
                                        (name_primer, 0, tweet_len),
                                        (name_ultim, 0, tweet_len)                                                        
                                                        ]
                elif diff <= THR_TOGETHER and fi_ultim >= tweet_len - THR_END:
                    """2.4
                    les 2 entitats juntes al final"""
                    result = [
                                    (name_primer, 0, tweet_len),
                                    (name_ultim, 0, tweet_len)
                                                    ]
                else:
                    "la primera pel mig"
                    if fi_ultim >= tweet_len - THR_END:
                        """2.5
                        la primera pel mig
                        l'ultima (segona) al final"""
                        result = [
                                        (name_primer, pos_primer, pos_ultim),
                                        (name_ultim, 0, tweet_len)                                                        
                                                        ]
                    else:
                        """2.6, 2.7
                        la primera pel mig
                        l'ultima (segona) pel mig tambe"""
                        result = [
                                        (name_primer, 0, pos_ultim),
                                        (name_ultim, pos_ultim, tweet_len) 
                                                        ]
            else:
                "more of 2 entities in the tweet"
                result = [(entity_name, 0, tweet_len) for entity_name in [z for x, y, z in entities_list]]
        return result

        
    def split_run3(self, tweet, entities_list, left_window=None, right_window=None):
        "una finestra de 10 a esquerra i dreta"

        if left_window == None:
            left_window = 7
        if right_window == None:
            right_window = 7
      
        #num = len(entities_list)
        tweet_len = len(tweet)
        result = []
        
        for entity_pos, entity_len, entity_name in entities_list:
            #result.append((entity_name, self.previous_stop(tweet, entity_pos),  self.next_stop(tweet, entity_pos) + 1)) # RUN30
            #result.append((entity_name, 0,  self.next_stop(tweet, entity_pos) + 1))  # RUN31
            #result.append((entity_name, self.previous_stop(tweet, entity_pos),  tweet_len)) # RUN32
            result.append((entity_name, max(0, entity_pos-left_window),  min(tweet_len, entity_pos+entity_len+right_window))) # RUN33 -> el baseline
        return result




class PolaritySum:
    "just one dictionary"

    def __init__(self, file_dic_list=None, cols=None, dnames=None, min_len=1):
        if file_dic_list != None:
            self.load_dictionaries(file_dic_list, cols=cols, dnames=dnames, min_len=min_len)

    def load_dictionaries(self, file_dic_list, cols=None, dnames=None, min_len = 1):
        dic_files = file_dic_list
        dics = []
        dic_types = []
        if cols == None or len(cols) != len(dic_files):
            dic_cols = [1]*len(dic_files)
        else:
            dic_cols = cols

        if dnames == None or len(dnames) != len(dic_files):
            dnames = ["DIC%d" % i for i in range(len(dic_files))]
        for i in range(len(dic_files)):
            pol_file = dic_files[i]
            pol_dic = {}
            for line in open(pol_file):
                #TODO: more than one word rsplit!
                spt = line.strip().rsplit(maxsplit=dic_cols[i])
                phrase = spt[0]
                pol_dic[phrase] = [float(x) for x in spt[1:]]
            dics.append(pol_dic)
        self.dic_files = dic_files
        self.dics = dics
        self.dic_names = dnames
        self.dic_cols = dic_cols
        self.dic_types = dic_types


    def fit_transform(self, sample_list):
        return self.extract_features(sample_list, use_numpy=True)


    def transform(self, sample_list):
        return self.extract_features(sample_list, use_numpy=True)


    def extract_features(self, sample_list, use_numpy=True):

        rows = len(sample_list)
        columns = sum(x for x in self.dic_cols)
        matrix = numpy.zeros((rows, columns))# , dtype=float, order='C')
        for i in range(rows):
            sample = sample_list[i]
            #TODO: consider phrases not just words
            spt = sample.split()
            rs = []
            for j in range(len(self.dic_cols)):
                rd = numpy.zeros((self.dic_cols[j]))#rows, columns))# , dtype=float, order='C')
                d = self.dics[j]
                for word in spt:
                    #print (word)
                    v = d.get(word, [0]*self.dic_cols[j])
                    #print (v)
                    rd += v
                    #print (rd)
                rs.append(rd)
            row = numpy.hstack(rs)
            matrix[i] = row
        return matrix


class PolarityCount:
    def __init__(self, file_dic_list=None, norm=0, dnames=None, purity_level=0, all_one=False, first=None, min_len=1):
        self.norm = norm
        if file_dic_list != None:
            self.load_dictionaries(file_dic_list, dnames=dnames, purity_level=purity_level, all_one=False, min_len=min_len)

    def load_dictionaries(self, file_dic_list, dnames=None, purity_level=0, all_one=False, min_len = 1, first = None, clean_mode=0):
        """
          purity_level:
              0: no purity is applied
              1: purity among all classes of the same dic
              2: purity among all classes of all dic
        """

        dic_files = file_dic_list
        dics = []
        dic_types = []
        if dnames == None or len(dnames) != len(dic_files):
            if all_one == False:
                dnames = ["DIC%d" % i for i in range(len(dic_files))]
            else:
                dnames = ["DIC" % i for i in range(len(dic_files))]
        if all_one:
            pol_dic = {}
        for i in range(len(dic_files)):
            pol_file = dic_files[i]
            if not all_one:
                pol_dic = {}
            cnt = 0
            for line in open(pol_file):
                spt = line.strip().split()
                if len(spt) == 2:
                    word, cname = spt
                else:
                    word = spt[0]
                    cname = dnames[i]
                word =  clean_tweet(word, mode=clean_mode)
                if len(word) >= min_len:
                    pol_dic.setdefault(cname, set()).add(word)
                    cnt += 1
                if first != None and cnt == first:
                    break
            if purity_level > 0:
                #ESTA MAL!!!!!
                for k1 in pol_dic:
                    for k2 in pol_dic:
                        if k1 != k2:
                            pol_dic[k1].difference_update(pol_dic[k2])
            if not all_one:
                dics.append(pol_dic)
                dic_types.append(sorted(pol_dic.keys()))
        if all_one:
            dics.append(pol_dic)
            dic_types.append(sorted(pol_dic.keys()))
        if purity_level == 2:
            auxi_dics = []
            for dic1 in dics:
                auxi_dic = {}
                for k1 in dic1:
                    auxi_dic[k1] = set()
                    auxi_dic[k1].update(dic1[k1])
                    #hiper_set.update(dic1[k1])
                    for dic2 in dics:
                        for k2 in dic2:
                            if dic1 != dic2 or k1 != k2:
                                auxi_dic[k1].difference_update(dic2[k2])
                auxi_dics.append(auxi_dic)
            dics = auxi_dics

        self.purity=purity_level
        self.clean_mode=clean_mode
        self.min_len=min_len
        self.first=first
        self.dic_files = dic_files
        self.dics = dics
        self.dic_types = dic_types

        if False:
            print (self.dic_files)
            print (self.dic_types)
            print (self.dics)

            print ('='*20)
            for x in self.dics:
                print (len(x), type(x))
                print (x)
                for y in x:
                    print ('\t',y,len(x[y]),type(x[y]))
        #print (self.dic_types)


    def fit_transform(self, sample_list):
        return self.extract_features(sample_list, use_numpy=True)

    def transform(self, sample_list):
        return self.extract_features(sample_list, use_numpy=True)

    def extract_features(self, sample_list, use_numpy=True):
        matrix = []
        for sample in sample_list:
            spt = sample.split()
            N = len(spt)
            global_cnts = []
            for i in range(len(self.dics)):
                cnts = []
                for pol in self.dic_types[i]:
                    cnt = sum(1 if w in self.dics[i][pol] else 0 for w in spt)
                    if self.norm==1:
                        cnts.append(cnt/N)
                    else:
                        cnts.append(cnt)
                global_cnts.extend(cnts)
            matrix.append(global_cnts)
        if use_numpy:
            matrix = numpy.array(matrix)
        return matrix

    def __str__(self):
        r = 'PC'+str(self.first)+':norm:' + str(self.norm) + '-pur:' + str(self.purity) + "-min_len:"+str(self.min_len) + "-clean:" + str(self.clean_mode)
        return r

class C2VVectorizer:

    def __init__(self, size=300, min_count=50, window=15, sample=1e-3, N=3, unigram=True):

        self.nfeatures = size
        self.min_count=min_count
        self.window=window
        self.sample=sample
        self.N=N
        self.unigram=unigram

    def extract_grams(self, samples):
        samples = [extract_nchars(s.lower(), self.N, self.unigram) for s in samples]

        return samples

    def fit_transform(self, samples):

        samples = self.extract_grams(samples)

        #print (samples[0])

        # Entrenar el modelo
        self.model = word2vec.Word2Vec(samples, workers=4,
                          size=self.nfeatures, min_count = self.min_count,
                          window = self.window, sample = self.sample)

        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        self.model.init_sims(replace=True)

        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        self.index2word_set = set(self.model.index2word)

        r = self.getAvgFeatureVecs(samples)

        return self.getAvgFeatureVecs(samples)

    def transform(self, samples):
        samples = self.extract_grams(samples)
        return self.getAvgFeatureVecs(samples)


    def makeFeatureVec(self, words):
        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        featureVec = np.zeros((self.nfeatures,),dtype="float32")
        #
        nwords = 0.
        #
        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in self.index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, self.model[word])
        #
        # Divide the result by the number of words to get the average
        featureVec = np.divide(featureVec,nwords)
        return featureVec


    def getAvgFeatureVecs(self, samples):
        # Given a set of samples (each one a list of words), calculate
        # the average feature vector for each one and return a 2D numpy array
        #
        # Initialize a counter
        counter = 0.
        #
        # Preallocate a 2D numpy array, for speed
        reviewFeatureVecs = np.zeros((len(samples),self.nfeatures),dtype="float32")
        #
        # Loop through the reviews
        for sample in samples:
            # Print a status message every 1000th review
            #if counter%1000. == 0.:
             #   print ("Review %d of %d" % (counter, len(reviews)))

            # Call the function (defined above) that makes average feature vectors
            reviewFeatureVecs[counter] = self.makeFeatureVec(sample)

            # Increment the counter
            counter = counter + 1.
    #TODO: passat a sparse del scipy per compatibilitat
    #    return reviewFeatureVecs
        return scipy.sparse.csr.csr_matrix(reviewFeatureVecs)







class W2VVectorizer:

    def __init__(self, size=300, min_count=50, window=15, sample=1e-3):

        self.nfeatures = size
        self.min_count=min_count
        self.window=window
        self.sample=sample


    def tokenize(self, samples):
        return samples
        #return [x.lower().split() for x in samples]


    def fit_transform(self, samples):

        samples = self.tokenize(samples)

        #print (samples[0])

        # Entrenar el modelo
        self.model = word2vec.Word2Vec(samples, workers=4,
                          size=self.nfeatures, min_count = self.min_count,
                          window = self.window, sample = self.sample)

        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        self.model.init_sims(replace=True)

        # Index2word is a list that contains the names of the words in
        # the model's vocabulary. Convert it to a set, for speed
        self.index2word_set = set(self.model.index2word)
        #r = self.getAvgFeatureVecs(samples)

        return self.getAvgFeatureVecs(samples)

    def transform(self, samples):
        token_list = self.tokenize(samples)
        return self.getAvgFeatureVecs(token_list)


    def makeFeatureVec(self, words):
        # Function to average all of the word vectors in a given
        # paragraph
        #
        # Pre-initialize an empty numpy array (for speed)
        featureVec = np.zeros((self.nfeatures,),dtype="float32")
        #
        nwords = 0.
        #
        # Loop over each word in the review and, if it is in the model's
        # vocaublary, add its feature vector to the total
        for word in words:
            if word in self.index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, self.model[word])
        #
        # Divide the result by the number of words to get the average
        featureVec = np.divide(featureVec,nwords)
        return featureVec


    def getAvgFeatureVecs(self, samples):
        # Given a set of samples (each one a list of words), calculate
        # the average feature vector for each one and return a 2D numpy array
        #
        # Initialize a counter
        counter = 0.
        #
        # Preallocate a 2D numpy array, for speed
        reviewFeatureVecs = np.zeros((len(samples),self.nfeatures),dtype="float32")
        #
        # Loop through the reviews
        for sample in samples:
            # Print a status message every 1000th review
            #if counter%1000. == 0.:
             #   print ("Review %d of %d" % (counter, len(reviews)))

            # Call the function (defined above) that makes average feature vectors
            reviewFeatureVecs[counter] = self.makeFeatureVec(sample)

            # Increment the counter
            counter = counter + 1.
    #TODO: passat a sparse del scipy per compatibilitat
    #    return reviewFeatureVecs
        return scipy.sparse.csr.csr_matrix(reviewFeatureVecs)










import os.path
class WikipediaReader:
    PATTERN_BEGIN = "<doc"
    PATTERN_END = "</doc>"
    def __init__(self, base_dir=None):
        self.base_dir = base_dir


    def next_article(self):
        root_dir = self.base_dir
        for dir_name, subdir_list, filelist in os.walk(root_dir):
            for fname in filelist:
                filename = os.path.join(os.path.join(self.base_dir, dir_name), fname)
                with open(filename) as fh:
                    text = fh.read()
                    begin=0
                    end = text.find(self.PATTERN_END, begin)
                    while end != -1:
                        title_beg = text.find(self.PATTERN_BEGIN, begin)
                        title_end = text.find('\n', title_beg)
                        title = text[title_beg:title_end]
                        piece_of_new = text[title_end+1:end]
                        begin = end + len(self.PATTERN_END)
                        end = text.find(self.PATTERN_END, begin)
                        yield piece_of_new










































class FreelingInterface:
    FREELINGDIR = "/usr/local/share/freeling/"

    def __init__(self, lang, local_dir=None, freeling_dir=None, NER=True):

        if local_dir == 'ENV':
            try:
                local_dir = os.path.join(os.path.join(os.environ['LOCALFREELINGSHARE'], "tweets"), "es")
            except KeyError:
                local_dir = None

        if freeling_dir == None:
            #freeling_dir = FreelingInterface.FREELINGDIR
            freeling_dir = self.FREELINGDIR
        freeling_dir = os.path.join(freeling_dir, lang)
        if local_dir == None:
            local_dir = freeling_dir
        
        freeling.util_init_locale("default")

        # create options set for maco analyzer.
        """
        set_active_modules  (
                bool    umap,
                bool    suf, 
                bool    mw,  
                bool    num, # numeració
                bool    pun, # puntuació
                bool    dat, # dates
                bool    qt,  # quantitats
                bool    dic, # diccionari
                bool    prb, # probabilitats
                bool    ner, # NER
                bool    orto # correcció ortogràfica
            )   
        
        set_data_files  (
                const std::wstring &    usr,
                const std::wstring &    loc,
                const std::wstring &    qty,
                const std::wstring &    suf,
                const std::wstring &    prb,
                const std::wstring &    dic,
                const std::wstring &    nps,
                const std::wstring &    pun,
                const std::wstring &    corr 
            )
        """
        op= freeling.maco_options(lang)
        op.set_active_modules(1,1,1,1,1,1,1,1,1,NER,0)
        op.set_data_files(
            os.path.join(local_dir, "es-twit-map.dat"),
            os.path.join(freeling_dir, "locucions.dat"),
            os.path.join(freeling_dir, "quantities.dat"),
            os.path.join(freeling_dir, "afixos.dat"),
            os.path.join(freeling_dir, "probabilitats.dat"),
            os.path.join(freeling_dir, "dicc.src"),
            os.path.join(freeling_dir, "np.dat") if NER else '',
            os.path.join(freeling_dir, "../common/punct.dat")
            
            )

        # create analyzers
        self.tk=freeling.tokenizer(os.path.join(local_dir, "tokenizer.dat"))
        self.sp=freeling.splitter(os.path.join(local_dir, "splitter.dat"))
        self.mf=freeling.maco(op)
        self.tg=freeling.hmm_tagger(os.path.join(freeling_dir, "tagger.dat"), 1, 2)

    def freeling_analyze(self, sentence):
        l = self.tk.tokenize(sentence)
        #ls = self.sp.split(l, 0)
        ls = self.sp.split(l, 1)
        ls = self.mf.analyze(ls)
        result = []
        for s in ls :
           ws = s.get_words()
           for w in ws :
                result.append({'word': w.get_form(), 'lemma':w.get_lemma(), 'POS':w.get_tag()})
        return result

    def full_analyze(self, sentence, arroba=False, almo=False):
        t = ' '.join(twokenize_ES.tokenize(sentence))
        c = emoticons_ES.analyze_tweet(t, arroba=arroba, almo=almo)
        return self.freeling_analyze(c)


def clean_tweet(raw_tweet, mode=0):
    if mode == 0:
        return raw_tweet
    elif mode == 1:
        tokens = twokenize_ES.tokenize(raw_tweet)
        ctweet = emoticons_ES.analyze_tweet(' '.join(tokens), arroba=False, almo=False)
        ctweet = ctweet.replace('#', ' # ').replace('@', ' @ ')
        spt = [x for x in ctweet.split() if x.isalpha()]
        tweet = ' '.join(spt)
    elif mode == 2:
        tokens = twokenize_ES.tokenize(raw_tweet)
        ctweet = emoticons_ES.analyze_tweet(' '.join(tokens), arroba=False, almo=False)
        tweet = ctweet
    elif mode == 3:
        tokens = twokenize_ES.tokenize(raw_tweet)
        tweet = ' '.join(tokens)
    elif mode == 4:
        tokens = twokenize_ES.tokenize(raw_tweet)
        ctweet = emoticons_ES.analyze_tweet(' '.join(tokens), arroba=False, almo=False)
        spt = [x for x in ctweet.split() if x.isalpha()]
        tweet = ' '.join(spt)
    elif mode == 5:
        tokens = twokenize_ES.tokenize(raw_tweet)
        ctweet = emoticons_ES.analyze_tweet(' '.join(tokens), arroba=True, almo=True)
        spt = [x for x in ctweet.split() if x not in ('<HTTP>', '<ALMO>', '<ARROBA>', '.', ',', '...', ':',
                                                      '"', '“', '”', "'", ')', '(')]
        tweet = ' '.join(spt)
    elif mode == 14:
        tokens = twokenize_ES.tokenize(raw_tweet)
        if tokens[0] == 'RT':
            tokens = tokens[2:]
        ctweet = emoticons_ES.analyze_tweet(' '.join(tokens), arroba=False, almo=False)
        spt = [x for x in ctweet.split() if x.isalpha()]
        tweet = ' '.join(spt)
    elif mode == 15:
        tokens = twokenize_ES.tokenize(raw_tweet)
        if tokens[0] == 'RT':
            tokens = tokens[2:]
        ctweet = emoticons_ES.analyze_tweet(' '.join(tokens), arroba=False, almo=False)
        #elongates!!
        spt = [reduce_elongate(x, n=2) for x in ctweet.split() if x.isalpha()]
        tweet = ' '.join(spt)
    return tweet



class TwokenizeWrapper:
    def __init__(self):
        pass

    def just_tokenize(self, tweet):#, arroba=False, almo=False):
        t = ' '.join(twokenize_ES.tokenize(tweet))
        return t

    def just_emoticon_detection(seft, tweet, arroba=False, almo=False):
        t = emoticons_ES.analyze_tweet(tweet, arroba=arroba, almo=almo)
        return t

    def full_tokenize(self, tweet, arroba=False, almo=False):
        r = emoticons_ES.analyze_tweet(self.just_tokenize(tweet), arroba=arroba, almo=almo)
        return r

class PostprocessFreeling:
    def __init__(self):
        pass

    """
    -> parser.add_option("-s", "--select_POS1", dest="which_POS1", default=None, action="store", type="string", help="select the first char of the POS to be included")

    -> parser.add_option("-S", "--select_POS2", dest="which_POS2", default=None, action="store", type="string", help="select the first two chars of the POS to be included")

    "unification"
    "use list of polarity for word and lemma"

    -> parser.add_option("-N", "--numbers", dest="UNION_NUMBERS", default=False, action="store_true", help="unification of all numbers")

    -> parser.add_option("-D", "--date", dest="UNION_DATE", default=False, action="store_true", help="unification of all date")

    -> parser.add_option("-F", "--punctuation", dest="UNION_PUNCTUATION", default=False, action="store_true", help="unification of punctuation marks of the same type")

    -> parser.add_option("-M", "--all_punctuation", dest="UNION_ALL_PUNCTUATION", default=False, action="store_true", help="unification of all punctuation marks")

    if POS[0] == "F" and options.UNION_PUNCTUATION:
        word = "<%s>" % POS
        lemma = "<%s>" % POS
    if POS[0] == "F" and options.UNION_ALL_PUNCTUATION:
        word = "<PM>"
        lemma = "<PM>"
    if POS[0] == "W" and options.UNION_DATE:
        word = "<%s>" % POS
        lemma = "<%s>" % POS
    if POS[0] == "Z" and options.UNION_NUMBERS:
        word = "<%s>" % POS
        lemma = "<%s>" % POS

    if options.which_POS1 or options.which_POS2:
        if POS[0] not in options.which_POS1 and POS[:2] not in options.which_POS2:
            return None, None, None, False

    """

    def postprocess(self, l ,join_number=True
                        ,join_date=True
                        ,join_punctuation=True
                        #,join_all_punctuation=True
                        ,which_POS1=None
                        ,which_POS2=None
                ):
        """"""

        "join POS"
        pattern = []
        if join_number: pattern.append('Z')
        if join_date: pattern.append('W')
        if join_punctuation: pattern.append('F')
        l = [w if w['POS'][0] not in pattern else
                                {'word':'<%s>'%w['POS'],
                                'lemma':'<%s>'%w['POS'],
                                'POS':w['POS']} for w in l]

        "filter words taking into account POS"
        if which_POS1 != None or which_POS2 != None:
            if which_POS1 == None: which_POS1 = []
            if which_POS2 == None: which_POS2 = []
            l = [w for w in l if w['POS'][0] in which_POS1 or w['POS'][:2] in which_POS2]

        return l

        
if __name__ == "__main__":

    dics = ["/home/lhurtado/corpora/Twitter/polarity_lists/davide/es/SWN/SentiWN_es_ambtabs.txt",
            "/home/lhurtado/corpora/Twitter/polarity_lists/davide/es/Whissell/whissell_es_ambtabs.txt",
            "/home/lhurtado/corpora/Twitter/polarity_lists/davide/es/polarity-AFINN.txt",
            "/home/lhurtado/corpora/Twitter/polarity_lists/davide/es/polarity-AFINN-2cols.txt"
            ]

    ps = PolaritySum(dics, cols=[2,3,1,2])
    samples = ["abandonar aborto yeso", "que viva españa"]
    m = ps.transform(samples)

    print (m.shape)
    print (m)
    #noseque("hola que tal nopuedomas")

    if False:
        fi = FreelingInterface("es", local_dir='ENV')

        s = "#RT, @pepe Sergio Ramos sale vaamos de mi padre jejeje!!!! :-) http://noseque.cat"

        #t = ' '.join(twokenize_ES.tokenize(s))
        #c = emoticons_ES.analyze_tweet(t, arroba=False, almo=True)
        #c = emoticons_ES.analyze_tweet(t, arroba=False, almo=False)
        #c = emoticons_ES.analyze_tweet(t, arroba=False, almo=False)
        r = fi.full_analyze(s)
        r = fi.freeling_analyze(s)

        #print (r)
        print (s)
        for w in r:
            print (w[0],w[1],w[2])

    if False:
        c = "hoollaaaa"
        print (c)
        print (reduce_elongate(c, n=2))
