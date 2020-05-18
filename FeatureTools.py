import LabelTools as lt
import TreeTools as tt
from DisSentTools import DisSentTools as dst

import dill
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from collections import defaultdict, Counter
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from time import time
from tqdm.auto import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import gensim
import torch

exp_path = '/data/work/data/reddit_cmv/experiments/'

def extend_matrix(matrix, parents, n, only_parents = False, parent_selection = None):
    entries_num = matrix.shape[0]
    if only_parents:
        new_matrix = np.empty(shape=(entries_num, 0))
    else:
        new_matrix = matrix
    if n > 1:
        features_num = matrix.shape[1]
        matrix = np.append(matrix, np.zeros(shape=(1, features_num)), axis=0)
        entries_num = matrix.shape[0]
        if only_parents:
            new_matrix = np.empty(shape=(entries_num, 0))
        else:
            new_matrix = matrix
        parents = np.append(parents, [-1])
        if parent_selection is not None:
            dist_matrix = new_matrix
        for i in range(n-1):
            if parent_selection == 'dist_first' or parent_selection == 'dist_prev':
                parent_matrix = dist_matrix - matrix[parents, :]
            else:
                parent_matrix = matrix[parents, :]
            if parent_selection == 'dist_prev':
                dist_matrix = matrix[parents, :]
            new_matrix = np.hstack((new_matrix, parent_matrix))
            parents = parents[parents]
        new_matrix = np.delete(new_matrix, -1, axis=0)
    return new_matrix


class ScalingAdder(BaseEstimator, TransformerMixin):
    def _create_scaler(self, scaler):
        if scaler == 'std':
            self._sc = StandardScaler()
        if scaler == 'minmax':
            self._sc = MinMaxScaler()
        if scaler == 'maxabs':
            self._sc = MaxAbsScaler()

    def __init__(self, scaler = None):
        self.scaler = scaler
        self._create_scaler(scaler)

    def set_params(self, scaler = None, **parameters):
        self.scaler = scaler
        self._create_scaler(scaler)
        return self

    def get_params(self, **kwargs):
        return {"scaler": self.scaler}

    def transform(self, X, **transform_params):
        if self.scaler is None:
            return X
        if (X.shape[1] > 1):
            return np.hstack((X[:, :1], self._sc.transform(X[:, 1:])))
        return np.hstack((X[:, :1], np.zeros(shape=(X.shape[0], 1))))

    def fit(self, X, y=None, **fit_params):
        if self.scaler is not None:
            if X.shape[1] > 1:
                self._sc.fit(X[:, 1:], y)
        return self


class DisAdder(BaseEstimator, TransformerMixin):
    @staticmethod
    def load_dis(dis_path = '/data/work/data/reddit_cmv/dissent_probas/all_trees.dispr'):
        with open(dis_path, 'rb') as f:
            DisAdder.probas = dill.load(f)

    @staticmethod
    def get_m(all_probas, lengths, agg_t, gr, lpad, rpad, threshold, maxlength):
        bigram = False
        full = False
        if gr == 'unigram':
            dim = 15
        if gr == 'bigram':
            dim = 225
            if lpad or rpad:
                dim += 30
            bigram = True
        if gr == 'full':
            dim = 15*(maxlength if maxlength > 0 else 64)
            full = True
        res = np.zeros(shape=(all_probas.shape[0], dim))
        for i, nprobas in enumerate(all_probas):
            max_l = min(maxlength, lengths[i]) if maxlength > 0 else lengths[i]
            resrow = np.zeros(dim)
            if max_l > 0:
                prev1 = -1
                for j, sprobas in enumerate(nprobas[:max_l]):
                    if np.max(sprobas) >= threshold:
                        seqfound = True
                        cur = np.argmax(sprobas)
                        idx = cur
                        if full:
                            idx += j*15
                        else:
                            if bigram:
                                if prev1 > -1:
                                    idx += prev1*15
                                else:
                                    if j == 0 and lpad:
                                        idx += 225
                                    else:
                                        seqfound = False
                        if seqfound:
                            resrow[idx] += 1
                        if j+1 == max_l and bigram and rpad:
                            resrow[240+cur] += 1
                    else:
                        cur = -1
                    prev1 = cur
                if agg_t == 'avg_binary':
                    resrow = resrow / max_l
                if agg_t == 'singular':
                    resrow[resrow > 1] = 1
                if agg_t == 'sum_binary':
                    resrow = resrow
            res[i] = resrow
        return res

    @staticmethod
    def prefit(data_prefix):
        df_path = exp_path + 'data/' + data_prefix + '.csv'
        df = pd.read_csv(df_path, encoding="utf-8")
        DisAdder.load_dis()
        parents = df['parent'].to_numpy().flatten()
        print('DisAdder PreFit: Begin.')
        m = np.zeros(shape=(df.shape[0], 64, 15))
        lengths = np.zeros(df.shape[0], dtype=int)
        for index, row in df.iterrows():
            tid = row['tree_id']
            nid = row['node_id']
            idx = DisAdder.probas[tid]['hasht'][nid]
            nprobas = DisAdder.probas[tid]['probas'][idx]
            max_l = DisAdder.probas[tid]['lengths'][idx]
            m[index] = nprobas
            lengths[index] = max_l
        fitdata = {'parents': parents, 'm': m, 'lengths': lengths}
        save_fitdata(data_prefix, 'DisAdder', fitdata)
        print('DisAdder PreFit: Done.')

    @staticmethod
    def loadfit(data_prefix):
        DisAdder.fitdata = load_fitdata(data_prefix, 'DisAdder')

    def __init__(self, disable=False, aggtype='singular', gr='unigram', lpad=True, rpad=True, threshold=0.0, maxlength=0, ngrams=1, ps = 'abs', prescale = False):
        self.ngrams = ngrams
        self.aggtype = aggtype
        self.gr = gr
        self.lpad = lpad
        self.rpad = rpad
        self.threshold = threshold
        self.maxlength = maxlength
        self.ps = ps
        self.disable = disable
        self.prescale = prescale

    @staticmethod
    def create_matrix(aggtype, gr, lpad, rpad, threshold, maxlength, ngrams, ps):
        my_ps = get_ps(ngrams, ps)
        fitdata = DisAdder.fitdata
        m = DisAdder.get_m(fitdata['m'], fitdata['lengths'], aggtype, gr, lpad, rpad, threshold, maxlength)
        matrix = extend_matrix(m, fitdata['parents'], ngrams, parent_selection=my_ps)
        return matrix

    def fit(self, X, y = None, **fit_params):
        if not self.disable:
            self.matrix = DisAdder.create_matrix(self.aggtype, self.gr, self.lpad, self.rpad, self.threshold, self.maxlength, self.ngrams, self.ps)
        if self.prescale:
            self.matrix = MinMaxScaler().fit_transform(self.matrix)
        return self

    def transform(self, X, **transform_params):
        if self.disable:
            return X
        # get the first column of X - it represents rows indexes
        rows = X[:,0].astype(int)
        matrix = np.hstack((X, self.matrix[rows, :]))
        return matrix

    def set_params(self, disable=False, aggtype='singular', gr='unigram', lpad=True, rpad=True, threshold=0.0, maxlength=0, ngrams=1, ps='abs', prescale=False, **parameters):
        self.ngrams = ngrams
        self.aggtype = aggtype
        self.gr = gr
        self.lpad = lpad
        self.rpad = rpad
        self.threshold = threshold
        self.maxlength = maxlength
        self.ps = ps
        self.disable = disable
        self.prescale = prescale
        return self

    def get_params(self, **kwargs):
        return {"disable": self.disable, "aggtype": self.aggtype, "gr": self.gr, "lpad" : self.lpad, "rpad": self.rpad, "threshold": self.threshold,
                "maxlength": self.maxlength, "ngrams": self.ngrams, "ps": self.ps, "prescale": self.prescale}

class GloveAdder(BaseEstimator, TransformerMixin):
    @staticmethod
    def load_glove(glove_path = '/data/work/data/reddit_cmv/glove/glove.42B.300d.txt'):
        dic_size = 1917495
        # 1917494 lines in Glove + 1 for zeros
        GloveAdder.dic = defaultdict(int)
        GloveAdder.index = list()
        GloveAdder.dim = 300

        GloveAdder.full_glove_matrix = np.empty(shape=(dic_size,300))
        GloveAdder.full_glove_matrix[0] = [0.0] * 300
        GloveAdder.index.append('')
        print('Loading Glove from: ', glove_path)
        with open(glove_path, 'r') as glove_file:
            for counter, line in tqdm(enumerate(glove_file)):
                split_line = line.split()
                vec = [float(val) for val in split_line[1:]]
                word = split_line[0]
                GloveAdder.index.append(word)
                GloveAdder.dic[word] = counter+1
                GloveAdder.full_glove_matrix[counter+1] = vec
        print('Glove loaded successfully.')

    @staticmethod
    def prefit(data_prefix, avg = ['mean', 'idf']):
        df_path = exp_path + 'data/' + data_prefix + '.csv'
        df = pd.read_csv(df_path, encoding="utf-8").fillna('')
        GloveAdder.load_glove()
        corpus = df['text']
        parents = df['parent'].to_numpy().flatten()
        vectorizer = TfidfVectorizer(preprocessor=lt.clean_text, tokenizer=str.split, smooth_idf=False)
        vectorizer.fit_transform(corpus)
        idfs = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        m = defaultdict()
        m['mean'] = np.empty(shape=(0,GloveAdder.dim))
        m['idf'] = np.empty(shape=(0,GloveAdder.dim))
        for text in df['text'].values:
            words_idx = [GloveAdder.dic[word] for word in lt.clean_text(text).split(' ') if GloveAdder.dic[word] != 0]
            idf_weights = [idfs[word] for word in lt.clean_text(text).split(' ') if GloveAdder.dic[word] != 0]
            if sum(idf_weights) > 0.0:
                mean_vec = np.mean(a=GloveAdder.full_glove_matrix[words_idx, :],axis=0)
                idf_vec = np.average(a=GloveAdder.full_glove_matrix[words_idx, :], axis=0, weights=idf_weights)
            else:
                mean_vec = idf_vec = [0.0] * GloveAdder.dim
            m['mean'] = np.append(m['mean'], [mean_vec], axis=0)
            m['idf'] = np.append(m['idf'], [idf_vec], axis=0)
        fitdata = {'parents': parents, 'm': m, 'avg': avg}
        save_fitdata(data_prefix, 'GloveAdder', fitdata)

    @staticmethod
    def loadfit(data_prefix):
        GloveAdder.fitdata = load_fitdata(data_prefix, 'GloveAdder')

    def __init__(self, disable=False, ngrams=1, avg='mean', ps='abs', prescale=False):
        self.ngrams = ngrams
        self.avg = avg
        self.ps = ps
        self.disable = disable
        self.prescale = prescale

    @staticmethod
    def create_matrix(avg, ngrams, ps):
        my_ps = get_ps(ngrams, ps)
        fitdata = GloveAdder.fitdata
        if avg not in fitdata['avg']:
            print('GloveAdder: bad avg [%d] parameter, no prefitted data. Using [%d]' % (avg, fitdata['avg'][0]))
            avg = fitdata['avg'][0]
        matrix = extend_matrix(fitdata['m'][avg], fitdata['parents'], ngrams, parent_selection=my_ps)
        return matrix

    def fit(self, X, y = None, **fit_params):
        if not self.disable:
            self.matrix = GloveAdder.create_matrix(self.avg, self.ngrams, self.ps)
            if self.prescale:
                self.matrix = MinMaxScaler().fit_transform(self.matrix)
        return self

    def transform(self, X, **transform_params):
        if self.disable:
            return X
        # get the first column of X - it represents rows indexes
        rows = X[:,0].astype(int)
        matrix = np.hstack((X, self.matrix[rows, :]))
        return matrix

    def set_params(self, disable = False, ngrams = 1, avg = 'mean', ps='abs', prescale=False, **parameters):
        self.disable = disable
        self.ngrams = ngrams
        self.avg = avg
        self.ps = ps
        self.prescale = prescale
        return self

    def get_params(self, **kwargs):
        return {"disable": self.disable, "ngrams": self.ngrams, "avg": self.avg, "ps": self.ps, "prescale": self.prescale}


class Doc2VecAdder(BaseEstimator, TransformerMixin):
    @staticmethod
    def load_doc2vec(doc2vec_path = '/data/work/data/reddit_cmv/doc2vec/'):
        Doc2VecAdder.models = defaultdict(dict)
        print('Loading doc2vec from: ', doc2vec_path)
        for epochs in [100, 500]:
            for dim in [100, 500, 1000]:
                path = doc2vec_path + f'd2v_{epochs}e_{dim}d.txt'
                Doc2VecAdder.models[epochs][dim] = gensim.models.doc2vec.Doc2Vec.load(path)
        Doc2VecAdder.index = {}
        with open(doc2vec_path + 'doc2vec_train.txt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line not in Doc2VecAdder.index:
                    Doc2VecAdder.index[line] = i
        print('Doc2Vec loaded.')

    @staticmethod
    def prefit(data_prefix, method=['fetch', 'infer'], e=[100, 500], d=[100, 500, 1000]):
        df_path = exp_path + 'data/' + data_prefix + '.csv'
        df = pd.read_csv(df_path, encoding="utf-8")
        Doc2VecAdder.load_doc2vec()
        parents = df['parent'].to_numpy().flatten()
        print('Doc2Vec PreFit: Begin.')
        ms = defaultdict(lambda: defaultdict())
        for dim in d:
            for epochs in e:
                print(f'\ndim={dim},epochs={epochs}\n')
                model = Doc2VecAdder.models[epochs][dim]
                m = defaultdict()
                m['fetch'] = np.empty(shape=(0, dim))
                m['infer'] = np.empty(shape=(0, dim))
                for text in tqdm(df['text'].values):
                    text = lt.clean_text(text)
                    if text != '':
                        if text in Doc2VecAdder.index:
                            fetch_vec = model.docvecs[Doc2VecAdder.index[text]]
                        else:
                            fetch_vec = [0.0] * dim
                        infer_vec = model.infer_vector(text.split())
                    else:
                        fetch_vec = [0.0] * dim
                        infer_vec = [0.0] * dim
                    m['fetch'] = np.append(m['fetch'], [fetch_vec], axis=0)
                    m['infer'] = np.append(m['infer'], [infer_vec], axis=0)
                ms[dim][epochs] = m
        fitdata = {'parents': parents, 'ms': ms, 'method': method, 'e': e, 'd': d}
        save_fitdata(data_prefix, 'Doc2VecAdder', fitdata)
        print('Doc2Vec PreFit: Done.')

    @staticmethod
    def loadfit(data_prefix):
        Doc2VecAdder.fitdata = load_fitdata(data_prefix, 'Doc2VecAdder')

    def __init__(self, disable=False, method='infer', ngrams=1, e=100, d=100, ps = 'abs', prescale = False):
        self.ngrams = ngrams
        self.method = method
        self.e = e
        self.d = d
        self.ps = ps
        self.disable = disable
        self.prescale = prescale

    @staticmethod
    def create_matrix(method, ngrams, e, d, ps):
        my_ps = get_ps(ngrams, ps)
        fitdata = Doc2VecAdder.fitdata
        if d not in fitdata['d']:
            print('Doc2VecAdder: bad dim [%d] parameter, no prefitted data. Using [%d]' % (d, fitdata['d'][0]))
            d = fitdata['d'][0]
        if e not in fitdata['e']:
            print('Doc2VecAdder: bad epochs [%d] parameter, no prefitted data. Using [%d]' % (e, fitdata['e'][0]))
            e = fitdata['e'][0]
        if method not in fitdata['method']:
            print('Doc2VecAdder: bad method [%d] parameter, no prefitted data. Using [%d]' % (method, fitdata['method'][0]))
            e = fitdata['method'][0]
        matrix = extend_matrix(fitdata['ms'][d][e][method], fitdata['parents'], ngrams, parent_selection=my_ps)
        return matrix

    def fit(self, X, y = None, **fit_params):
        if not self.disable:
            self.matrix = Doc2VecAdder.create_matrix(self.method, self.ngrams, self.e, self.d, self.ps)
        if self.prescale:
            self.matrix = MinMaxScaler().fit_transform(self.matrix)
        return self

    def transform(self, X, **transform_params):
        if self.disable:
            return X
        # get the first column of X - it represents rows indexes
        rows = X[:,0].astype(int)
        matrix = np.hstack((X, self.matrix[rows, :]))
        return matrix

    def set_params(self, disable=False, method='infer', ngrams=1, e=100, d=100, ps='abs', prescale=False, **parameters):
        self.ngrams = ngrams
        self.method = method
        self.e = e
        self.d = d
        self.ps = ps
        self.disable = disable
        self.prescale = prescale
        return self

    def get_params(self, **kwargs):
        return {"disable": self.disable, "method": self.method, "ngrams": self.ngrams, "e": self.e, "d": self.d, "ps": self.ps, "prescale": self.prescale}


class BowAdder(BaseEstimator, TransformerMixin):
    @staticmethod
    def prefit(data_prefix, max_features=[500, 1000, 1500, 2000, 3000, 5000], vectype = ['binary', 'count', 'tfidf']):
        df_path = exp_path + 'data/' + data_prefix + '.csv'
        df = pd.read_csv(df_path, encoding="utf-8").fillna('')
        with open('/data/work/data/reddit_cmv/bow_vocabulary.dill', 'rb') as voc_file:
            vocabulary_dict = dill.load(voc_file)
        vocabulary = list(vocabulary_dict.keys())
        corpus = df['text']
        parents = df['parent'].to_numpy().flatten()
        m = defaultdict(dict)
        for vec_t in vectype:
            for max_f in max_features:
                if vec_t == 'binary':
                    bow_vectorizer = CountVectorizer(vocabulary=vocabulary[:max_f], tokenizer=str.split, preprocessor=lt.clean_text, binary=True)
                if vec_t == 'count':
                    bow_vectorizer = CountVectorizer(vocabulary=vocabulary[:max_f], tokenizer=str.split, preprocessor=lt.clean_text)
                if vec_t == 'tfidf':
                    bow_vectorizer = TfidfVectorizer(vocabulary=vocabulary[:max_f], tokenizer=str.split, preprocessor=lt.clean_text)
                m[vec_t][max_f] = extend_matrix(bow_vectorizer.fit_transform(corpus).toarray(), parents, 1)
        fitdata = {'parents': parents, 'm': m, 'max_features': max_features, 'vectype': vectype}
        save_fitdata(data_prefix, 'BowAdder', fitdata)

    @staticmethod
    def loadfit(data_prefix):
        BowAdder.fitdata = load_fitdata(data_prefix, 'BowAdder')

    @staticmethod
    def create_matrix(max_f, vec_t, ngrams, ps):
        my_ps = get_ps(ngrams, ps)
        fitdata = BowAdder.fitdata
        if max_f not in fitdata['max_features']:
            print('BowAdder: bad max_f [%d] parameter, no prefitted data. Using [%d]' % (max_f, fitdata['max_features'][0]))
            max_f = fitdata['max_features'][0]
        if vec_t not in fitdata['vectype']:
            print('BowAdder: bad vec_ty [%s] parameter, no prefitted data. Using [%s]' % (vec_t, fitdata['vectype'][0]))
            vec_t = fitdata['vectype'][0]
        matrix = extend_matrix(fitdata['m'][vec_t][max_f], fitdata['parents'], ngrams, parent_selection=my_ps)
        return matrix

    def __init__(self, disable=False, ngrams=1, max_features=1000, vectype='tfidf', ps = 'abs', prescale=False):
        self.disable = disable
        self.ngrams = ngrams
        self.max_features = max_features
        self.ps = ps
        self.vectype = vectype
        self.prescale = prescale

    def fit(self, X, y = None, **fit_params):
        if not self.disable:
            self.matrix = BowAdder.create_matrix(self.max_features, self.vectype, self.ngrams, self.ps)
        if self.prescale:
            self.matrix = MinMaxScaler().fit_transform(self.matrix)
        return self

    def transform(self, X, **transform_params):
        if self.disable:
            return X
        # get the first column of X - it represents rows indexes
        rows = X[:,0].astype(int)
        matrix = np.hstack((X, self.matrix[rows, :]))
        return matrix

    def set_params(self, disable = False, ngrams = 1, max_features = 1000, vectype = 'tfidf', ps = 'abs', prescale = False, **parameters):
        self.disable = disable
        self.ngrams = ngrams
        self.max_features = max_features
        self.ps = ps
        self.prescale = prescale
        self.vectype = vectype
        return self

    def get_params(self, **kwargs):
        return {"disable": self.disable, "max_features": self.max_features, "vectype": self.vectype, "ngrams": self.ngrams, "ps": self.ps, "prescale": self.prescale}


class LiwcAdder(BaseEstimator, TransformerMixin):
    @staticmethod
    def prefit(data_prefix, vectype=['count', 'tfidf', 'binary']):
        df_path = exp_path + 'data/' + data_prefix + '.csv'
        df = pd.read_csv(df_path, encoding="utf-8").fillna('')
        corpus = df['text'].apply(lt.clean_text).apply(tt.text_to_liwc)
        parents = df['parent'].to_numpy().flatten()
        tfidf_matrix = TfidfVectorizer(tokenizer=str.split, preprocessor=None).fit_transform(corpus)
        count_matrix = CountVectorizer(tokenizer=str.split, preprocessor=None).fit_transform(corpus)
        binary_matrix = CountVectorizer(tokenizer=str.split, preprocessor=None, binary = True).fit_transform(corpus)
        ms = defaultdict()
        ms['count'] = count_matrix.toarray()
        ms['tfidf'] = tfidf_matrix.toarray()
        ms['binary'] = binary_matrix.toarray()
        fitdata = {'parents': parents, 'ms': ms, 'vectype': vectype}
        save_fitdata(data_prefix, 'LiwcAdder', fitdata)

    @staticmethod
    def loadfit(data_prefix):
        LiwcAdder.fitdata = load_fitdata(data_prefix, 'LiwcAdder')

    def __init__(self, disable=False, vectype='count', ngrams=1, ps='abs', prescale=False):
        self.disable = disable
        self.ngrams = ngrams
        self.vectype = vectype
        self.ps = ps
        self.prescale = prescale

    @staticmethod
    def create_matrix(vectype, ngrams, ps):
        my_ps = get_ps(ngrams, ps)
        fitdata = LiwcAdder.fitdata
        if vectype not in fitdata['vectype']:
            print('LiwcAdder: bad vectype [%d] parameter, no prefitted data. Using [%d]' % (vectype, fitdata['vectype'][0]))
            vectype = fitdata['vectype'][0]
        matrix = extend_matrix(fitdata['ms'][vectype], fitdata['parents'], ngrams, parent_selection=my_ps)
        return matrix

    def fit(self, X, y = None, **fit_params):
        if not self.disable:
            self.matrix = LiwcAdder.create_matrix(self.vectype, self.ngrams, self.ps)
        if self.prescale:
            self.matrix = MinMaxScaler().fit_transform(self.matrix)
        return self

    def transform(self, X, **transform_params):
        if self.disable:
            return X
        # get the first column of X - it represents rows indexes
        rows = X[:,0].astype(int)
        matrix = np.hstack((X, self.matrix[rows, :]))
        return matrix

    def set_params(self, disable = False, ngrams = 1, vectype = 'count', ps = 'abs', prescale=False, **parameters):
        self.disable = disable
        self.ngrams = ngrams
        self.vectype = vectype
        self.ps = ps
        self.prescale = prescale
        return self

    def get_params(self, **kwargs):
        return {"disable": self.disable, "vectype": self.vectype, "ngrams": self.ngrams, "ps": self.ps, "prescale": self.prescale}


class LTPmiAdder(BaseEstimator, TransformerMixin):
    #ltpmivecspath = 'C:/Users/vaule/PycharmProjects/Reddit/src/remote/22919-games/'
    ltpmivecspath = '/data/work/data/reddit_cmv/'
    @staticmethod
    def loadpmivecs(path = ltpmivecspath):
        with open(path + 'liwc_tag_pmi_vectors_mv.dill', 'rb') as f:
            LTPmiAdder.mvltpmivecs = dill.load(f)
        with open(path + 'liwc_tag_npmi_vectors_mv.dill', 'rb') as f:
            LTPmiAdder.mvltnpmivecs = dill.load(f)

        LTPmiAdder.ltpmimatrix = {}
        LTPmiAdder.ltnpmimatrix = {}

        for v in LTPmiAdder.mvltpmivecs.keys():
            LTPmiAdder.ltpmimatrix[v] = np.array(list(LTPmiAdder.mvltpmivecs[v].values()))
            LTPmiAdder.ltpmimatrix[v][np.isneginf(LTPmiAdder.ltpmimatrix[v])] = 0
        for v in LTPmiAdder.mvltnpmivecs.keys():
            LTPmiAdder.ltnpmimatrix[v] = np.array(list(LTPmiAdder.mvltnpmivecs[v].values()))

    @staticmethod
    def prefit(data_prefix, v = ['-v0', '-v1', '-v2', '-v3b', '-v3c', '-v3t'],
                            liwcvectype=['count', 'tfidf', 'binary'], variation=['pmi', 'npmi'],
                            agg=['sum', 'avg']):
        LTPmiAdder.loadpmivecs()
        df_path = exp_path + 'data/' + data_prefix + '.csv'
        df = pd.read_csv(df_path, encoding="utf-8").fillna('')
        corpus = df['text'].apply(lt.clean_text).apply(tt.text_to_liwc)
        parents = df['parent'].to_numpy().flatten()
        ms = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
        voc1 = {liwc: i for i, liwc in enumerate(list(LTPmiAdder.mvltpmivecs.values())[0].keys())}
        tfidf_matrix = TfidfVectorizer(tokenizer=str.split, vocabulary=voc1, preprocessor=None).fit_transform(corpus)
        count_matrix = CountVectorizer(tokenizer=str.split, vocabulary=voc1, preprocessor=None).fit_transform(corpus)
        binary_matrix = CountVectorizer(tokenizer=str.split, vocabulary=voc1, preprocessor=None,
                                        binary=True).fit_transform(corpus)
        vectypems = {'count': count_matrix.toarray(), 'tfidf': tfidf_matrix.toarray(),
                     'binary': binary_matrix.toarray()}
        for vt in v:
            voc3 = {liwc: i for i, liwc in enumerate(LTPmiAdder.mvltpmivecs[vt].keys())}
            voc2 = {liwc: i for i, liwc in enumerate(LTPmiAdder.mvltnpmivecs[vt].keys())}
            if voc1 != voc2 != voc3:
                print('Different V for Pmi Voc and Npmi Voc are not equal! Problem.')
            for lvt in liwcvectype:
                vtm = vectypems[lvt]
                for vart in variation:
                    for aggt in agg:
                        if vart == 'pmi':
                            m = np.dot(vtm, LTPmiAdder.ltpmimatrix[vt])
                        if vart == 'npmi':
                            m = np.dot(vtm, LTPmiAdder.ltnpmimatrix[vt])
                        if aggt == 'avg':
                            s = np.sum(vtm, axis=1)
                            b = s[:, None]
                            m = m / b
                            m[np.isnan(m)] = 0
                        ms[vt][lvt][vart][aggt] = m

            fitdata = {'parents': parents, 'ms': ms, 'liwcvectype': liwcvectype, 'variation': variation,
                       'agg': agg, 'v': v}
            save_fitdata(data_prefix, 'LTPmiAdder', fitdata)

    @staticmethod
    def loadfit(data_prefix):
        LTPmiAdder.fitdata = load_fitdata(data_prefix, 'LTPmiAdder')

    def __init__(self, disable=False, v='-v1', liwcvectype='count', variation='npmi', agg='sum', prescale=False, ngrams=1, ps='abs'):
        self.disable = disable
        self.ngrams = ngrams
        self.v = v
        self.liwcvectype = liwcvectype
        self.variation = variation
        self.agg = agg
        self.prescale = prescale
        self.ps = ps

    @staticmethod
    def create_matrix(v, liwcvectype, variation, agg, ngrams, ps):
        my_ps = get_ps(ngrams, ps)
        fitdata = LTPmiAdder.fitdata
        if v not in fitdata['v']:
            print('LTPmiAdder: bad v [%s] parameter, no prefitted data. Using [%s]' % (v, fitdata['v'][0]))
            v = fitdata['v'][0]
        if liwcvectype not in fitdata['liwcvectype']:
            print('LTPmiAdder: bad liwcvectype [%s] parameter, no prefitted data. Using [%s]' % (liwcvectype, fitdata['liwcvectype'][0]))
            liwcvectype = fitdata['liwcvectype'][0]
        if variation not in fitdata['variation']:
            print('LTPmiAdder: bad variation [%s] parameter, no prefitted data. Using [%s]' % (variation, fitdata['variation'][0]))
            variation = fitdata['variation'][0]
        if agg not in fitdata['agg']:
            print('LTPmiAdder: bad agg [%s] parameter, no prefitted data. Using [%s]' % (agg, fitdata['agg'][0]))
            agg = fitdata['agg'][0]
        matrix = extend_matrix(fitdata['ms'][v][liwcvectype][variation][agg], fitdata['parents'], ngrams, parent_selection=my_ps)
        return matrix

    def fit(self, X, y = None, **fit_params):
        if not self.disable:
            self.matrix = LTPmiAdder.create_matrix(self.v, self.liwcvectype, self.variation, self.agg, self.ngrams, self.ps)
        if self.prescale:
            self.matrix = MinMaxScaler().fit_transform(self.matrix)
        return self

    def transform(self, X, **transform_params):
        if self.disable:
            return X
        # get the first column of X - it represents rows indexes
        rows = X[:,0].astype(int)
        matrix = np.hstack((X, self.matrix[rows, :]))
        return matrix

    def set_params(self, disable=False, v='-v1', liwcvectype='count', variation='npmi', agg='sum', prescale=False, ngrams=1, ps='abs'):
        self.disable = disable
        self.ngrams = ngrams
        self.v = v
        self.liwcvectype = liwcvectype
        self.variation = variation
        self.agg = agg
        self.prescale = prescale
        self.ps = ps
        return self

    def get_params(self, **kwargs):
        return {"disable": self.disable, "v": self.v, "liwcvectype": self.liwcvectype, "variation": self.variation, "agg": self.agg,
                "prescale": self.prescale, "ngrams": self.ngrams, "ps": self.ps}


class TagAdder(BaseEstimator, TransformerMixin):
    target_tag = None

    @staticmethod
    def prefit(data_prefix):
        df_path = exp_path + 'data/' + data_prefix + '.csv'
        df = pd.read_csv(df_path, encoding="utf-8").fillna('')
        tags = sorted(set(df.columns.values.tolist()) -
                      set(['Unnamed: 0', 'node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent']))
        tags_matrix = df[tags].to_numpy()
        parents = df['parent'].to_numpy().flatten()
        fitdata = {'parents': parents, 'tags_matrix': tags_matrix, 'tags': tags}
        save_fitdata(data_prefix, 'TagAdder', fitdata)

    @staticmethod
    def loadfit(data_prefix):
        TagAdder.fitdata = load_fitdata(data_prefix, 'TagAdder')

    def __init__(self, disable=False, current=True, prev=2):
        self.disable = disable
        self.current = current
        self.prev = prev

    @staticmethod
    def create_matrix(current, prev):
        fitdata = TagAdder.fitdata
        target_key = TagAdder.target_tag if current else None
        if target_key is not None:
            notarget_matrix = np.delete(fitdata['tags_matrix'], fitdata['tags'].index(target_key), axis=1)
        else:
            notarget_matrix = np.empty(shape=(fitdata['tags_matrix'].shape[0], 0))
        parents_matrix = extend_matrix(fitdata['tags_matrix'], fitdata['parents'], prev+1, only_parents=True)
        matrix = np.hstack((notarget_matrix, parents_matrix))
        return matrix

    def fit(self, X, y = None, **fit_params):
        if not self.disable:
            self.matrix = TagAdder.create_matrix(self.current, self.prev)
        return self

    def transform(self, X, **transform_params):
        if self.disable:
            return X
        # get the first column of X - it represents rows indexes
        rows = X[:,0].astype(int)
        matrix = np.hstack((X, self.matrix[rows, :]))
        return matrix

    def set_params(self, disable = False, prev = 2, current = True, **parameters):
        self.disable = disable
        self.prev = prev
        self.current = current
        return self

    def get_params(self, **kwargs):
        return {"disable": self.disable, "prev": self.prev, "current": self.current}


class IndexRemover(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return X[:, 1:]


class TagSelector(BaseEstimator, TransformerMixin):
    @staticmethod
    def prefit(data_prefix):
        df_path = exp_path + 'data/' + data_prefix + '.csv'
        df = pd.read_csv(df_path, encoding="utf-8").fillna('')
        tags = sorted(set(df.columns.values.tolist()) -
                      set(['Unnamed: 0', 'node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent']))
        tags_matrix = df[tags].to_numpy()
        fitdata = {'tags_matrix': tags_matrix, 'tags': tags}
        save_fitdata(data_prefix, 'TagSelector', fitdata)

    @staticmethod
    def loadfit(data_prefix):
        TagSelector.fitdata = load_fitdata(data_prefix, 'TagSelector')

    def __init__(self, disable=False, target_tag='CBE'):
        self.disable = disable
        self.target_tag = target_tag

    @staticmethod
    def create_matrix(target_tag):
        fitdata = TagSelector.fitdata
        matrix = fitdata['tags_matrix'][:, [fitdata['tags'].index(target_tag)]].reshape(fitdata['tags_matrix'].shape[0], 1)
        return matrix

    def fit(self, X, y = None, **fit_params):
        if not self.disable:
            self.matrix = TagSelector.create_matrix(self.target_tag)
        return self

    def transform(self, X, **transform_params):
        if self.disable:
            return X
        # get the first column of X - it represents rows indexes
        rows = X[:,0].astype(int)
        matrix = np.hstack((X, self.matrix[rows, :]))
        return matrix

    def set_params(self, disable = False, target_tag='CBE', **parameters):
        self.disable = disable
        self.target_tag = target_tag
        return self

    def get_params(self, **kwargs):
        return {"disable": self.disable, "target_tag": self.target_tag}


class TensorTransfomer(BaseEstimator, TransformerMixin):
    def __init__(self, disable=False, tensor_shape=None):
        self.disable = disable
        self.tensor_shape = tensor_shape

    def fit(self, X, y = None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        if self.disable:
            return X
        tensor = torch.from_numpy(X)
        if self.tensor_shape is not None:
            tensor = tensor.view(self.tensor_shape)
        return tensor

    def set_params(self, disable = False, tensor_shape=None, **parameters):
        self.disable = disable
        self.tensor_shape = tensor_shape
        return self

    def get_params(self, **kwargs):
        return {"disable": self.disable, "tensor_shape": self.tensor_shape}


class ClsAdder(BaseEstimator, ClassifierMixin, TransformerMixin):

    total_counter = 0
    perclasscounter = Counter()
    times = defaultdict(int)

    @staticmethod
    def report_times(filename):
        with open(filename, 'w') as outfile:
            print('Total counts: ', ClsAdder.total_counter, file=outfile)
            print('Per-Class counts: ', ClsAdder.perclasscounter, file=outfile)
            print("Per-Class times: ", ClsAdder.times, file=outfile)

    def _create_classifier(self, cls):
        if cls == "SVM":
            self.classifier = LinearSVC()
        if cls == "NB":
            self.classifier = ComplementNB()
        if cls == "LR":
            self.classifier = LogisticRegression()
        if cls == "DT":
            self.classifier = DecisionTreeClassifier()
        if cls == "RF":
            self.classifier = RandomForestClassifier()
        if cls == "MLP":
            self.classifier = MLPClassifier(activation='tanh', hidden_layer_sizes=(100,100,100), max_iter=1000, solver='lbfgs')
            #self.classifier = MLPClassifier(hidden_layer_sizes=(128,))

    def __init__(self, cls = None):
        self.cls = cls
        self._create_classifier(cls)

    def set_params(self, cls, **parameters):
        self.cls = cls
        self._create_classifier(cls)
        return self

    def get_params(self, **kwargs):
        return {"cls": self.cls}

    def _count_print(self):
        if ClsAdder.total_counter == 0:
            ClsAdder.t0 = time()

        ClsAdder.times[self.cls] += time() - ClsAdder.t0
        ClsAdder.total_counter+=1
        ClsAdder.perclasscounter[self.cls] += 1
        ClsAdder.t0 = time()

        if ClsAdder.total_counter % 50 == 0:
            print('Total counts: ', ClsAdder.total_counter)
            print('Per-Class counts: ', ClsAdder.perclasscounter)
            print("Per-Class times: ", ClsAdder.times)

    def transform(self, X, **transform_params):
        return X

    def fit(self, X, y, **fit_params):
        self._count_print()
        myX = X[:, 1:] if X.shape[1] > 1 else np.zeros(shape=(X.shape[0], 1))
        self.classifier.fit(myX, y)
        return self

    def predict(self, X, **kwargs):
        myX = X[:, 1:] if X.shape[1] > 1 else np.zeros(shape=(X.shape[0], 1))
        return self.classifier.predict(myX)

    def score(self, X, y, sample_weight=None):
        myX = X[:, 1:] if X.shape[1] > 1 else np.zeros(shape=(X.shape[0], 1))
        return self.classifier.score(myX, y, sample_weight)

def create_d2vtrain_lines(trees_list_path, out_file_name):
    trees = tt.load_list_of_trees(trees_list_path)
    with open(out_file_name, 'w', encoding="utf-8") as out_file:
        for tree in trees:
            for node in tt.get_nodes(tree):
                if 'text' in node:
                    text = lt.clean_text(node['text'])
                    if text != '':
                        out_file.write(text)
                        out_file.write('\n')
    print('DONE: ', out_file_name)


def get_ps(n, ps):
    if n == 1:
        return 'abs'
    if n == 2:
        if ps == 'abs':
            return 'abs'
        else:
            return 'dist_first'
    if n >= 3:
        return ps


def train_doc2vec(train_lines, out_model, epochs, dim):
    with open(train_lines, encoding='utf-8') as in_file:
        train_corpus = [gensim.models.doc2vec.TaggedDocument(line.split(), [i]) for i, line in enumerate(in_file)]
        model = gensim.models.doc2vec.Doc2Vec(train_corpus, vector_size=dim, min_count=11, epochs=epochs, workers=16)
        model.save(out_model)
        print('DONE: ', out_model)


def save_fitdata(data_prefix, adder_name, fitdata):
    with open(exp_path + 'prefit/' + data_prefix + '_' + adder_name + '.fitdata', 'wb') as fit_data_file:
        dill.dump(fitdata, fit_data_file, dill.HIGHEST_PROTOCOL)


def load_fitdata(data_prefix, adder_name):
    with open(exp_path + 'prefit/' + data_prefix + '_' + adder_name + '.fitdata', 'rb') as fit_data_file:
        return dill.load(fit_data_file)


def prefit_adders(data_prefix, adders):
    if adders is None or 'b' in adders:
        BowAdder.prefit(data_prefix)
    if adders is None or 'g' in adders:
        GloveAdder.prefit(data_prefix)
    if adders is None or 'l' in adders:
        LiwcAdder.prefit(data_prefix)
    if adders is None or 't' in adders:
        TagAdder.prefit(data_prefix)
    if adders is None or 's' in adders:
        TagSelector.prefit(data_prefix)
    if adders is None or 'p' in adders:
        LTPmiAdder.prefit(data_prefix)
    if adders is None or 'r' in adders:
        DisAdder.prefit(data_prefix)
    if adders is None or 'd' in adders:
        Doc2VecAdder.prefit(data_prefix, method=['fetch', 'infer'], e=[100, 500], d=[100, 500, 1000])
        #Doc2VecAdder.prefit(data_prefix, method=['fetch'], e=[500], d=[1000])

def load_adders(data_prefix, adders = None):
    if adders is None or 'b' in adders:
        BowAdder.loadfit(data_prefix)
    if adders is None or 'g' in adders:
        GloveAdder.loadfit(data_prefix)
    if adders is None or 'l' in adders:
        LiwcAdder.loadfit(data_prefix)
    if adders is None or 't' in adders:
        TagAdder.loadfit(data_prefix)
    if adders is None or 's' in adders:
        TagSelector.loadfit(data_prefix)
    if adders is None or 'p' in adders:
        LTPmiAdder.loadfit(data_prefix)
    if adders is None or 'd' in adders:
        Doc2VecAdder.loadfit(data_prefix)
    if adders is None or 'r' in adders:
        DisAdder.loadfit(data_prefix)
    print("Adders loaded for %s from : %s" % (data_prefix, exp_path+'prefit/'))
