import LabelTools as lt
import TreeTools as tt
import FeatureTools as ft
import SequenceTools as st
from pipelinehelper import PipelineHelper

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
import tldextract
from collections import defaultdict, Counter
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from time import time
from tqdm import tqdm
import pprint
from copy import deepcopy
import os
import glob
import re

import skorch
from joblib import Parallel, delayed


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def transform_results(dir=ft.exp_path+'results/'):
    filenames = [file for file in glob.glob(dir + '*.csv')]
    for fn in filenames:
        df = pd.read_csv(fn)
        df_columns = set(df.columns.values.tolist())
        baseline_columns = {'mean_fit_time','std_fit_time','mean_score_time','std_score_time','params','split0_test_f1','split1_test_f1','split2_test_f1','split3_test_f1','split4_test_f1','mean_test_f1','std_test_f1','rank_test_f1','split0_test_recall','split1_test_recall','split2_test_recall','split3_test_recall','split4_test_recall','mean_test_recall','std_test_recall','rank_test_recall','split0_test_precision','split1_test_precision','split2_test_precision','split3_test_precision','split4_test_precision','mean_test_precision','std_test_precision','rank_test_precision','prior','improvement'}
        lstmsa_columns = {'epoch','t.prior','t.imprv','t.f1','t.pr','t.rec','t.acc','t.loss','v.prior','v.imprv','v.f1','v.pr','v.rec','v.acc','v.loss','dur'}
        if df_columns >= baseline_columns:
            if 'f1_naive' not in df_columns:
                df['f1_naive'] = 2.0*df['prior'] / (1.0 + df['prior'])
            if 'f1_imprv' not in df_columns:
                df['f1_imprv'] = df['mean_test_f1'] / df['f1_naive']
            if 'precision*recall' not in df_columns:
                df['precision*recall'] = df['mean_test_precision'] * df['mean_test_recall']
            if 'pr_imprv' not in df_columns:
                df['pr_imprv'] = df['precision*recall'] / df['prior']
            df = df.rename(columns={'improvement': 'improvement_oldbad'})
            df.to_csv(fn)
            print('DONE: %s' % fn)
        if df_columns >= lstmsa_columns:
            if 'v.f1_naive' not in df_columns:
                df['v.f1_naive'] = 2.0*df['v.prior'] / (1.0 + df['v.prior'])
            if 'v.f1_imprv' not in df_columns:
                df['v.f1_imprv'] = df['v.pr'] / df['v.f1_naive']
            if 'v.p*r' not in df_columns:
                df['v.p*r'] = df['v.pr'] * df['v.rec']
            if 'v.p*r_imprv' not in df_columns:
                df['v.pr_imprv'] = df['v.p*r'] / df['v.prior']
            df = df.rename(columns={'v.imprv': 'v.improvement_oldbad'})
            df.to_csv(fn)
            print('DONE: %s' % fn)


def tag_averager(tags, pipe_prefix, data_prefix, avg_by='f1_imprv'):
    all_scores = defaultdict(dict)
    avg_scores = defaultdict(lambda: defaultdict(list))
    avg_of_best_scores = defaultdict(dict)
    keys = ['prior', 'f1_naive', 'mean_test_f1', 'f1_imprv', 'precision*recall', 'pr_imprv', 'mean_test_precision', 'mean_test_recall', 'rank_test_f1', 'rank_test_precision',
            'rank_test_recall']

    # keys.append('improvement')
    results_dir = ft.exp_path + 'results/'
    scores_dir = ft.exp_path + 'scores/'
    prefix = data_prefix + '_' + pipe_prefix
    priors = []
    best_indexes = {}
    for tag in tags:
        results_df = pd.read_csv(results_dir + prefix + '_' + tag + '.csv')
        all_scores[tag]['params'] = results_df['params'].tolist()
        for key in keys:
            all_scores[tag][key] = results_df[key].to_numpy()
        best_indexes[tag] = np.argmax(all_scores[tag][avg_by])
        priors.append(results_df['prior'].tolist()[0])

    for tag_counter in range(1, len(tags)):
        if len(all_scores[tags[tag_counter]]['params']) != len(all_scores[tags[tag_counter-1]]['params']):
            print('Validation failed. Different amount of Params list! %d vs %d' % (
                len(all_scores[tags[tag_counter]]['params']), len(all_scores[tags[tag_counter-1]]['params'])))
            exit()
        for i in range(len(all_scores[tags[tag_counter]]['params'])):
            if all_scores[tags[tag_counter]]['params'][i] != all_scores[tags[tag_counter-1]]['params'][i]:
                print('Validation failed. Different Params Lists!')
                exit()

    total_params = len(all_scores[tags[0]]['params'])

    weights_variations = {
        '0': np.ones(len(priors)),
        '1': np.array(priors),
        '2': np.sqrt(priors),
        '3': np.cbrt(priors)
    }

    for w_type, weights in weights_variations.items():
        for i in range(total_params):
            for key in keys:
                vals = np.empty(shape=(len(tags),))
                for tag_counter, tag in enumerate(tags):
                    vals[tag_counter] = all_scores[tag][key][i]
                avg_scores[w_type][key].append(np.average(vals, weights=weights))
            avg_scores[w_type]['params'].append(all_scores[tags[0]]['params'][i])
        pd.DataFrame(avg_scores[w_type]).to_csv(scores_dir + prefix + '_average_'+w_type+'.csv')
        print('DONE: %s' % (scores_dir + prefix + '_average_'+w_type+'.csv'))
    best_of_all = defaultdict(dict)
    for w_type, weights in weights_variations.items():
        keys.append('params')
        best_avg_index = np.argmax(avg_scores[w_type]['mean_test_f1'])
        best_avg_scores = defaultdict(dict)

        for tag in tags:
            best_tag_index = best_indexes[tag]
            for key in keys:
                best_avg_scores[key][tag] = all_scores[tag][key][best_avg_index]
                best_of_all[key][tag] = all_scores[tag][key][best_tag_index]

        keys.remove('params')
        for key in keys:
            best_avg_scores[key]['average'] = avg_scores[w_type][key][best_avg_index]
            best_of_all[key]['best_of_average'+w_type] = avg_scores[w_type][key][best_avg_index]
            vals = np.empty(shape=(len(tags),))
            for tag_counter, tag in enumerate(tags):
                vals[tag_counter] = best_of_all[key][tag]
            best_of_all[key]['average_of_best'+w_type] = np.average(vals, weights=weights)
        best_of_all['params']['best_of_average'+w_type] = avg_scores[w_type]['params'][best_avg_index]
        best_of_all['params']['average_of_best'+w_type] = 'undefined'

        pd.DataFrame(best_avg_scores).to_csv(scores_dir + prefix + '_best_of_average_'+w_type+'.csv')
        print('DONE: %s' % (scores_dir + prefix + '_best_of_average_'+w_type+'.csv'))

    pd.DataFrame(best_of_all).to_csv(scores_dir + prefix + '_best_of_all.csv')
    print('DONE: %s' % (scores_dir + prefix + '_best_of_all.csv'))


def split_list(l, wanted_parts=1):
    length = len(l)
    return [ l[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts) ]


def aggregate_scores(data_prefix, pipe_prefix):
    df_path = ft.exp_path + 'data/' + data_prefix + '.csv'
    df = pd.read_csv(df_path, encoding="utf-8")
    all_tags = sorted(set(df.columns.values.tolist()) -
                      set(['Unnamed: 0', 'node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent']))
    tag_averager(all_tags, pipe_prefix, data_prefix)


def train_test(train_data, test_data, pipe_prefix, params_prefix, mode='avg0', classifier=None, target_tags=None, adders=None):
    train_df = pd.read_csv(ft.exp_path + 'data/' + train_data + '.csv', encoding="utf-8", keep_default_na=False)
    test_df = pd.read_csv(ft.exp_path + 'data/' + test_data + '.csv', encoding="utf-8", keep_default_na=False)
    params_dir = ft.exp_path + 'params/'
    results_dir = ft.exp_path + 'results/'
    scores_dir = ft.exp_path + 'scores/'
    test_scores_dir = ft.exp_path + 'test_scores/'
    train_tags = sorted(set(train_df.columns.values.tolist()) -
                      set(['Unnamed: 0', 'node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent']))
    test_tags = sorted(set(test_df.columns.values.tolist()) -
                      set(['Unnamed: 0', 'node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent']))
    if train_tags != test_tags:
        print('Error! Train & Test tags are not identical. Imcompatible DataSets.')
        return
    if target_tags is None:
        target_tags = test_tags

    mode_prefix = 'best_of_average_0'
    if mode == 'best':
        mode_prefix = 'best_of_all'
    if mode == 'avg0':
        mode_prefix = 'best_of_average_0'
    if mode == 'avg1':
        mode_prefix = 'best_of_average_1'
    if mode == 'manual':
        manual_pipes = {}
        manual_params = {}
        manual_cls = {}
        with open(params_dir + params_prefix + '.txt') as params_file:
            for line in params_file:
                split_index = line.index('{')
                key = line[:split_index].split()[0].strip()
                pipe = line[:split_index].split()[1].strip()
                params = eval(line[split_index:].strip())
                cls = params['cls__cls']
                del params['cls__cls']
                if classifier is not None:
                    cls = classifier
                manual_params[key] = params
                manual_cls[key] = cls
                manual_pipes[key] = pipe
    if mode != 'manual':
        best_filename = '_'.join([train_data, pipe_prefix, params_prefix, mode_prefix])+'.csv'
        tag_params = {}
        tag_classifiers = {}
        with open(scores_dir + best_filename) as scores_file:
            scores_df = pd.read_csv(scores_file, keep_default_na=False)
            for _, row in scores_df.iterrows():
                if row[0] in target_tags:
                    tag = row[0]
                    params = eval(row['params'])
                    tag_classifiers[tag] = classifier if classifier is not None else params['cls__cls']
                    del params['cls__cls']
                    tag_params[tag] = params
        cls_prefix = 'cls' if classifier is None else classifier
        scores_filename = test_scores_dir + '_'.join([train_data, test_data, pipe_prefix, params_prefix, 'testscores', mode, cls_prefix]) + '.csv'
        train_test_phase(train_data, test_data, target_tags, pipe_prefix, adders, tag_params, tag_classifiers, scores_filename)
    else:
        tag_params_list = [{tag: params for tag in target_tags} for params in manual_params.values()]
        tag_classifiers_list = [{tag: cls for tag in target_tags} for cls in manual_cls.values()]
        pipes_list = list(manual_pipes.values())
        cls_prefix = 'cls' if classifier is None else classifier
        score_filename_list = [test_scores_dir + '_'.join([train_data, test_data, pipe_prefix, params_prefix, 'testscores', mode, cls_prefix, key]) + '.csv' for key in manual_params]
        zipped_tuples = zip(tag_params_list, tag_classifiers_list, pipes_list, score_filename_list)
        Parallel(n_jobs=len(score_filename_list))(delayed(train_test_phase)(train_data, test_data, target_tags, pipe, adders, tag_params, tag_classifiers, scores_filename) for tag_params, tag_classifiers, pipe, scores_filename in zipped_tuples)

def aggregate_traintest(train_data, test_data, pipe_prefix, params_prefix, mode='avg0', classifier=None, target_tags=None, adders=None):
    test_scores_dir = ft.exp_path + 'test_scores/'
    cls_prefix = 'cls' if classifier is None else classifier
    scores_filename_prefix = test_scores_dir + '_'.join([train_data, test_data, pipe_prefix, params_prefix, 'testscores', mode, cls_prefix])
    filenames = [file for file in glob.glob(scores_filename_prefix + '*.csv')]
    keys = ['prior', 'f1_naive', 'f1_imprv', 'f1', 'precision', 'recall']
    rows = ['average0', 'average1', 'average2', 'average3']
    aggregated = defaultdict(dict)
    for fn in filenames:
        entry = fn[len(scores_filename_prefix)+1:-4]
        df = pd.read_csv(fn, keep_default_na=False)
        for i, row in df.iterrows():
            if row[0] in rows:
                for key in keys:
                    aggregated[row[0] + '_' + key][entry] = row[key]
    outname = scores_filename_prefix + '_aggregated.csv'
    pd.DataFrame(aggregated).to_csv(outname)
    print('DONE:', outname)

def train_test_phase(train_data, test_data, tags, pipe_prefix, adders, tag_params, tag_classifiers, filename):
    pipes_dir = ft.exp_path + 'pipes/'
    train_df = pd.read_csv(ft.exp_path + 'data/' + train_data + '.csv', encoding="utf-8", keep_default_na=False)
    test_df = pd.read_csv(ft.exp_path + 'data/' + test_data + '.csv', encoding="utf-8", keep_default_na=False)
    train_tags = sorted(set(train_df.columns.values.tolist()) -
                      set(['Unnamed: 0', 'node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent']))
    test_tags = sorted(set(test_df.columns.values.tolist()) -
                      set(['Unnamed: 0', 'node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent']))
    if train_tags != test_tags:
        print('Error! Train & Test tags are not identical. Imcompatible DataSets.')
        return
    if tags is None:
        tags = test_tags
    with open(pipes_dir + pipe_prefix + '.txt', 'r') as pipe_file:
        transform_pipe = re.sub(r",?\s*\('cls', ft.ClsAdder\(\)\)",'',pipe_file.read())
    # TRAIN PHASE #
    X_trn = np.arange(len(train_df.index)).reshape((len(train_df.index), 1))
    train_pipe = eval(transform_pipe)
    ft.load_adders(train_data, adders)
    trained_classifiers = {}
    for tag, params in tag_params.items():
        print('Running TRAIN PHASE for: %s' % tag)
        ft.TagAdder.target_tag = tag
        train_pipe.set_params(**params)
        cls = ft.ClsAdder(tag_classifiers[tag])
        transformed_X = train_pipe.fit_transform(X_trn)
        cls.fit(transformed_X, train_df[tag])
        trained_classifiers[tag] = cls

    # TEST PHASE #
    X_tst = np.arange(len(test_df.index)).reshape((len(test_df.index), 1))
    test_pipe = eval(transform_pipe)
    ft.load_adders(test_data, adders)
    scores = defaultdict(dict)
    for tag, cls in trained_classifiers.items():
        print('Running TEST PHASE for: %s' % tag)
        ft.TagAdder.target_tag = tag
        test_pipe.set_params(**tag_params[tag])
        transformed_X = test_pipe.fit_transform(X_tst)
        y_predicted = cls.predict(transformed_X)
        y_true = test_df[tag]
        scores['prior'][tag] = np.mean(y_true)
        scores['f1_naive'][tag] = metrics.f1_score(y_true, [1]*len(y_true))
        f1 = metrics.f1_score(y_true, y_predicted)
        scores['f1_imprv'][tag] = f1 / scores['f1_naive'][tag]
        scores['f1'][tag] = f1
        scores['precision'][tag] = metrics.precision_score(y_true, y_predicted)
        scores['recall'][tag] = metrics.recall_score(y_true, y_predicted)
        scores['params'][tag] = str(tag_params[tag])
        scores['cls'][tag] = tag_classifiers[tag]
        scores['pipe'][tag] = transform_pipe

    priors = [scores['prior'][tag] for tag in tags]
    avg_keys = ['prior', 'f1_naive', 'f1_imprv', 'f1', 'precision', 'recall']
    weights_variations = {
        '0': np.ones(len(priors)),
        '1': np.array(priors),
        '2': np.sqrt(priors),
        '3': np.cbrt(priors)
    }
    for key in avg_keys:
        for w_type, weights in weights_variations.items():
            vals = [scores[key][tag] for tag in tags]
            scores[key]['average'+w_type] = np.average(vals, weights=weights)
    pd.DataFrame(scores).to_csv(filename)
    print('DONE:', filename)


def tag_scorer(X, df, pipe, params, cv, target_tags, data_prefix, pipe_prefix, params_prefix, adders, outdir):
    prefix = data_prefix + '_' + pipe_prefix + '_' + params_prefix
    ft.load_adders(data_prefix, adders)
    row_count = df.shape[0]
    priors = {tag: np.count_nonzero(df[tag].to_numpy()) / row_count for tag in target_tags}
    print('Running %s GridSearch for tags: ' % prefix, target_tags)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=cv)
    for tag in target_tags:
        ft.TagAdder.target_tag = tag
        search = GridSearchCV(pipe, params, cv=skf, scoring=['f1', 'recall', 'precision'], refit='f1')
        search.fit(X, df[tag])
        print("Tag: %s, Prior: %0.3f, Best parameter (CV score=%0.3f):" % (tag, priors[tag], search.best_score_))
        print('Tag: ', tag, search.best_params_)
        results = deepcopy(search.cv_results_)

        results['prior'] = [priors[tag]] * len(results['params'])
        results['improvement_oldbad'] = [results['mean_test_f1'][i] / results['prior'][i] for i in range(len(results['params']))]
        results['f1_naive'] = [results['prior'][i]*2.0/(1.0+results['prior'][i]) for i in range(len(results['params']))]
        results['f1_imprv'] = [results['mean_test_f1'][i] / results['f1_naive'][i] for i in range(len(results['params']))]
        results['precision*recall'] = [results['mean_test_precision'][i]*results['mean_test_recall'][i] for i in range(len(results['params']))]
        results['pr_imprv'] = [results['precision*recall'][i] / results['prior'][i] for i in range(len(results['params']))]

        pd.DataFrame(results).to_csv(outdir + prefix + '_' + tag + '.csv')
        print('DONE: %s ' % (outdir + prefix + '_' + tag + '.csv'))


def grid_search(data_prefix, pipe_prefix, params_prefix, cv=5, split_tags=None, target_tags=None, adders=None):
    """
    """
    df_path = ft.exp_path + 'data/' + data_prefix + '.csv'
    df = pd.read_csv(df_path, encoding="utf-8")
    pipes_dir = ft.exp_path + 'pipes/'
    params_dir = ft.exp_path + 'params/'
    results_dir = ft.exp_path + 'results/'
    all_tags = sorted(set(df.columns.values.tolist()) -
                      set(['Unnamed: 0', 'node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent']))

    if cv is None:
        cv = 5

    if target_tags is None:
        target_tags = all_tags

    if split_tags is not None:
        k = split_tags[0]
        n = split_tags[1]
        target_tags = split_list(all_tags, n)[k-1]

    X = np.arange(len(df.index)).reshape((len(df.index), 1))

    with open(pipes_dir + pipe_prefix + '.txt', 'r') as pipe_file:
        pipe = eval(pipe_file.read())
    with open(params_dir + params_prefix + '.txt', 'r') as params_file:
        params = eval(params_file.read())

    Parallel(n_jobs=len(target_tags))(delayed(tag_scorer)(X, df, pipe, params, cv, [target_tag], data_prefix, pipe_prefix, params_prefix, adders, results_dir) for target_tag in target_tags)

def aggregate_scores_lstmsa(data_prefix, tags = None, agg_by ='v.f1_imprv'):
    best_scores = defaultdict(dict)
    keys = ['v.prior', 'v.f1_naive','v.f1', 'v.f1_imprv', 'v.p*r', 'v.pr_imprv', 'v.pr', 'v.rec', 'v.acc', 't.prior', 'epoch']
    avg_keys = ['v.prior', 'v.f1_naive', 'v.f1', 'v.f1_imprv', 'v.p*r', 'v.pr_imprv', 'v.pr', 'v.rec', 'v.acc', 't.prior']
    priors = []
    results_dir = ft.exp_path + 'results/'
    scores_dir = ft.exp_path + 'scores/'
    prefix = 'lstmsa_' + data_prefix
    if not tags:
        tags = ['']
    filenames = [file for tag in tags for file in glob.glob(results_dir+prefix+'*' + tag + '.csv')]
    tags = set([tag for filename in filenames for tag in re.findall(r'\.*_([^_|w]+).csv', filename)])
    if '_SPLIT_' in ''.join(filenames):
        tags.add('_SPLIT_')
    for tag in tags:
        tag_scores = defaultdict(list)
        tag_filenames = [fn for fn in filenames if fn.endswith(tag+'.csv')]
        for fn in tag_filenames:
            results_df = pd.read_csv(fn)
            pipe_x = fn[:-len('_'+tag+'.csv')].split('_')[-2]
            dim = fn[:-len('_'+tag+'.csv')].split('_')[-1]
            for key in keys:
                tag_scores[key].append(results_df.iloc[results_df[agg_by].idxmax()][key])
            tag_scores['dim'].append(dim)
            tag_scores['pipe_x'].append(pipe_x)
            tag_scores['filename'].append(fn)
        tags_df = pd.DataFrame(tag_scores)
        for key in keys+['dim', 'pipe_x', 'filename']:
            best_scores[key][tag] = tags_df.iloc[tags_df[agg_by].idxmax()][key]
        priors.append(tag_scores['v.prior'][0])
        tags_df.to_csv(scores_dir + prefix + '_agg_' + tag + '.csv')
        print('DONE: %s' % (scores_dir + prefix + '_agg_' + tag + '.csv'))
    best_df = pd.DataFrame(best_scores)
    weights_variations = {
        '0': np.ones(len(priors)),
        '1': np.array(priors),
        '2': np.sqrt(priors),
        '3': np.cbrt(priors)
    }
    for key in avg_keys:
        for w_type, weights in weights_variations.items():
            best_scores[key]['average_'+w_type] = np.average(best_df[key].to_numpy(), weights=weights)
    best_df = pd.DataFrame(best_scores)
    best_df.to_csv(scores_dir + prefix + '_best_all.csv')
    print('DONE: %s' % (scores_dir + prefix + '_best_all.csv'))

def lstm_sa(data_prefix, target_tags, pipes, hidden_dims):
    if target_tags:
        target_tags = target_tags.split(',')
    else:
        target_tags = sorted(set(pd.read_csv(ft.exp_path + 'data/' + data_prefix + '.csv', encoding="utf-8").columns.values.tolist()) -
                      set(['Unnamed: 0', 'node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent']))

    pipes = pipes.split(',')
    hidden_dims = [int(d) for d in hidden_dims.split(',')]
    params_grid = [(target_tag, pipe, hidden_dim) for target_tag in target_tags for pipe in pipes for hidden_dim in hidden_dims]
    Parallel(n_jobs=len(params_grid))(delayed(lstm_sa_run)(data_prefix, target_tag, pipe, hidden_dim) for (target_tag, pipe, hidden_dim) in params_grid)


def lstm_sa_run(data_prefix, target_tag, pipe, hidden_dim):
    ft.load_adders(data_prefix)
    df_path = ft.exp_path + 'data/' + data_prefix + '.csv'
    pipes_dir = ft.exp_path + 'pipes/'
    results_dir = ft.exp_path + 'results/'
    results_file = results_dir+'_'.join(['lstmsa', data_prefix, pipe, str(hidden_dim), target_tag])+'.csv'
    with open(pipes_dir + pipe + '_pipe.txt', 'r') as pipe_file:
        pipe_x = eval(pipe_file.read())
    reporter = st.LSTM_SA_REPORTER(results_path=results_file)
    stopper = skorch.callbacks.EarlyStopping(monitor='v.f1', patience=50, threshold=0.0001,
                                             threshold_mode='rel', lower_is_better=False)
    trainer = st.LSTM_SA_TRAINER(df_path=df_path, target_tag=target_tag, pipe_x=pipe_x, lr=0.01,
                                 hidden_dim=hidden_dim, epochs=1000, callbacks=[reporter, stopper])
    trainer.fit()
