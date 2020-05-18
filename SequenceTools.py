import TreeTools as tt
import LabelTools as lt
import FeatureTools as ft
import ClassTools as ct
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from copy import deepcopy
import skorch
import sklearn

torch.manual_seed(1)


def return_seq(i, rows, marked_rows):
    if i in marked_rows:
        return None
    else:
        seq = []
        tree_id = rows[i][1]['tree_id']
        while i != -1:
            marked_rows.add(i)
            seq.insert(0, i)
            i = rows[i][1]['parent']
        return tree_id, seq


def get_sequences(df_path):
    df = pd.read_csv(df_path, encoding="utf-8")
    rows = list(df.iterrows())
    marked_rows = set()
    sequences = defaultdict(list)
    for i in reversed(range(len(rows))):
        seq = return_seq(i, rows, marked_rows)
        if seq is not None:
            sequences[seq[0]].insert(0, seq[1])
    all_sequences = []
    groups = []
    for tree_id, seq in sequences.items():
        all_sequences.extend(seq)
        groups.extend([tree_id] * len(seq))
    return all_sequences, groups


def seq_to_column(seq):
    return np.array(seq).reshape((len(seq),1)).astype(dtype=np.int32)


class LSTM_SA(nn.Module):
    def __init__(self, hidden_dim, transform_pipe):
        super(LSTM_SA, self).__init__()
        self.hidden_dim = hidden_dim
        input_dim = transform_pipe.fit_transform(seq_to_column([0])).size
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2tag = nn.Sequential(nn.Linear(hidden_dim, 2))
        self.transform_pipe = transform_pipe

    def forward(self, seq):
        X = torch.from_numpy(self.transform_pipe.fit_transform(seq)).float()
        lstm_out, _ = self.lstm(X.view(len(seq), 1, -1))
        tag_scores = self.hidden2tag(lstm_out.view(len(seq), -1))
        return tag_scores


class LSTM_SA_TRAINER(skorch.NeuralNet):
    def __init__(self, df_path, target_tag, pipe_x, hidden_dim, epochs, lr, *args, **kwargs):
        self.all_sequences, groups = get_sequences(df_path)
        train, test = GroupKFold(n_splits=5).split(X=self.all_sequences, groups=groups).__next__()
        self.train = train.astype(dtype=np.int32)
        self.test = test.astype(dtype=np.int32)
        ft.TagAdder.target_tag = target_tag
        super().__init__(module=LSTM_SA, module__hidden_dim=hidden_dim, module__transform_pipe=pipe_x,
                         max_epochs=epochs, batch_size=-1, lr=lr, optimizer=torch.optim.SGD,
                         criterion=torch.nn.CrossEntropyLoss,
                         train_split=lambda *_: ((self.train, np.zeros(len(self.train), dtype=np.int32)),
                                                 (self.test, np.zeros(len(self.test), dtype=np.int32))),
                         *args, **kwargs)
        self.df_path = df_path
        self.pipe_y = Pipeline(steps=[
                        ('tag', ft.TagSelector(target_tag=target_tag)),
                        ('ir', ft.IndexRemover())])

    def epoch_step(self, Xi, train_mode=True):
        losses = np.empty(Xi.size())
        loss_weights = np.empty(Xi.size())
        epoch_predicted = {}
        epoch_targets = {}
        for i, seq_index in enumerate(Xi):
            self.module_.zero_grad()
            seq = seq_to_column(self.all_sequences[seq_index])
            targets = self.pipe_y.fit_transform(seq)
            targets = torch.from_numpy(targets).view((-1,))
            epoch_targets.update(zip(self.all_sequences[seq_index], targets.squeeze().tolist()))
            tag_scores = self.module_(seq)
            tag_predicted = torch.argmax(tag_scores, dim=-1)
            epoch_predicted.update(zip(self.all_sequences[seq_index], tag_predicted.squeeze().tolist()))
            loss = self.criterion_(tag_scores, targets)
            losses[i] = loss
            loss_weights[i] = seq.size
            if train_mode:
                loss.backward()
                self.optimizer_.step()
        loss_avg = np.average(losses, weights=loss_weights)
        epoch_targets = list(epoch_targets.values())
        epoch_predicted = list(epoch_predicted.values())
        scores = {
            'f1': sklearn.metrics.f1_score(epoch_targets, epoch_predicted),
            'accuracy': sklearn.metrics.accuracy_score(epoch_targets, epoch_predicted),
            'precision': sklearn.metrics.precision_score(epoch_targets, epoch_predicted),
            'recall': sklearn.metrics.recall_score(epoch_targets, epoch_predicted),
            'prior': sum(epoch_targets) / len(epoch_targets)
        }
        scores['f1_naive'] = scores['prior'] * 2.0 / (1.0 + scores['prior'])
        scores['f1_imprv'] = scores['f1'] / scores['f1_naive']
        scores['p*r'] = scores['precision'] * scores['recall']
        scores['pr_imprv'] = scores['p*r'] / scores['prior']

        prefix = 't.' if train_mode else 'v.'
        self.history.record(prefix+'prior', scores['prior'])
        self.history.record(prefix+'f1_imprv', scores['f1_imprv'])
        self.history.record(prefix+'f1_naive', scores['f1_naive'])
        self.history.record(prefix+'p*r', scores['p*r'])
        self.history.record(prefix+'pr_imprv', scores['pr_imprv'])
        self.history.record(prefix+'f1', scores['f1'])
        self.history.record(prefix+'pr', scores['precision'])
        self.history.record(prefix+'rec', scores['recall'])
        self.history.record(prefix+'acc', scores['accuracy'])
        return {'loss': loss_avg, 'y_pred': epoch_predicted, 'f1': scores['f1'],
                'acc': scores['accuracy'], 'pr': scores['precision'], 'rec': scores['recall']}

    def train_step(self, Xi, yi):
        return self.epoch_step(Xi, train_mode=True)

    def validation_step(self, Xi, yi):
        with torch.no_grad():
            scores = self.epoch_step(Xi, train_mode=False)
        return scores

    def fit(self):
        super().fit(self.train)


class LSTM_SA_REPORTER(skorch.callbacks.Callback):
    def __init__(self, results_path):
        self.results_path = results_path

    def on_train_begin(self, net, **kwargs):
        with open(self.results_path, 'w') as out_file:
            out_file.write('epoch,t.prior,t.f1,t.pr,t.rec,t.acc,t.loss,v.prior,v.f1_naive,v.f1,v.f1_imprv,v.p*r,v.pr_imprv,v.pr,v.rec,v.acc,v.loss,dur\n')

    def on_epoch_end(self, net, **kwargs):
        with open(self.results_path, 'a') as out_file:
            scores = net.history[-1]
            out_file.write(('%d,' + ','.join(['%.4f'] * 17)) % (scores['epoch'], scores['t.prior'], scores['t.f1'],
                                                                scores['t.pr'], scores['t.rec'],
                                                      scores['t.acc'], scores['train_loss'], scores['v.prior'], scores['v.f1_naive'], scores['v.f1'],
                                                      scores['v.f1_imprv'], scores['v.p*r'], scores['v.pr_imprv'],
                                                      scores['v.pr'], scores['v.rec'], scores['v.acc'],
                                                      scores['valid_loss'], scores['dur']))
            out_file.write('\n')
