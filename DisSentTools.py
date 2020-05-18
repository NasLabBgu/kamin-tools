import LabelTools as lt, TreeTools as tt
import numpy as np, math, pandas as pd
import re, csv
import torch
from torch.autograd import Variable
import dill, glob, itertools
from joblib import Parallel, delayed
from collections import Counter, defaultdict

model_path = '/data/work/data/reddit_cmv/dissent/dis-model.pickle'
GLOVE_PATH = '/data/work/data/reddit_cmv/glove/glove.840B.300d.txt'


class DisSentTools():
    dissent = None
    i = 0
    EN_DISCOURSE_MARKERS = [
        "after",
        "also",
        "although",
        "and",
        "as",
        "because",
        "before",
        "but",
        "if",
        "so",
        "still",
        "then",
        "though",
        "when",
        "while"
    ]

    @staticmethod
    def init_dissent(all_trees, model_path, glove_path):
        map_locations = torch.device('cpu')
        dissent = torch.load(model_path, map_location=map_locations)
        dissent.encoder.set_glove_path(glove_path)
        sentences = [sent for tree in all_trees for node in tt.get_nodes(tree) for sent in
                     DisSentTools.get_sentences(node['text'])]
        dissent.encoder.build_vocab(sentences)
        DisSentTools.dissent = dissent

    @staticmethod
    def get_sentences(text):
        text = lt.clean_text(text, do_lower=False)
        sent_split_re = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
        return re.split(sent_split_re, text)

    @staticmethod
    def get_probas_for_tree(tree):
        print('Starting tree:', tree['node']['id'])
        total_nodes = len(tt.get_nodes(tree))
        probas = np.zeros(shape=(total_nodes, 64, 15))
        hasht = {}
        lengths = []
        for i, node in enumerate(tt.get_nodes(tree)):
            hasht[node['id']] = i
            node_pr, node_len = DisSentTools.get_probas(node['text'])
            probas[i] = node_pr
            lengths.append(node_len)
        print('Finished tree:', tree['node']['id'])
        return {'probas': probas, 'hasht': hasht, 'lengths': lengths}

    @staticmethod
    def get_probas(text):
        DisSentTools.i += 1
        if (DisSentTools.i % 10) == 0:
            print(DisSentTools.i)
        probas = np.zeros(shape=(64, 15))
        sentences = DisSentTools.get_sentences(text)
        total = 0
        if len(sentences) > 1:
            if len(sentences) > 65:
                print('Error! Too many sentences', len(sentences))
                sentences = sentences[0:65]
            dissent = DisSentTools.dissent
            s1_raw = sentences[:-1]
            s2_raw = sentences[1:]
            s1_prepared, s1_len = dissent.encoder.prepare_samples(s1_raw, tokenize=True, verbose=False, no_sort=True)
            s2_prepared, s2_len = dissent.encoder.prepare_samples(s2_raw, tokenize=True, verbose=False, no_sort=True)
            b1 = Variable(dissent.encoder.get_batch(s1_prepared, no_sort=True))
            b2 = Variable(dissent.encoder.get_batch(s2_prepared, no_sort=True))
            discourse_preds = dissent((b1, s1_len), (b2, s2_len))
            out_proba = torch.nn.Softmax()(discourse_preds).detach().numpy()
            total = len(s1_raw)
            for i in range(total):
                probas[i] = out_proba[i]
        return probas, total

    @staticmethod
    def run_tree(all_trees, tree, path_to_save='dissent_probas/'):
        DisSentTools.init_dissent(all_trees, model_path=model_path, glove_path=GLOVE_PATH)
        res = DisSentTools.get_probas_for_tree(tree)
        with open(path_to_save + tree['node']['id'] + '.disprobas', 'wb') as dis_probas_file:
            dill.dump({tree['node']['id']: res}, dis_probas_file, dill.HIGHEST_PROTOCOL)

    @staticmethod
    def run_trees(trees_file, path_to_save='dissent_probas/'):
        trees = tt.load_list_of_trees(trees_file)
        Parallel(n_jobs=len(trees))(delayed(DisSentTools.run_tree)(trees, tree, path_to_save) for tree in trees)

    @staticmethod
    def merge_probas(probas_dir='dissent_probas/', out_path='dissent_probas/all_trees.dispr'):
        filenames = [file for file in glob.glob(probas_dir + '*.disprobas')]
        trees_probas = {}
        for file in filenames:
            print('Opening:', file)
            with open(file, 'rb') as f:
                tree_probas = dill.load(f)
                trees_probas.update(tree_probas)
        with open(out_path, 'wb') as f:
            dill.dump(trees_probas, f, dill.HIGHEST_PROTOCOL)
        print('DONE:', len(filenames), 'trees.')
        print('Output saved to:', out_path)

    @staticmethod
    def print_branch(trees_path, probas_path, branch_atlas_id, out_file):
        trees = tt.load_list_of_trees(trees_path)
        with open(probas_path, 'rb') as f:
            probas = dill.load(f)
        tid = branch_atlas_id.split('_')[0]
        tree = None
        for t in trees:
            if t['node']['id'] == tid:
                tree = t
                break;
        if tree is None:
            print('Error! Tree', tid, 'is not found in the trees file', trees_path)
            return
        if tid not in probas:
            print('Error! Tree', tid, 'is not found in the probas file', probas_path)
            return
        br = None
        for branch in tt.get_branches(tree):
            if ('extra_data' in branch[-1] and 'file:line' in branch[-1]['extra_data'] and
                    branch[-1]['extra_data']['file:line'].split(':')[0] == branch_atlas_id):
                br = branch
                break;
        if br is None:
            print('Error! Branch', branch_atlas_id, 'is not found in the trees file', trees_path)
            return
        mypr = probas[tid]
        with open(out_file, 'w', encoding="utf-8", newline='') as csvfile:
            csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            csv_writer.writerow(['TreeID', 'NodeID', 'BranchID', 'Sentence', 'Predicted'] + DisSentTools.EN_DISCOURSE_MARKERS)
            for n in br:
                sentences = DisSentTools.get_sentences(n['text'])
                node_probas = mypr['probas'][mypr['hasht'][n['id']]]
                max_l = mypr['lengths'][mypr['hasht'][n['id']]]
                snt_probas = node_probas[0]
                snt_probas_str = ['%.4f' % elem for elem in snt_probas] if max_l > 0 else []
                marker = DisSentTools.EN_DISCOURSE_MARKERS[np.argmax(snt_probas)] if max_l > 0 else ''
                csv_writer.writerow([tree['node']['id'], n['id'], branch_atlas_id, sentences[0], marker] + snt_probas_str)
                for i in range(1, len(sentences)):
                    snt_probas = node_probas[i]
                    snt_probas_str = ['%.4f' % elem for elem in snt_probas] if i < max_l else []
                    marker = DisSentTools.EN_DISCOURSE_MARKERS[np.argmax(snt_probas)] if i < max_l else ''
                    csv_writer.writerow(['', '', '', sentences[i], marker] + snt_probas_str)
        print('DONE:',out_file)

    @staticmethod
    def get_binaries_for_node(tree_probas, nid):
        res = [0] * 15
        if nid in tree_probas['hasht'] and tree_probas['lengths'][tree_probas['hasht'][nid]] > 0:
            max_l = tree_probas['lengths'][tree_probas['hasht'][nid]]
            node_probas = tree_probas['probas'][tree_probas['hasht'][nid]]
            for i in range(max_l):
                res[np.argmax(node_probas[i])] = 1
        return res

    @staticmethod
    def print_dis_tags_npmi(trees_path, probas_path, out_file, just_count = False, just_pmi = False):
        trees = tt.load_list_of_trees(trees_path)
        with open(probas_path, 'rb') as f:
            probas = dill.load(f)
        label_counts = Counter()
        dis_counts = Counter()
        both_counts = Counter()
        total_labeled_nodes = 0
        for tree in trees:
            tid = tree['node']['id']
            for node in tt.get_nodes(tree):
                nid = node['id']
                labels = lt.flat_labels(node['labels'] if 'labels' in node else None)
                if node['text'] == '[removed]' or node['text'] == '[deleted]' or not labels:
                    continue
                binaries = DisSentTools.get_binaries_for_node(probas[tid], nid)
                dis_connectors = [DisSentTools.EN_DISCOURSE_MARKERS[i] for i in range(15) if binaries[i] == 1]
                if not dis_connectors:
                    continue
                label_counts.update(labels)
                dis_counts.update(dis_connectors)
                both_counts.update(itertools.product(*[labels, dis_connectors]))
                total_labeled_nodes += 1
        pmi_table = defaultdict(lambda: defaultdict(str))
        for label in sorted(label_counts):
            for dis_con in sorted(dis_counts):
                if not just_count:
                    p_x_y = both_counts[(label, dis_con)] / total_labeled_nodes
                    p_x = label_counts[label] / total_labeled_nodes
                    p_y = dis_counts[dis_con] / total_labeled_nodes
                    if p_x_y > 0.0:
                        denom = 1 if just_pmi else -math.log2(p_x_y)
                        pmi_table[label][dis_con] = '{:.3f}'.format(math.log2(p_x_y / (p_x * p_y)) / denom)
                else:
                    pmi_table[label][dis_con] = both_counts[(label, dis_con)]
        df = pd.DataFrame(pmi_table)
        df.to_csv(out_file, na_rep='-∞' if just_pmi else '-1')
        print('DONE: ' + out_file)
