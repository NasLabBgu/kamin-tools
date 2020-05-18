import TreeTools as tt
import os, sys, glob, csv
from collections import defaultdict, Counter
import re, itertools, pandas as pd, math, json, tldextract

from sklearn.metrics import matthews_corrcoef


def print_label_tables(trees, out_path, tagged_branches_path = None):
    """
    Goes over nodes in trees and creates for each branch a tsv file named 'treeid_brnidx_brnlen_tab.txt' with columns
    [node_id, speaker, time_unix, [annotators]], rows are filled with node's info and it's labels per annotator.
    Skips the branches which are not in the tagged_branches_path (if not None)
    If some Node doesn't have labels per certain annotator, it's considered to have a label 'NON' for that annotator.
    :param trees: labeled trees
    :param out_path: the output directory to create files in
    :return: None
    """
    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)
    tgbr = None
    if tagged_branches_path is not None:
        tgbr = find_tagged_branches(tagged_branches_path)
    for tree in trees:
        branches = tt.get_branches(tree)
        for i, branch in enumerate(branches):
            annotators = set()
            for node in branch:
                if 'labels' in node:
                    for annotator in node['labels']:
                        annotators.add(annotator)
            ### ###
            if not annotators or (tgbr and i not in tgbr[tree['node']['id']]):
                continue;
            ### ###
            annotators = sorted(list(annotators))
            table = []
            for node in branch:
                entry = {'id' : node['id'], 'speaker' : node['author'], 'time_unix' : node['created_utc']}
                for annotator in annotators:
                    if 'labels' in node and annotator in node['labels']:
                        entry[annotator] = ','.join(sorted(node['labels'][annotator]))
                    else:
                        entry[annotator] = 'NON'
                table.append(entry)
            out_fn = '%s_%d_%d_tab.txt' % (tree['node']['id'], i, len(branch))
            outfile = open(out_path + out_fn, 'w')
            outfile.write('id\tspeaker\ttime_unix')
            for ann in annotators:
                outfile.write(('\t'+ann).encode('utf-8'))
            outfile.write('\n')
            for entry in table:
                text = '%s\t%s\t%d' % (entry['id'], entry['speaker'], entry['time_unix'])
                for ann in annotators:
                    text += '\t' + entry[ann]
                text += '\n'
                outfile.write(text.encode('utf-8'))
            outfile.close()
            print ('DONE: ' + out_fn)


def flat_labels(labels_dict):
    return [] if not labels_dict else sorted(set([label for ann, labels in labels_dict.items() for label in labels]))


def print_forward_backward_transitions(trees, out_dir, ngrams=[2, 3]):
    forward_dict = defaultdict(lambda: defaultdict(Counter))
    backward_dict = defaultdict(lambda: defaultdict(Counter))

    trees = tt.load_list_of_trees(trees)
    for tree in trees:
        for n in ngrams:
            node_ngrams = get_tree_ngrams(tree, n)
            for nngr in node_ngrams:
                label_ngram_sets = [
                    node['labels']['consolidated'] if 'labels' in node and 'consolidated' in node['labels'] and
                                                      node['labels']['consolidated'] else ['NONE'] for node in nngr]
                label_ngrams = list(itertools.product(*label_ngram_sets))
                # each label_ngram is of size n
                prefixes = [label_ngram[:-1] for label_ngram in label_ngrams]
                suffixes = [label_ngram[-1] for label_ngram in label_ngrams]
                for prefix, suffix in zip(prefixes, suffixes):
                    forward_dict[n][prefix][suffix] += 1
                    backward_dict[n][suffix][prefix] += 1
    os.makedirs(out_dir, exist_ok=True)
    transition_lists = defaultdict(list)
    stringify = lambda t: str(t) if len(t) > 1 else t[0]
    for n in ngrams:
        normalized_forward = defaultdict(lambda: defaultdict(float))
        normalized_backward = defaultdict(lambda: defaultdict(float))
        for k, c in forward_dict[n].items():
            for ck, cv in c.items():
                normalized_forward[stringify(k)][stringify(ck)] = cv / sum(c.values())
        for k, c in backward_dict[n].items():
            for ck, cv in c.items():
                normalized_backward[stringify(k)][stringify(ck)] = cv / sum(c.values())
        # normalized_forward = {str(prefix): {str(suffix): count/sum(forward_dict[n][prefix].values()) for suffix, count in forward_dict[n][prefix].items()} for prefix in forward_dict[n].keys()}
        # normalized_backward = {str(suffix): {str(prefix): count/sum(backward_dict[n][suffix].values()) for prefix, count in backward_dict[n][suffix].items()} for suffix in backward_dict[n].keys()}
        df = pd.DataFrame.from_dict(normalized_forward)
        out_file = out_dir + '/' + str(n) + 'gram_forward.csv'
        df.to_csv(out_file)
        print('DONE: ' + out_file)
        out_file = out_dir + '/' + str(n) + 'gram_backward.csv'
        df = pd.DataFrame.from_dict(normalized_backward)
        df.to_csv(out_file)
        print('DONE: ' + out_file)

        for prefix, suffix_dict in normalized_forward.items():
            for suffix, proba in suffix_dict.items():
                transition_lists[str(n) + 'gram_forward'].append(suffix + '|' + prefix)
                transition_lists[str(n) + 'gram_forward_proba'].append(proba)
        for suffix, prefix_dict in normalized_backward.items():
            for prefix, proba in prefix_dict.items():
                transition_lists[str(n) + 'gram_backward'].append(prefix + '|' + suffix)
                transition_lists[str(n) + 'gram_backward_proba'].append(proba)
    out_file = out_dir + '/transition_lists.csv'
    df = pd.DataFrame.from_dict(transition_lists, orient='index')
    df.transpose().to_csv(out_file)
    print('DONE: ' + out_file)

def print_label_ngram_lists(trees, out_file, ngrams=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], min_count=10):
    """
    Goes over a labeled trees and prints overall label-ngram counts as lists in columns.
    :param trees:
    :param out_file:
    :param ngrams:
    :return:
    """
    trees = tt.load_list_of_trees(trees)
    ngram_counts_all = defaultdict(Counter)
    for tree in trees:
        for n in ngrams:
            ngram_counts_tree = Counter()
            node_ngrams = get_tree_ngrams(tree, n)
            for nngr in node_ngrams:
                label_ngram = [
                    node['labels']['consolidated'] if 'labels' in node and 'consolidated' in node['labels'] and
                                                      node['labels']['consolidated'] else ['NONE'] for node in nngr]
                ngram_counts_all[n].update(itertools.product(*label_ngram))

    ngram_lists = defaultdict(list)
    for n in ngrams:
        for ngram, count in ngram_counts_all[n].most_common():
            if count >= min_count:
                key_n = str(n) + '_ngrams'
                key_c = str(n) + '_ngram_counts'
                ngram_lists[key_n].append(ngram)
                ngram_lists[key_c].append(count)

    df = pd.DataFrame.from_dict(ngram_lists, orient='index')
    df.transpose().to_csv(out_file)
    print('DONE: ' + out_file)

def print_label_ngrams(trees, out_dir, ngrams = [2,3]):
    """
    Goes over a labeled trees and prints per-tree & overall label-ngram counts.
    :param trees:
    :param out_dir:
    :param ngrams:
    :return:
    """
    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ngram_counts_all = defaultdict(Counter)
    log_files = {n : open(out_dir + '/' + 'ngrams_log_' + str(n)+'.txt', 'w') for n in ngrams}
    [log_files[n].write(("Label%d\t"*n + "Begin:File:Line\t" + "End:File:Line\n") % tuple(range(1, n+1))) for n in ngrams]
    for tree in trees:
        for n in ngrams:
            ngram_counts_tree = Counter()
            node_ngrams = get_tree_ngrams(tree, n)
            for nngr in node_ngrams:
                label_ngram = [flat_labels(node['labels']) if 'labels' in node else ['NONE'] for node in nngr]
                ngram_counts_tree.update(itertools.product(*label_ngram))
                ngram_counts_all[n].update(itertools.product(*label_ngram))
                begin_file_line = nngr[0]['extra_data']['file:line'] if 'file:line' in nngr[0]['extra_data'] else ''
                end_file_line = nngr[-1]['extra_data']['file:line'] if 'file:line' in nngr[-1]['extra_data'] else ''
                for label_seq in itertools.product(*label_ngram):
                    log_files[n].write(('%s\t'*n + '%s\t%s\n') % (label_seq + (begin_file_line,end_file_line)))
            out_fn = '%s_%d_%dgram.txt' % (tree['node']['id'], len(node_ngrams), n)
            outfile = open(out_dir + '/' + out_fn, 'w')
            for key, value in ngram_counts_tree.most_common():
                text = '%s\t%d\n' % ('%s' % '\t'.join(key), value)
                outfile.write(text)
            outfile.close()
            print ('DONE: ' + out_fn)
    [f.close() for f in log_files.values()]
    for n in ngrams:
        out_fn = 'alltrees_%d_%dgram.txt' % (len(ngram_counts_all[n]), n)
        outfile = open(out_dir + '/' + out_fn, 'w')
        for key, value in ngram_counts_all[n].most_common():
            text = '%s\t%d\n' % ('%s' % '\t'.join(key), value)
            outfile.write(text)
        outfile.close()
        print ('DONE: ' + out_fn)


def print_detailed_label_ngrams(trees, out_path, ngrams = [2,3], tagged_branches_path = None):
    """
    Goes over labeled trees and counts label n-grams for each branch, for each annotator. Then prints the n-grams
    into a file named 'treeid_brnidx_brnlen_annotatorname_Ngram.txt', where N is an integer.
    A combined ngram count for all trees, and per tree, is also created.
    Branches that are not in tagged_branches_path directory are not processed (if not None).
    N-grams are the consequent series of N labels for every N consequent nodes in branch, per each annotator.
    If some Node doesn't have labels per certain annotator, it's considered to have a label 'NON' for that annotator.
    :param trees: list of labeled trees
    :param out_path: path to directory to save the produced files in
    :param ngrams: list of ints (n-grams). Default: [2, 3]
    :param tagged_branches_path: directory of the tagged bundle to check if we should process or skip a certain branch
    :return: None
    """
    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)
    tgbr = None
    if tagged_branches_path is not None:
        tgbr = find_tagged_branches(tagged_branches_path)
    for tree in trees:
        branches = tt.get_branches(tree)
        for i, branch in enumerate(branches):
            branch_all_labels_list = []
            annotators = set()
            for node in branch:
                if 'labels' in node:
                    labels = flat_labels(node['labels']) or ['NONE']
                    branch_all_labels_list.add(sorted(labels))
                    for annotator in node['labels']:
                        annotators.add(annotator)

            ### ###
            if not annotators or (tgbr and i not in tgbr[tree['node']['id']]):
                continue;
            ### ###
            annotators = sorted(list(annotators))
            for annotator in annotators:
                label_list = []
                for node in branch:
                    if 'labels' in node and annotator in node['labels']:
                        label_list.append(sorted(node['labels'][annotator]))
                    else:
                        label_list.append(['NON'])
                for ngram in ngrams:
                    ngram_counts = {}
                    for counter in range(ngram,len(branch)+1):
                        result = list(itertools.product(*label_list[counter-ngram:counter]))
                        for item in result:
                            key = ','.join(item)
                            if key not in ngram_counts:
                                ngram_counts[key] = 0
                            ngram_counts[key] += 1
                    out_fn = '%s_%d_%d_%s_%dgram.txt' % (tree['node']['id'], i, len(branch), annotator, ngram)
                    outfile = open(out_path + out_fn, 'w')
                    for key in ngram_counts:
                        text = '%s\t%d\n' % (key, ngram_counts[key])
                        outfile.write(text.encode('utf-8'))
                    outfile.close()
                    print ('DONE: ' + out_fn)


def count_label_subsets(trees):
    """
    :param trees: List of trees to count labels for.
    :return: Countings of each subset of labels that we encounter in trees. Both Total and PerUser.
                {"total" : {"Stringified_Subset" : count}, "per_user" : {User1 : {"Stringified_Subset"} : count}}
    """
    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)
    total = 'total'
    per_user = 'per_user'
    counts = {total : {}, per_user : {}}
    stringify_labels = lambda list_of_labels : '_'.join(sorted(list_of_labels))
    for tree in trees:
        branches = tt.get_branches(tree)
        visited_ids = set()
        for branch in branches:
            for node in branch:
                if node['id'] in visited_ids or 'labels' not in node :
                    continue
                visited_ids.add(node['id'])
                labels = flat_labels(node['labels'])
                subs = subsets(labels)
                subs.remove([])
                for subset in subs:
                    stringified = stringify_labels(subset)
                    if stringified not in counts[total]:
                        counts[total][stringified] = 0
                    counts[total][stringified] += 1
                    user = node['author']
                    if user not in counts[per_user]:
                        counts[per_user][user] = {}
                    if stringified not in counts[per_user][user]:
                        counts[per_user][user][stringified] = 0
                    counts[per_user][user][stringified] += 1
    return counts


global_subsets = dict()


def subsets(input_list, wanted_type='tuple', k=-1):
    """
    Creates a list of all possible sorted subsets up to size k for a list. Each subset is a list itself.
    :param input_list: input list of elements
    :return: list of lists for all possible subsets all elements
    """
    if k == -1:
        k = len(input_list)
    global global_subsets
    if tuple(input_list) in global_subsets:
        return global_subsets[tuple(input_list)]
    subs = []
    min_runs = min(len(input_list), k)
    for i in range(k):
        subs.extend(itertools.combinations(input_list, i))
    if wanted_type == 'list':
        subs = [list(s) for s in subs]
    global_subsets[tuple(input_list)] = subs
    return subs


def find_tagged_branches(tagged_bundles_dir):
    """
    Goes over a given directory and figures out from the filenames which branches are supposed to be tagged.
    :param tagged_bundles_dir: Directory path with tagged branches
    :return: dict {tree_id : set(tagged_branches_idxes)}
    """
    tagged_branches = os.listdir(tagged_bundles_dir)
    tgbr = defaultdict(set)
    for br in tagged_branches:
        tree_id = br.split('_')[0]
        brn_idx = int(br.split('_')[1])
        tgbr[tree_id].add(brn_idx)
    return tgbr


def get_tree_ngrams(tree, ngram = 2):
    """
    Goes over a tree and return all it's distinct ngrams, in term of nodes.
    :param tree:
    :return: list of lists : each inner list is ngram of nodes
    """
    visited_nodes = set()
    nodes_ngrams = []
    for branch in tt.get_branches(tree):
        reversed_branch = branch[::-1]
        for rev_idx, node in enumerate(reversed_branch):
            orig_idx = len(branch) - rev_idx - 1
            if node['id'] in visited_nodes:
                continue;
            visited_nodes.add(node['id'])
            if orig_idx < ngram-1:  # if we don't have enough elements
                break;
            nodes_ngrams.append(branch[orig_idx-ngram+1:orig_idx+1])
    return nodes_ngrams


def clean_text(text, do_lower = True):
    if type(text) is not str:
        text = ''
    if text == '[deleted]' or text == '[removed]':
        text = ''
    deltabot_re = re.compile(r'^Confirmed: \d+ delta awarded to .*', re.DOTALL)
    if deltabot_re.match(text):
        text = ''
    if do_lower:
        text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    mentions_re = re.compile(r'/u/\w*', re.MULTILINE)
    quote_re = re.compile(r'<quote>.[^<]*</quote>', re.MULTILINE)
    url_re = re.compile(r'http://[^\s]*', re.MULTILINE)
    for m in mentions_re.findall(text):
        text = text.replace(m, '_mention_')
    for q in quote_re.findall(text):
        text = text.replace(q, '_quote_')
    for url in url_re.findall(text):
        text = '_url_' + text.replace(url, tldextract.extract(url).domain)
    #text = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', " ", text)
    #text = re.sub(r'<quote>.[^<]*</quote>', "_quote_", text)
    #text = re.sub('\W', ' ', text)
    text = re.sub(r'(\w)(\W)', r'\1 \2', text)
    text = re.sub(r'(\W)(\w)', r'\1 \2', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


def print_liwc_subset_counts(trees, outfile, nodes_threshold=10):
    """
    Goes over a list of trees and prints LIWC categories subsets counts.
    :param trees: trees
    :param nodes_threshold: minimum number of nodes in order for a tree to be processed
    :param outfile: out filename
    :return: None
    """
    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)
    assert sys.version_info >= (3, 2)  # Python 3.2+ only!
    subset_counts = Counter()
    for tree in trees:
        if len(tt.get_nodes(tree)) < nodes_threshold:  # don't count trees with less than 10 nodes
            continue
        for node in tt.get_nodes(tree):
            cats = sorted(list(get_liwc_cats(node['text'], node['id'])))
            subset_counts.update(subsets(cats, wanted_type='tuple', k = 7))

    with open(outfile, 'w') as out:
        for key, value in subset_counts.most_common():
            s = str(value)
            s += '\t' + '\t'.join(key) + '\n'
            out.write(s)

    print('DONE: ' + outfile)


def print_liwc_pmi_list(trees, outfile, nodes_threshold=10):
    """
    Goes over a list of trees and prints LIWC bitwise-categories PMI table.
    :param trees: trees
    :param nodes_threshold: minimum number of nodes in order for a tree to be processed
    :param outfile: out filename
    :return: None
    """
    assert sys.version_info >= (3, 2)  # Python 3.2+ only!
    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)
    total_nodes = 0
    single_cat_counts = Counter()
    double_cat_counts = Counter()
    for tree in trees:
        if len(tt.get_nodes(tree)) < nodes_threshold:  # don't count trees with less than 10 nodes
            continue
        for node in tt.get_nodes(tree):
            cats = list(get_liwc_cats(node['text'], node['id']))
            if not cats:
                continue
            single_cat_counts.update(cats)
            double_cat_counts.update(itertools.product(*[cats, cats]))  # PYTHON 3.2+ ONLY!
            total_nodes+=1

    print('Total Nodes = %d' % total_nodes)
    with open(outfile, 'w') as out:
        for i, cat_x in enumerate(sorted(single_cat_counts)):
            for cat_y in sorted(single_cat_counts)[i::]:
                p_x_y = double_cat_counts[(cat_x, cat_y)] / total_nodes
                p_x = single_cat_counts[cat_x] / total_nodes
                p_y = single_cat_counts[cat_y] / total_nodes
                if p_x_y > 0.0:
                    out.write(cat_x + '\t' + cat_y + '\t' + '{:.3f}'.format(math.log2(p_x_y / (p_x*p_y))) + '\n')
    print('DONE: ' + outfile)


def print_tags_npmi_table(trees, out_file, just_count = False, just_pmi = False, log_file = None, return_dont_print = False):
    """
    Goes over a list of trees and prints Tags bitwise PMI table.
    If just_count == True - only counts are made.
    If log_file specified - the list of all tag co-occurences will be created
    :param trees: trees
    :param outfile: out filename
    :return: None
    """
    assert sys.version_info >= (3, 2)  # Python 3.2+ only!

    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)

    total_labeled_nodes = 0
    single_label_counts = Counter()
    double_label_counts = Counter()
    if log_file:
        logf = open(log_file, mode='w')
        logf.write("Label1\tLabel2\tFile:Line\n")
    for tree in trees:
        #print("TreeID %s has %d branches" % (tree['node']['id'], len(tt.get_branches(tree))))
        for node in tt.get_nodes(tree):
            labels = flat_labels(node['labels'] if 'labels' in node else None)
            if node['text'] == '[removed]' or node['text'] == '[deleted]' or not labels:
                    continue
            single_label_counts.update(labels)
            double_label_counts.update(itertools.product(*[labels, labels]))  # PYTHON 3.2+ ONLY!
            if log_file:
                file_line = node['extra_data']['file:line'] if 'file:line' in node['extra_data'] else ''
                for (lab_x, lab_y) in itertools.product(*[labels, labels]):
                    if lab_x != lab_y:
                        logf.write('%s\t%s\t%s\n' % (lab_x, lab_y, file_line))
            total_labeled_nodes += 1
    labels_pmi = defaultdict(lambda: defaultdict(str))
    labels_pmi_float = defaultdict(lambda: defaultdict(float))
    for label_x in sorted(single_label_counts):
        for label_y in sorted(single_label_counts):
            if not just_count:
                p_x_y = double_label_counts[(label_x, label_y)] / total_labeled_nodes
                p_x = single_label_counts[label_x] / total_labeled_nodes
                p_y = single_label_counts[label_y] / total_labeled_nodes
                if p_x_y > 0.0:
                    denom = 1 if just_pmi else -math.log2(p_x_y)
                    labels_pmi[label_x][label_y] = '{:.3f}'.format(math.log2(p_x_y / (p_x * p_y)) / denom)
                    labels_pmi_float[label_x][label_y] = math.log2(p_x_y / (p_x * p_y)) / denom
            else:
                labels_pmi[label_x][label_y] = double_label_counts[(label_x, label_y)]
                labels_pmi_float[label_x][label_y] = double_label_counts[(label_x, label_y)]
    if return_dont_print:
        return labels_pmi_float
    df = pd.DataFrame(labels_pmi)
    df.to_csv(out_file, na_rep='-∞' if just_pmi else '-1' )
    print('DONE: ' + out_file)


def print_tags_matthews(trees, out_file):
    trees = tt.load_list_of_trees(trees)
    all_labels = set()
    for tree in trees:
        for node in tt.get_nodes(tree):
            if 'labels' in node and 'consolidated' in node['labels']:
                all_labels.update(node['labels']['consolidated'])
    all_labels =  sorted(all_labels)
    labels_data = defaultdict(list)
    for tree in trees:
        for node in tt.get_nodes(tree):
            for label in all_labels:
                if 'labels' in node and 'consolidated' in node['labels'] and label in node['labels']['consolidated']:
                    labels_data[label].append(1)
                else:
                    labels_data[label].append(0)
    matthews_table = defaultdict(dict)
    for label1, label2 in itertools.product(all_labels, repeat=2):
        matthews_table[label1][label2] = matthews_corrcoef(labels_data[label1], labels_data[label2])
    df = pd.DataFrame(matthews_table)
    df.to_csv(out_file)
    print('DONE: ' + out_file)


def apply_split_labels_to_trees(trees, out_trees):
    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)
    for tree in trees:
        for full_node in tt.get_full_nodes(tree):
            if len(full_node['children']) >= 2:
                # we are a split
                if 'labels' not in full_node['node']:
                    full_node['node']['labels'] = {}
                if 'consolidated' in full_node['node']['labels']:
                    full_node['node']['labels']['consolidated'].append('_SPLIT_')
                else:
                    full_node['node']['labels']['consolidated'] = ['_SPLIT_']
                if 'extra_data' not in full_node['node']:
                    full_node['node']['extra_data'] = {}
                if 'full_labels' not in full_node['node']['extra_data']:
                    full_node['node']['extra_data']['full_labels'] = []
                full_node['node']['extra_data']['full_labels'].append(('_SPLIT_', 'AUTO', full_node['node']['text']))
    tt.save_list_of_trees(trees, out_trees)


def apply_start_split_end_labels_to_trees(trees, out_trees):
    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)
    new_trees = []
    for tree in trees:
        start_node = {'node': {'id' : tree['node']['id']+'_START',
                               'text' : '',
                               'timestamp': '',
                               'author': '',
                               'labels' : {'consolidated' : ['_START_']},
                               'extra_data' : {
                                   'full_labels': [('_START_', 'AUTO' , '')],
                                   'file:line': tree['node']['extra_data']['file:line'] if 'file:line' in tree['node']['extra_data'] else ''
                               }
                               },
                      'children': [tree]}
        new_trees.append(start_node)
        for full_node in tt.get_full_nodes(tree):
            if len(full_node['children']) == 0:
                # we are a leaf
                end_node = {'node': {'id' : full_node['node']['id']+'_END',
                                     'text' : '',
                                     'timestamp': '',
                                     'author': '',
                                     'labels' : {'consolidated' : ['_END_']},
                                     'extra_data': {
                                       'full_labels': [('_END_', 'AUTO', '')],
                                       'file:line': full_node['node']['extra_data']['file:line'] if 'file:line' in full_node['node']['extra_data'] else ''
                                   }
                            },
                            'children': []}
                full_node['children'] = [end_node]
            if len(full_node['children']) >= 2:
                # we are a split
                if 'labels' not in full_node['node']:
                    full_node['node']['labels'] = {}
                if 'consolidated' in full_node['node']['labels']:
                    full_node['node']['labels']['consolidated'].append('_SPLIT_')
                else:
                    full_node['node']['labels']['consolidated'] = ['_SPLIT_']
                if 'extra_data' not in full_node['node']:
                    full_node['node']['extra_data'] = {}
                if 'full_labels' not in full_node['node']['extra_data']:
                    full_node['node']['extra_data']['full_labels'] = []
                full_node['node']['extra_data']['full_labels'].append(('_SPLIT_', 'AUTO', full_node['node']['text']))
    tt.save_list_of_trees(new_trees, out_trees)


def remove_2nd_tags(trees_path, except_for=['CBE']):
    trees = tt.load_list_of_trees(trees_path)
    _2nd_labels = set()
    for tree in trees:
        for node in tt.get_nodes(tree):
            if 'labels' in node and 'consolidated' in node['labels']:
                for label in node['labels']['consolidated']:
                    if label.endswith('_2') or label.endswith('_3') or label.endswith('_4'):
                        good = False;
                        for exc in except_for:
                            if label.startswith(exc):
                                good = True;
                                break;
                        if not good:
                            _2nd_labels.add(label)
    settings = {"dont_consolidate": [],
                "ignore_labels": sorted(_2nd_labels),
                "merge_labels": []}
    rework_labels(trees_path, settings, trees_path)


def print_liwc_ngram_counts(trees, outfile, ngram=2, nodes_threshold=10):
    """
    Goes over the given trees and counts which LIWC categories tend to go one after another. Prints to the output file.
    :param trees: trees
    :param nodes_threshold: minimum number of nodes in order for a tree to be processed
    :param ngram: n
    :param outfile: outfile
    :return: None
    """
    assert sys.version_info >= (3, 2) # Python 3.2+ only!
    cat_ngram_counts = Counter()
    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)
    for tree in trees:
        if len(tt.get_nodes(tree)) < nodes_threshold:  # don't count trees with less than 10 nodes
            continue
        node_ngrams = get_tree_ngrams(tree, ngram=ngram)
        for node_ngram in node_ngrams:
            cats_lists = []
            for node in node_ngram:
                cats = list(get_liwc_cats(node['text'], node['id']))
                if not cats:
                    cats.append('EMPTY_CAT')
                cats_lists.append(cats)
            cat_ngram_counts.update(itertools.product(*cats_lists)) # PYTHON 3.2+ ONLY!

    with open(outfile, 'w') as out:
        for key, value in cat_ngram_counts.most_common():
            s = '\t'.join(key)
            s += '\t' + str(value) + '\n'
            out.write(s)
    print('DONE: ' + outfile)


def print_label_priors(trees, outfile, per_tree = False):
    """
    Goes over labeled trees and estimates prior distribution of each label
    :param trees: labeled trees
    :param outfile: output csv file with result table
    :return: None
    """
    assert sys.version_info >= (3, 2)  # Python 3.2+ only!

    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)

    if not per_tree:
        total_nodes = 0
        single_label_counts = Counter()
        full_label_counts = Counter()
        for tree in trees:
            print("TreeID %s has %d branches" % (tree['node']['id'], len(tt.get_branches(tree))))
            for node in tt.get_nodes(tree):
                labels = flat_labels(node['labels'] if 'labels' in node else None)
                if 'full_labels' in node['extra_data']:
                    full_label_counts.update(set([tag for tag, ann, _ in node['extra_data']['full_labels']]))
                if node['text'] == '[removed]' or node['text'] == '[deleted]':
                    labels.append('deleted_text')
                else:
                    if not labels:
                        labels = ['no_labels']
                single_label_counts.update(labels)
                total_nodes += 1

        label_priors = defaultdict(dict)
        for label, count in single_label_counts.most_common():
            label_priors["Counts"][label] = count
            label_priors["Percentage"][label] = "%05.2f%%" % (count * 100 / total_nodes)
            if label in full_label_counts:
                label_priors["Agreement"][label] = "%05.2f%%" % (count * 100 / full_label_counts[label])
            else:
                label_priors["Agreement"][label] = '-'

        label_priors["Counts"]["total_nodes"] = total_nodes
        label_priors["Percentage"]["total_nodes"] = '100.00%'
        consolidated_counts = sum([single_label_counts[tag] for tag in full_label_counts])
        full_counts = sum(full_label_counts.values())
        label_priors["Agreement"]["total_nodes"] = "%05.2f%%" % (consolidated_counts * 100 / full_counts)

        label_priors["Counts"]["labeled_nodes"] = total_nodes - single_label_counts["no_labels"]
        label_priors["Percentage"]["labeled_nodes"] = '%05.2f%%' % \
                                                         ((total_nodes - single_label_counts["no_labels"] -
                                                           single_label_counts["deleted_text"]) * 100 /
                                                          total_nodes)
        label_priors["Agreement"]["labeled_nodes"] = "%05.2f%%" % (consolidated_counts * 100 / full_counts)
    else:
        all_total_nodes = defaultdict(int)
        all_single_label_counts = defaultdict(lambda: Counter())
        all_full_label_counts = defaultdict(lambda: Counter())
        all_branches_counts = defaultdict(int)
        for tree in trees:
            single_label_counts = all_single_label_counts[tree['node']['id']]
            full_label_counts = all_full_label_counts[tree['node']['id']]
            all_branches_counts[tree['node']['id']] = len(tt.get_branches(tree))
            print("TreeID %s has %d branches" % (tree['node']['id'], len(tt.get_branches(tree))))
            for node in tt.get_nodes(tree):
                labels = flat_labels(node['labels'] if 'labels' in node else None)
                if node['text'] == '[removed]' or node['text'] == '[deleted]':
                    labels.append('deleted_text')
                else:
                    if not labels:
                        labels = ['no_labels']
                if 'full_labels' in node['extra_data']:
                    full_label_counts.update(set([tag for tag, ann, _ in node['extra_data']['full_labels']]))
                single_label_counts.update(labels)
                all_total_nodes[tree['node']['id']] += 1

        label_priors = defaultdict(dict)

        ttotal_nodes = 0
        tlabeled_nodes = 0
        tbranches = 0
        texcluding = 0
        tper_label = defaultdict(int)
        for tree_id, single_label_counts in all_single_label_counts.items():
            total_nodes = all_total_nodes[tree_id]
            for label, count in single_label_counts.most_common():
                label_priors[tree_id][label] = count
                tper_label[label] += count
            label_priors[tree_id]["total_nodes"] = total_nodes
            label_priors[tree_id]["excluding_deleted_text"] = total_nodes - single_label_counts["deleted_text"]
            label_priors[tree_id]["labeled_nodes"] = total_nodes - single_label_counts["no_labels"]
            label_priors[tree_id]["branches_count"] = all_branches_counts[tree_id]

            ttotal_nodes += total_nodes
            tbranches += all_branches_counts[tree_id]
            texcluding += (total_nodes - single_label_counts["deleted_text"])
            tlabeled_nodes += (total_nodes - single_label_counts["no_labels"])

        for label, count in tper_label.items():
            label_priors["Total"][label] = count
        label_priors["Total"]["total_nodes"] = ttotal_nodes
        label_priors["Total"]["excluding_deleted_text"] = texcluding
        label_priors["Total"]["labeled_nodes"] = tlabeled_nodes
        label_priors[tree_id]["branches_count"] = tbranches

    df = pd.DataFrame(label_priors)
    df.to_csv(outfile, na_rep='0')
    print('DONE: ' + outfile)


def print_liwc_priors(trees, outfile, nodes_threshold=10):
    """
    Goes over the given trees and estimates distribution of each LIWC category
    :param trees: trees
    :param nodes_threshold: minimum number of nodes in order for a tree to be processed
    :param outfile: outfile
    :return: None
    """
    assert sys.version_info >= (3, 2)  # Python 3.2+ only!

    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)

    total_nodes = 0
    single_cat_counts = Counter()
    for tree in trees:
        if len(tt.get_nodes(tree)) < nodes_threshold:  # don't count trees with less than 10 nodes
            continue
        for node in tt.get_nodes(tree):
            cats = list(get_liwc_cats(node['text'], node['id']))
            if not cats:
                cats = ['empty_cat']
            single_cat_counts.update(cats)
            total_nodes+=1

    cat_priors = defaultdict(dict)
    for cat_x, count in single_cat_counts.most_common():
        cat_priors["Total"][cat_x] = count
        cat_priors["% excluding empty"][cat_x] = "%05.2f%%" % \
                                                 (count * 100 / (total_nodes - single_cat_counts["empty_cat"]))
        cat_priors["% including empty"][cat_x] = "%05.2f%%" % (count * 100 / total_nodes)

    cat_priors["Total"]["total_nodes"] = total_nodes
    cat_priors["% excluding empty"]["total_nodes"] = '%05.2f%%' % \
                                                  (total_nodes * 100 / (total_nodes - single_cat_counts["empty_cat"]))
    cat_priors["% including empty"]["total_nodes"] = '100.00%'

    df = pd.DataFrame(cat_priors)
    df.to_csv(outfile)
    print('DONE: ' + outfile)

'''
def load_category_dict(fn="LIWC_Features.txt"):
    fin = open(fn)
    lines = fin.readlines()
    fin.close()
    liwc_cat_dict = {}  # {cat: (w1,w2,w3,...)}
    for line in lines[1:]:  # first line is a comment about the use of *
        tokens = line.strip().lower().split(', ')
        liwc_cat_dict[tokens[0]] = tokens[1:]
    return liwc_cat_dict


def create_cat_regex_dict(liwc_cat_dict):
    cat_regex_dict = {}
    for k, v in liwc_cat_dict.items():
        str = '|'.join(v)
        str = re.sub(r'\*', r'\\w*', str)
        cat_regex_dict[k] = re.compile(r'\b(' + str + r')\b')
    return cat_regex_dict

#re_test = re.compile(r'\b(ab\w*)\b')

#print(re_test.findall("abc123"))

liwc_cats = {}
liwc_regex_dict = create_cat_regex_dict(load_category_dict())
def get_liwc_cats(text, id, only_search = True):
    global liwc_cats
    if id in liwc_cats:
        return liwc_cats[id]
    global liwc_regex_dict
    if only_search:
        cats = [cat for cat, regex in liwc_regex_dict.items() if regex.search(clean_text(text))]
    else:
        cats = {cat: len(regex.findall(text.lower())) for cat, regex
                in liwc_regex_dict.items() if regex.findall(text.lower())}

    liwc_cats[id] = cats
    if len(liwc_cats) % 1000 == 0:
        print(len(liwc_cats))
    return cats
'''

def print_label_details(trees, out_file):
    """
    Prints all the labels info for the given trees
    :param trees: labeled trees
    :param out_file: out csv file
    :return:
    """
    if type(trees) is str:
        trees = tt.load_list_of_trees(trees)
    annotators = set()
    for node in [node for tree in trees for node in tt.get_nodes(tree)]:
        annotators.update([ann for tag, ann, _ in node['extra_data']['full_labels']]
                          if 'full_labels' in node['extra_data'] else [])
    annotators = sorted(annotators)

    with open(out_file, 'w', encoding="utf-8", newline='') as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        csv_writer.writerow(["tree_id", "node_id", "file:line", "text", "agreement", "consolidated"]+annotators)
        for tree in trees:
            for node in tt.get_nodes(tree):
                # get the frequency of the most common 'set of labels' among different annotators
                if 'labels' in node:
                    agreement = 0
                    consolidated = ''
                    if 'consolidated' in node['labels']:
                        agreement = len(node['labels']['consolidated'])
                        consolidated = ','.join(sorted(node['labels']['consolidated']))
                    per_ann = defaultdict(list)
                    if 'full_labels' in node['extra_data']:
                        for tag, ann in node['extra_data']['full_labels']:
                            per_ann[ann].append(tag)
                        full_tags = set([tag for tag, ann, _ in node['extra_data']['full_labels']])
                        agreement = agreement / len(full_tags) if len(full_tags) > 0 else ''
                    else:
                        agreement = ''
                    full_entries = [','.join(sorted(per_ann[ann])) for ann in annotators]
                    file_line = node['extra_data']['file:line'] if 'file:line' in node['extra_data'] else ''
                    entries = [tree['node']['id']] + [node['id']] + [file_line] + [node['text']] + [agreement]
                    entries += [consolidated] + full_entries
                    csv_writer.writerow(entries)
                else:
                    entries = [tree['node']['id']] + [node['id']] + [''] + [node['text']]
                    csv_writer.writerow(entries)
        print('DONE: ' + out_file)


def cut_non_labeled_branches(labeled_trees_path, out_cut_trees_path):
    """
    Goes over labeled trees and cuts out the unlabeled branches (only the exclusive parts).
    :param labeled_trees_path: input path to labeled trees file
    :param out_cut_trees_path: output trees will be saved here
    :return: None
    """
    in_trees = tt.load_list_of_trees(labeled_trees_path)
    for tree in in_trees:
        full_branches = tt._get_full_branches(tree)
        for br_idx, full_branch in enumerate(full_branches):
            for nd_idx, full_node in enumerate(full_branch):
                exclusive = True
                for br_idx2, full_branch2 in enumerate(full_branches):
                    # means this node appears in some other branch also
                    if full_node in full_branch2 and br_idx != br_idx2:
                        exclusive = False
                        break
                if exclusive:
                    labeled = False
                    for exclusive_node in full_branch[nd_idx::]:
                        if 'labels' in exclusive_node['node'] and \
                                len(flat_labels(exclusive_node['node']['labels'])) > 0:
                            labeled = True
                            break
                    if not labeled:
                        if nd_idx > 0:
                            # cut the exclusive part of non-labeled branch
                            full_branch[nd_idx-1]['children'].remove(full_node)
                        break
    with open(out_cut_trees_path, 'w+') as out_file:
        for tree in in_trees:
            json.dump(tree, out_file)
            out_file.write('\n')
        print('DONE: ' + out_cut_trees_path)


def add_labels(tree, tagged_file_path):
    """
    Adds all labels from tagged file to a given tree (first ensures tagged filename starts with tree id).
    If the tree already contains labels, only new labels will be added to it.

    :param tree: tree
    :param tagged_file_path: a tagged file from bundle. Name convention: treeID_branchIdx_branchLength_tagged.txt
    :return: True if tagged file contained any tags and tree was modified (any labels added)
    """

    # check that filename starts with tree-id
    if tree['node']['id'] != os.path.basename(tagged_file_path).split('_')[0]:
        return
    # get the branch index from filename
    branch_idx = int(os.path.basename(tagged_file_path).split('_')[1])
    flag = False
    with open(tagged_file_path) as tagged_file:
        #print(tagged_file_path)
        for (j, line) in enumerate(tagged_file):
            node = tt.get_node(tree, branch_idx, j)
            labeling_data = get_labeling_data_for_line(line)
            text_node = re.sub(r'[^\x00-\x7F]+','', node['text']).strip()
            text_line = re.sub(r'[^\x00-\x7F]+','', labeling_data['text']).strip()
            equal_texts = text_node == text_line
            if not equal_texts:
                # some bundle-lines end with '\t timestamp'. Let's see, may be that's the case here.
                text_line = re.sub(r'\t\d+\Z', '', text_line)
                equal_texts = text_node.strip() == text_line.strip()
            if not equal_texts or node['author'] != labeling_data['author']:
                raise Exception(f'Text in {tagged_file_path}:{j} (author {labeling_data["author"]}) does not match the text in node '
                                f'{node["id"]} (author {node["author"]}) for tree {tree["node"]["id"]}. The tree structure must have changed, such that '
                                f'the tree in given all_list_of_trees is not the exact same tree that was used to'
                                f'produce the bundle files.')
            if labeling_data['found_at_least_one'] == True:
                if 'extra_data' not in node:
                    node['extra_data'] = {}
                node['labels'] = labeling_data['labels']
                node['extra_data']['full_labels'] = labeling_data['full_labels']
                flag = True
            tagged_file_path = os.path.basename(tagged_file_path)
            if 'extra_data' not in node:
                node['extra_data'] = {}
            if 'file:line' not in node['extra_data']:
                node['extra_data']['file:line'] = tagged_file_path.replace('_tagged.txt', '') + ':' + str(j)
    return flag


def get_labeling_data_for_line(line, dont_consolidate='dont_consolidate.txt'):
    """
    Extracts Labels and Annotators from Line, and return Consolidated La
    :param line:
    :return:
    """
    dont_consolidate = 'dont_consolidate.txt'
    with open(dont_consolidate) as dont_consolidate_file:
        dont_consolidate = eval(dont_consolidate_file.read())
    tag_re = re.compile(r'<(/?\w+)[^<>]*owner=\'([^\']+)\'>')
    res = {'labels': {}, 'full_labels': {}, 'text': '', 'author' : '', 'found_at_least_one': False}
    just_text = re.sub(tag_re, '', line)
    author_re = re.compile(r'^\d+\.\s(.*?)\s:>>>')
    if (author_re.match(just_text) is not None):
        res['author'] = author_re.findall(just_text)[0]
    else:
        author_re = re.compile(r'^\d+\.\s(.*?):\s>>>')
        if (author_re.match(just_text) is not None):
            res['author'] = author_re.findall(just_text)[0]
    just_text = re.sub(author_re, '', just_text)
    #print(just_text)
    res['text'] = just_text
    found_tags = tag_re.findall(line)
    grained_texts = []
    for tag, owner in found_tags:
        if tag[0] != '/':
            textre = f'<{tag}[^<>]*owner=\'{owner}\'>(.*)</{tag}[^<>]*owner=\'{owner}\'>'
            texts = re.compile(textre).findall(line)
            texts = [re.sub(tag_re, '', text) for text in texts]
            texts = [re.sub(author_re, '', text) for text in texts]
            grained_texts.extend([(tag.upper(), owner, text) for text in texts])
        else:
            grained_texts.append((tag.upper(), owner, ''))
    #print(grained_texts)
    #print(found_tags)
    just_tags = [tag for tag, _, __ in grained_texts]
    non_overlapping_tags = []
    k_tags = Counter(set(just_tags))
    #print(k_tags)
    prev_tags = defaultdict(int)  # this
    for i, (tag, ann, text) in enumerate(grained_texts):
        if not tag.startswith('/'):
            if '/' + tag in just_tags[prev_tags[tag]:i]:
                # we have closed this tag earlier, and now it gets open again - meaning it's 'tag_K'
                #print(k_tags[tag])
                k_tags[tag] += 1
                prev_tags[tag] = i
            if k_tags[tag] > 1:
                non_overlapping_tags.append((tag + '_' + str(k_tags[tag]), ann, text))
            else:
                non_overlapping_tags.append((tag, ann, text))
            res['found_at_least_one'] = True
        else:
            # we are the closing tag
            None
    #print(non_overlapping_tags)
    res['full_labels'] = non_overlapping_tags
    consolidated_labels = [tag for tag, ann, text in non_overlapping_tags]
    consolidated_labels = list(
        set([tag for tag in consolidated_labels if consolidated_labels.count(tag) >= 2 or tag in dont_consolidate]))
    res['labels'] = {'consolidated': consolidated_labels}
    return res


def aggregate_labels_from_bundles_to_trees\
                (input_list_of_trees_filename, tagged_bundle_directory, output_list_of_trees_filename):
    """
    Goes over all tagged branches in tagged_bundle_directory, where each file named 'treeId_branchIdx_branchLen.txt',
    Finds a suitable tree in input tree list, extracts labels into it, and saves all modified trees into a new file.
    :param input_list_of_trees_filename: path to input list of trees (usually 'all_trees.txt')
    :param tagged_bundle_directory: path to the directory with tagged branches
    :param output_list_of_trees_filename: output list of trees file, only modified trees will be stored here
    :return map {'tree_id' : [set of branch indexes that had at least one tag]}:
    """
    list_of_trees = tt.load_list_of_trees(input_list_of_trees_filename)
    tagged_files = glob.glob(tagged_bundle_directory+'/*.txt')
    output_trees_ids = set()
    output_trees_list = []
    tagged_branches = defaultdict(set)
    for tagged_file_path in tagged_files:
        tree_id = os.path.basename(tagged_file_path).split('_')[0]
        tree = tt.find_tree_in_list(list_of_trees, tree_id)
        if tree:
            if add_labels(tree, tagged_file_path):
                if tree_id not in output_trees_ids:
                    output_trees_list.append(tree)
                    output_trees_ids.add(tree_id)
                # get the branch index from filename
                branch_idx = int(os.path.basename(tagged_file_path).split('_')[1])
                tagged_branches[tree_id].add(branch_idx)

    tt.save_list_of_trees(output_trees_list, output_list_of_trees_filename)
    return tagged_branches


def count_labels(labeled_list_of_trees_path):
    """
    Goes over labeled list of trees and counts labels per node
    (Same label from different annotators is considered as one)
    :param labeled_list_of_trees_path: labeled trees file
    :return: Counter {label: count}
    """
    trees = tt.load_list_of_trees(labeled_list_of_trees_path)
    return Counter([label for tree in trees for node in tt.get_nodes(tree) if 'labels' in node
                    for label in set([label for annotator in node['labels']
                                      for label in node['labels'][annotator]])])


def get_all_labels(labeled_trees):
    """
    Goes over all trees and collects all possible labels for nodes.
    :param labeled_trees: Labeled trees.
    :return: List of labels.
    """
    all_labels = set()
    for tree in labeled_trees:
        for i, branch in enumerate(tt.get_branches(tree)):
                for node in branch:
                    if 'labels' in node:
                        for annotator in node['labels']:
                            all_labels.update(node['labels'][annotator])
    return sorted(all_labels)


def create_data_csv(labeled_list_of_trees_path, output_path, rework_settings_path=None, delete_text=False):
    """
    Creates CSV file with columns: [node_id, tree_id, timestamp, author, text, [labels]]
    Doesn't create rows for '[removed]' or '[deleted]' text entries (skips them).

    :param labeled_list_of_trees_path: labeled trees filename
    :param output_path: output csv filename
    :param rework_settings_path: if not None, rework_labels applied with these settings path
    :param delete_text: delete '[deleted]' & '[removed]' nodes
    :return: created dataframe
    """
    trees = tt.load_list_of_trees(labeled_list_of_trees_path)
    if rework_settings_path:
        trees = rework_labels(trees, settings=rework_settings_path)

    # 1. Lets collect all known Labels in these trees
    all_labels = get_all_labels(trees)

    # 2. Now we should create a Dictionary to be later used as prototype for our DataFrame
    data_dict = defaultdict(list)

    indicies_table = {-1 : -1} # id -> idx in table
    parents_table = {} # id -> id

    for tree in trees:
        visited_nodes = set()
        for i, branch in enumerate(tt.get_branches(tree)):
            prev = -1
            for node_idx, node in enumerate(branch):
                if delete_text and (node['text'] == '[removed]' or node['text'] == '[deleted]'):
                    continue
                parents_table[node['id']] = prev
                prev = node['id']
                if node['id'] not in visited_nodes:
                    visited_nodes.add(node['id'])
                    labels = set()
                    if 'labels' in node:
                        for annotator in node['labels']:
                            labels.update(node['labels'][annotator])
                    indicies_table[node['id']] = len(data_dict['node_id'])
                    data_dict['node_id'].append(node['id'])
                    data_dict['tree_id'].append(tree['node']['id'])
                    data_dict['text'].append(node['text'])
                    data_dict['timestamp'].append(node['timestamp'])
                    data_dict['author'].append(node['author'])
                    data_dict['parent'].append(indicies_table[parents_table[node['id']]])
                    for label in all_labels:
                        data_dict[label].append(1 if label in labels else 0)

    # 3. Now lets create DataFrame using Pandas and export it to CSV
    columns = ['node_id', 'tree_id', 'timestamp', 'author', 'text', 'parent']
    df = pd.DataFrame(data=data_dict)
    df = df[columns + all_labels] # sort by columns
    df.to_csv(output_path, encoding='utf-8')
    print('DONE: ' + output_path)
    return df


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return itertools.chain.from_iterable(itertools.combinations(xs, n) for n in range(len(xs) + 1))


def print_label_cooc_lists(trees, out_file):
    trees = tt.load_list_of_trees(trees)
    nodes = [node for tree in trees for node in tt.get_nodes(tree)]
    subset_counts = Counter()
    for node in nodes:
        if 'labels' in node and 'consolidated' in node['labels']:
            labels = node['labels']['consolidated']
            subsets = powerset(labels)
            subset_counts.update([subset for subset in subsets if len(subset) >= 1])

    subset_lists = defaultdict(list)
    for subset, count in subset_counts.most_common():
        l = len(subset)
        key_s = str(l) + '_len_subsets'
        key_c = str(l) + '_len_counts'
        subset_lists[key_s].append(subset)
        subset_lists[key_c].append(count)

    df = pd.DataFrame.from_dict(subset_lists, orient='index')
    df.transpose().to_csv(out_file)
    print('DONE: ' + out_file)


def rework_labels(labeled_trees_path, settings, out_trees_path=None):
    """
    Goes over labeled_trees and reworks them according to the given settings (consolidates all labels
     ,except the given in 'dont_consolidate', with a majority vote == 2. Removes all labels given in 'ignore_labels'.
     Merges all labels given in 'merge_labels'.
    :param labeled_trees_path: labeled trees path
    :param settings: settings path (a json in the format:
                            {
                                "dont_consolidate": [labels],
                                "ignore_labels": [labels],
                                "merge_labels": [ ([from], to) ]
                            }
    :param out_trees_path: out file path
    :return:
    """
    trees = tt.load_list_of_trees(labeled_trees_path)
    if type(settings) is str:
        with open(settings) as settings_file:
            settings = eval(settings_file.read())

    for tree in trees:
        for node in tt.get_nodes(tree):
            if 'labels' in node:
                labels_dict = node['labels']
                # Consolidate labels:
                label_counts = Counter([label for annotator, labels in labels_dict.items() for label in labels])
                consolidated_labels = [label for label, count in label_counts.items()
                                       if label in settings['dont_consolidate'] or count >= 2 or
                                       ('consolidated' in labels_dict and label in labels_dict['consolidated'])]
                # Ignore labels:
                filtered_labels = [label for label in consolidated_labels if label not in settings['ignore_labels']]
                # Merge labels:
                all_froms = [from_label for _from_, _to_ in settings['merge_labels'] for from_label in _from_ ]
                all_tos = [_to_ for _from_, _to_ in settings['merge_labels']
                           for from_label in _from_ if from_label in filtered_labels]
                merged_labels = set(filtered_labels) - set(all_froms) | set(all_tos)
                node['labels'] = {'consolidated': sorted(merged_labels)}

                if 'extra_data' in node and 'full_labels' in node['extra_data']:
                    # to-do: fix full_labels if merge was applied as well
                    remove_labels = [entry for entry in node['extra_data']['full_labels'] if entry[0] in settings['ignore_labels']]
                    for entry in remove_labels:
                        node['extra_data']['full_labels'].remove(entry)
    if out_trees_path:
        tt.save_list_of_trees(trees, out_trees_path)
    else:
        return trees

tag_map = [
    ('CBE', 'CounterArgument'),
    ('OCQ', 'CriticalQuestion'),
    ('CBK', 'Clarification'),
    ('DNO', 'DirectNo'),
    ('ADT', 'Nitpicking'),
    ('SAC', 'Complaint'),
    ('CBL', 'Personal'),
    ('SC', 'Positive'),
    ('SAS', 'Sarcasm'),
    ('IRR', 'Irrelevance'),
    ('AGB', 'AgreeBut'),
    ('SA', 'Aggressive'),
    ('CBA', 'RequestClarification'),
    ('OSB', 'Softening'),
    ('SRC', 'Sources'),
    ('CA', 'Moderation'),
    ('SE', 'WQualifiers'),
    ('CBF', 'Extension'),
    ('CBO', 'AttackValidity'),
    ('SAB', 'Ridicule'),
    ('CBG', 'Convergence'),
    ('CBJ', 'NegTransformation'),
    ('REP', 'Repetition'),
    ('CDV', 'DoubleVoicing'),
    ('ALO', 'Alternative'),
    ('RAA', 'RephraseAttack'),
    ('BAD', 'BAD'),
    ('CBN', 'ViableTransformation'),
    ('CBZ', 'AgreeToDisagree'),
    ('ANS', 'Answer'),
    ('CBD', 'NoReasonDisagreement'),
]

def rename_labels(in_file, out_file):
    with open(in_file) as text:
        text = text.read()
        for src, dst in tag_map:
            r = r'([^a-zA-Z])(' + src + ')([^a-zA-Z])'
            text = re.sub(r, r'\1'+dst+r'\3', text)
        with open(out_file, 'w') as out:
            out.write(text)
            print('DONE: ' + out_file)

def rename_tags_in_trees(trees, out_trees):
    trees = tt.load_list_of_trees(trees)
    nodes = [node for tree in trees for node in tt.get_nodes(tree)]
    for node in nodes:
        if 'labels' in node and 'consolidated' in node['labels']:
            for i, label in enumerate(node['labels']['consolidated']):
                for old_tag, new_tag in tag_map:
                    if label == old_tag:
                        node['labels']['consolidated'][i] = new_tag
        if 'extra_data' in node and 'full_labels' in node['extra_data']:
            for i, full_label in enumerate(node['extra_data']['full_labels']):
                for j, entry in enumerate(full_label):
                    for old_tag, new_tag in tag_map:
                        if entry == old_tag:
                            node['extra_data']['full_labels'][i][j] = new_tag
    tt.save_list_of_trees(trees, out_trees)
    #print('DONE:', out_trees)

def rename_tags_in_df(df, out_df):
    df = pd.read_csv(df, index_col=0)
    df = df.rename(columns={old_tag: new_tag for old_tag, new_tag in tag_map})
    df.to_csv(out_df)
    print('DONE:', out_df)

def print_label_passes(trees, out_dir):
    """
    For each label L1, count how many times it appears i posts before a label L2 (i=0,1,2,3,4,5,6,7,8,9,10).
    The output for each label L1 should be a TSV file (called <L1>_unigram_backward.tsv) with 12 columns.
    The first column is for labels (a label per line), then each column for i in {0,...,10}
    will have the number of times this label appear in the i-th post before L1 (so the the CBJ line for BAD_unigram_backward.tsv
    for , for col i=0 we count how many times CBJ appears WITH BAD on the same post,
    for the i=1 column you count how many times CBJ appears in the post before the post with the BAD,
    for i=7 you count how many times CBJ appears in the seventh post before BAD, and so on.
    So in this case L1 is BAD and we used the example of L2=CBJ.
    This will give an intuition about the chains and what causes bad.
    Same thing but this time the files will be called <L1>_unigram_forward.tsv) in which for each L1 you count the
    number of times L2 appears i posts AFTER L1. (note that the second column [i=0] is supposed to be identical for each
    specific L1's backwords and forward files since it counts the time each L2 cooccurs with L1;
    also note that the files are NOT symmetric)
    :param trees:
    :param out_dir:
    :return:
    """
    trees = tt.load_list_of_trees(trees)
    branches = [branch for tree in trees for branch in tt.get_branches(tree)]
    nodes = [node for tree in trees for node in tt.get_nodes(tree)]

    all_labels = set()
    for node in nodes:
        if 'labels' in node and 'consolidated' in node['labels']:
            all_labels.update(node['labels']['consolidated'])
    r = 11
    forward_counts = {l1:{l2:{'i='+str(i):0 for i in range(r)} for l2 in all_labels} for l1 in all_labels}
    backward_counts = {l1:{l2:{'i='+str(i):0 for i in range(r)} for l2 in all_labels} for l1 in all_labels}

    visited_pairs = set()
    for branch in branches:
        for c1, n1 in enumerate(branch):
            for i in range(r):
                if c1+i >= len(branch):
                    break
                n2 = branch[c1+i]
                if (n1['id'], n2['id']) in visited_pairs:
                    continue
                visited_pairs.add((n1['id'], n2['id']))
                if 'labels' in n1 and 'consolidated' in n1['labels'] and 'labels' in n2 and 'consolidated' in n2['labels']:
                    for l1 in n1['labels']['consolidated']:
                        for l2 in n2['labels']['consolidated']:
                            forward_counts[l1][l2]['i='+str(i)]+=1
                            backward_counts[l2][l1]['i='+str(i)]+=1

    os.makedirs(out_dir+'/forward', exist_ok = True)
    os.makedirs(out_dir+'/backward', exist_ok = True)
    for l in all_labels:
        pd.DataFrame.from_dict(backward_counts[l]).transpose().to_csv(out_dir+'/backward/'+l+'_unigram_backward.csv')
        print('DONE:', out_dir+'/backward/'+l+'_unigram_backward.csv')
        pd.DataFrame.from_dict(forward_counts[l]).transpose().to_csv(out_dir+'/forward/'+l+'_unigram_forward.csv')
        print('DONE:', out_dir+'/forward/'+l+'_unigram_forward.csv')

def main(argv):
    print('Wrong usage, bro.')


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])