################################################
#               NasLab - BGU 2018              #
################################################

import json
import re, os
import heapq
import mpmath
from numpy import median
from numpy import mean
import hashlib
from collections import defaultdict
from collections import Counter
from io import StringIO
import csv


def get_branches(tree):
    """
    Calculates all the branches in Tree.

    Args:
        tree: The Tree in Unified JSON-like Format.

    Returns:
        A list of branches, where each branch is a list of nodes.
    """
    return _get_branches(tree)


def extract_networks_from_trees(list_of_trees, list_of_network_types, output_file_path):
    """
    Creates multi-graph edge list output file in the following format:

        tree_id	amount_of_branches:
            DA:
                tree:
                    u1	u2	ts1
                    ...........
                branches:
                    0:
                        u1 u2 ts1
                        .........
                    5:
                        u3 u1 ts2
            QU:
                tree:
                    u1	u2	ts1
                    ...........
                branches:
                    0:
                        u1 u2 ts1
                        .........
                    5:
                        u3 u1 ts2
            MN:
                ............
        tree_id_2	amount_of_branches_2:
            .......

    Args:
        list_of_trees: List of Trees (possibly loaded with 'load_list_of_trees(path)' method).
        list_of_network_types: Network Types needed for extraction: ['DA','QU','MN'] : Direct Answers, Quotes, Mentions.
        output_file_path: Path to the output file.
    """

    with open(output_file_path, 'w') as out:
        for tree in list_of_trees:
            total_branches = len(get_branches(tree))
            out.write(tree['node']['id']+'\t'+str(total_branches)+':\n')
            if 'DA' in list_of_network_types:
                out.write('\t'+'DA:\n')
                da_matrix = answers_matrix(tree)
                out.write(print_matrix(da_matrix, '\t\t'))
            if 'QU' in list_of_network_types:
                out.write('\t' + 'QU:\n')
                qu_matrix = quotes_matrix(tree)
                out.write(print_matrix(qu_matrix, '\t\t'))
            if 'MN' in list_of_network_types:
                out.write('\t' + 'MN:\n')
                mn_matrix = mentions_matrix(tree)
                out.write(print_matrix(mn_matrix, '\t\t'))
    return


def print_networks_from_tree(tree, out_dir, list_of_network_types = ['DA', 'QU', 'MN']):
    """
    Receives a Tree, and prints all it's networks into distinct files to the given output directory

    Args:
        tree: a Tree
        out_dir: Output directory
        list_of_network_types: Networks to print ('DA' = Direct Answers, 'QU' = Quotes, 'MN' = Mentions)
                                Default Parameter: ['DA', 'QU', 'MN']
    """
    for network in list_of_network_types:
        if network == 'DA':
            suffix = 'da'
            matrix = answers_matrix(tree)
        elif network == 'QU':
            suffix = 'qu'
            matrix = quotes_matrix(tree)
        elif network == 'MN':
            suffix = 'mn'
            matrix = mentions_matrix(tree)
        else:
            continue

        # Weighted, for whole Tree
        edgelist = translate_matrix_to_edgelist(matrix['tree_map'], True)
        filename = 'el_' + tree['node']['id'] + '_' + suffix + '_wt_tr.tsv'
        print_edgelist(edgelist, os.path.join(out_dir, filename))

        # Multigraph, for whole Tree
        edgelist = translate_matrix_to_edgelist(matrix['tree_map'], False)
        filename = 'el_' + tree['node']['id'] + '_' + suffix + '_mg_tr.tsv'
        print_edgelist(edgelist, os.path.join(out_dir, filename))

        for (i, branch_matrix) in enumerate(matrix['per_branch']):
            if branch_matrix:
                # Weighted, per Branch
                edgelist = translate_matrix_to_edgelist(branch_matrix, True)
                filename = 'el_' + tree['node']['id'] + '_' + suffix + '_wt_br_' + str(i) + \
                           '_' + str(len(edgelist)) + '.tsv'
                print_edgelist(edgelist, os.path.join(out_dir, filename))

                # Multigraph, per Branch
                edgelist = translate_matrix_to_edgelist(branch_matrix, False)
                filename = 'el_' + tree['node']['id'] + '_' + suffix + '_mg_br_' + str(i) + \
                           '_' + str(len(edgelist)) + '.tsv'
                print_edgelist(edgelist, os.path.join(out_dir, filename))


def translate_matrix_to_edgelist(matrix, weighted_output = False):
    """
    Receives a matrix and converts it to edge-list (list of 3-val tuples).

    Args:
        matrix: in form of dictionary - {'user1' : [(user2, timestamp), ...],
                                         ...................................}

        weighted_output: A boolean to indicate if expected weighted-graph output or multi-graph (with timestamp)

    Returns:
        Edge-list - a list of 3-val tuples. If 'weighted_output' == True, then the tuples are (user1, user2, weight),
        else - the tuples will be (user1, user2, timestamp).
    """
    edgelist = []
    for user1 in matrix:
        if not weighted_output:
            for (user2, timestamp) in matrix[user1]:
                edgelist.append((user1, user2, timestamp))
        else:
            weights = defaultdict(int)
            for (user2, timestamp) in matrix[user1]:
                weights[user2] += 1
            for (user2, weight) in weights.items():
                edgelist.append((user1, user2, weight))
    return edgelist


def print_edgelist(edgelist, filename):
    with open(filename, 'w') as out:
        for (user1, user2, meta) in edgelist:
            out.write(user1+'\t'+user2+'\t'+str(meta)+'\n')


def get_node_address(tree, node):
    """
    Goes over a given tree and finds a first tuple (br_idx, nd_idx) that defines the Node's address.
    :param tree: tree to search for node
    :param node: (br_idx, nd_idx) tuple - node's address within the tree. None if not present.
    :return:
    """
    for br_idx, br in enumerate(get_branches(tree)):
        for nd_idx, nd in enumerate(br):
            if nd == node:
                return br_idx, nd_idx
    return None


def print_matrix(matrix, prefix=''):
    """
    Prints a given edges matrix of a tree to a String.
    Matrix should be obtained with 'mentions_matrix()', 'answers_matrix()', or 'quotes_matrix()' function.
    Matrix will be printed in the following format:

        tree:
            u1 u2 ts1
            .........
        branches:
            0:
                u1 u2 ts1
                .........
            5:
                u3 u5 ts3
                .........

    * Note: Branches with empty edge list will be skipped.

    Args:
        matrix: The Matrix obtained with a dedicated function.
        prefix: Will be added at the beginning of each new line in the output string.
                Typically used for proper indentation.

    Returns:
        The output string.

    """
    out_strings = []
    out_strings.append(prefix + 'tree:\n')
    for user1 in matrix['tree_map']:
        for (user2, timestamp) in matrix['tree_map'][user1]:
            out_strings.append(prefix + '\t' + user1 + '\t' + user2 + '\t' + str(timestamp)+'\n')
    out_strings.append((prefix+'branches:\n'))
    for (branch_counter, branch_map) in enumerate(matrix['per_branch']):
        if branch_map:
            out_strings.append(prefix + '\t' + str(branch_counter) + ':\n')
            for user1 in branch_map:
                for (user2, timestamp) in branch_map[user1]:
                    out_strings.append(prefix + '\t\t' + user1 + '\t' + user2 + '\t' + str(timestamp) + '\n')
    return ''.join(out_strings)


def mentions_matrix(tree):
    r"""
    Fetch the Edge List matrix for a tree & for each of its branches.
    The edge exists iff: User1 mentions User2 (with a special tag '\u\User2')

    Args:
        tree: The Tree in Unified JSON-like Format.

    Returns:
        Dictionary with the following structure:

        {
            'tree_map' :  {'user1' : [(user2, timestamp), ...],
                           ...................................}
            'per_branch': [ {'user1' : [(user2, timestamp), ...], ...... },
                           ............................................... ]
        }
    """
    mentions_map = {}
    id_node = 'id'
    mentions_list_per_branch = []
    visited_ids_set = set()
    text_node = 'text'
    author_node = 'author'
    timestamp_node = 'timestamp'
    mentions_re = re.compile(r'/u/(\w*)', re.MULTILINE)
    branches = get_branches(tree)

    for branch in branches:
        branch_map = {}
        for reply in branch:
            text = reply[text_node]
            user1 = reply[author_node]
            timestamp = reply[timestamp_node]

            for user2 in mentions_re.findall(text):
                # Lets update the whole-tree matrix
                entry_id = reply[id_node] + '_' + user2
                if user1 not in mentions_map:
                    mentions_map[user1] = []  # list of (user, time) tuples for tree
                if entry_id not in visited_ids_set:
                    mentions_map[user1].append((user2, timestamp))
                    visited_ids_set.add(entry_id)
                # And now the current branch matrix
                if user1 not in branch_map:
                    branch_map[user1] = []  # for branch
                branch_map[user1].append((user2, timestamp))

        mentions_list_per_branch.append(branch_map)

    return {'tree_map': mentions_map, 'per_branch': mentions_list_per_branch}


def answers_matrix(tree):
    """
    Fetch the Edge List matrix for a tree & for each of its branches.
    The edge exists iff: User1 directly answers to User2 (next in Branch)

    Args:
        tree: The Tree in Unified JSON-like Format.

    Returns:
        Dictionary with the following structure:

        {
            'tree_map' :  {'user1' : [(user2, timestamp), ...],
                           ...................................}
            'per_branch': [ {'user1' : [(user2, timestamp), ...], ...... },
                           ............................................... ]
        }
    """
    answers_map = {}
    answers_list_per_branch = []
    author_node = 'author'
    timestamp_node = 'timestamp'
    id_node = 'id'
    visited_ids_set = set()
    branches = get_branches(tree)

    for branch in branches:
        branch_map = {}
        for reply_counter, reply in enumerate(branch):
            if reply_counter == 0:
                continue
            previous_reply = branch[reply_counter-1]
            user1 = reply[author_node]
            user2 = previous_reply[author_node]
            timestamp = reply[timestamp_node]

            # Lets update the whole-tree matrix
            if user1 not in answers_map:
                answers_map[user1] = []
            entry_id = reply[id_node] + '_' + previous_reply[id_node]
            if entry_id not in visited_ids_set:
                answers_map[user1].append((user2, timestamp))
                visited_ids_set.add(entry_id)

            # And now the current branch matrix
            if user1 not in branch_map:
                branch_map[user1] = []
            branch_map[user1].append((user2, timestamp))

        answers_list_per_branch.append(branch_map)

    return {'tree_map': answers_map, 'per_branch': answers_list_per_branch}


def quotes_matrix(tree):
    """
    Fetch the Edge List matrix for a tree & for each of its branches.
    The edge exists iff: User1 quotes User2 (from the same Branch) (quoting done with tag <quote>...</quote>)

    Args:
        tree: The Tree in Unified JSON-like Format.

    Returns:
        Dictionary with the following structure:

        {
            'tree_map' :  {'user1' : [(user2, timestamp), ...],
                           ...................................}
            'per_branch': [ {'user1' : [(user2, timestamp), ...], ...... },
                           ............................................... ]
        }

    """
    quotes_map = {}
    quotes_list_per_branch = []
    author_node = 'author'
    text_node = 'text'
    id_node = 'id'
    timestamp_node = 'timestamp'
    visited_ids_set = set()
    branches = get_branches(tree)

    def _find_author(_branch, _quoted_text, _max=0):
        for _counter, _reply in enumerate(_branch):
            if _counter >= _max > 0:
                break
            _quote_itself = re.compile(r'<quote>\s*'+re.escape(_quoted_text)+r'\s*</quote>', re.MULTILINE)
            _text = _reply[text_node]
            _author = _reply[author_node]
            if _quoted_text in _text and _quote_itself.match(_text) is None:
                # Test if the Quoted Part is in Text, but it is not just another Quote
                # Meaning - we found the original comment
                return _author
        return None

    for branch in branches:
        branch_map = {}
        for reply_counter, reply in enumerate(branch):
            if reply_counter == 0:
                continue
            quote_re = re.compile(r'<quote>\s*(.*?)\s*</quote>', re.MULTILINE)
            text = reply[text_node]
            user1 = reply[author_node]
            timestamp = reply[timestamp_node]

            for quoted_text in quote_re.findall(text):
                entry_id = reply[id_node] + '_' + hashlib.md5(quoted_text.encode('utf8')).hexdigest()
                user2 = _find_author(branch, quoted_text, reply_counter)
                if user2 is not None:
                    # Lets update the whole-tree matrix
                    if user1 not in quotes_map:
                        quotes_map[user1] = []
                    if entry_id not in visited_ids_set:
                        quotes_map[user1].append((user2, timestamp))
                        visited_ids_set.add(entry_id)

                    # And now the current branch matrix
                    if user1 not in branch_map:
                        branch_map[user1] = []
                    branch_map[user1].append((user2, timestamp))

        quotes_list_per_branch.append(branch_map)

    return {'tree_map': quotes_map, 'per_branch': quotes_list_per_branch}


# Undocumented function which calculates some complex metric for the given tree.
def find_avg_length_top_k_distinct_branches(tree, list_of_ks):
    counter = [0]  # first we simplify the branch to be represented by simple integer array
    id_to_number = {}

    def _assign_number(_node):
        if _node['id'] not in id_to_number:
            id_to_number[_node['id']] = counter[0]
            counter[0] += 1
        return id_to_number[_node['id']]

    branches = get_branches(tree)
    branches_simplified = []
    for branch in branches:
        branch_simplified = list(map(_assign_number, branch))
        branches_simplified.append(branch_simplified)

    mpmath.mp.dps = 50

    def _find_distance(_branch, _branches):
        if len(_branches) == 0:
            return mpmath.mpf(1)
        _branch = list(_branch)  # copy the inputs, we can't modify them
        _branches = [list(_br) for _br in _branches]

        _similarity = []
        _branch.pop(0)  # skip root

        for _branch_other in _branches:
            _i = 0
            _branch_other.pop(0)  # skip root
            while _i < len(_branch) and _i < len(_branch_other) and _branch[_i] == _branch_other[_i]:
                _i += 1
            _similarity.append(_i)
        _distances = []
        for _i, _sim in enumerate(_similarity):
            _distances.append(mpmath.mpf((len(_branches[_i]) - _sim + len(_branch) - _sim)) /
                              mpmath.mpf((len(_branches[_i]) + len(_branch))))
        _quadratic_mean = mpmath.mpf(0)
        for _dis in _distances:
            _quadratic_mean += _dis * _dis
            _quadratic_mean = mpmath.sqrt(_quadratic_mean / mpmath.mpf(len(_distances)))
        return _quadratic_mean

    list_of_ms = []
    branches_copy = list(branches_simplified)
    for i in range(max(list_of_ks)):
        m = mpmath.mpf(0)
        chosen = -1
        for j, br in enumerate(branches_copy):
            others = list(branches_copy)
            others.pop(j)
            d = _find_distance(br, others)*mpmath.mpf(len(br)-1)
            if d > m:
                m = d
                chosen = j
        list_of_ms.append(m)
        branches_copy.pop(chosen)
        if len(branches_copy) == 0:
            break
    list_of_averages = []
    for k in list_of_ks:
        list_of_averages.append(mean(list_of_ms[:min(k, len(list_of_ms))]))
    return list_of_averages


def get_tree_stats(tree, params=None):
    """
    Calculate Statistics for a given Tree, according to Parameters.

    Args:
        tree: The Tree in Unified JSON-like Format.
        params: The Parameters describing a desired statistics. If None, defaults will be used.

    Returns:
        The Statistics - a Dictionary.

    """
    if params is None:
        params = {
                    'nodes_count': True,
                    'branches_count': True,
                    'users_count': True,
                    'branching_factor': True,
                    'median_branch_length': True,
                    'median_number_of_comments_per_user': True,
                    'avg_length_top_k_branches': [1, 3, 5],
                    'avg_comments_top_k_users': [1, 3, 5],
                    'avg_length_top_k_distinct_branches': [5, 7]
                }
    basic_stats = _traverse_and_gather_stats(tree)
    full_stats = {}

    branches = get_branches(tree)
    branches_count = len(branches)
    nodes_count = basic_stats["nodes_count"]
    branches_lens = [len(br) for br in branches]
    branching_factor = 0 if nodes_count == 1 else float(nodes_count - 1) / (nodes_count - branches_count)
    users_comment_counts_unf = [_value["comments_count"] if _key != "[deleted]" or _key != "[removed]" else None for _key, _value in basic_stats["user_stats"].items()]  # we don't count users named 'deleted' or 'removed'
    users_comment_counts = [n for n in users_comment_counts_unf if n is not None]  # filter out bad users
    users_count = len(users_comment_counts)
    avg_branch_length = (sum(branches_lens) * 1.0) / len(branches_lens)
    avg_comments_per_user = (sum(users_comment_counts) * 1.0) / len(users_comment_counts)

    for (key, value) in params.items():
        if value is True:
            if key == "median_branch_length":
                full_stats[key] = median(branches_lens)
            if key == "users_count":
                full_stats[key] = users_count
            if key == "branches_count":
                full_stats[key] = branches_count
            if key == "nodes_count":
                full_stats[key] = nodes_count
            if key == "branching_factor":
                full_stats[key] = branching_factor
            if key == "median_number_of_comments_per_user":
                full_stats[key] = median(users_comment_counts)

        if type(value) is list:
            full_stats[key] = {}
            __list_of_dis_avgs = None
            if key == "avg_length_top_k_distinct_branches":
                __list_of_dis_avgs = find_avg_length_top_k_distinct_branches(tree, value)
            for (idx, k) in enumerate(value):
                if key == "avg_length_top_k_branches":
                    full_stats[key][k] = sum(heapq.nlargest(k, branches_lens)) / float(k) \
                        if branches_count >= k else avg_branch_length
                if key == "avg_comments_top_k_users":
                    full_stats[key][k] = sum(heapq.nlargest(k, users_comment_counts)) / float(k) \
                        if users_count >= k else avg_comments_per_user
                if key == "avg_length_top_k_distinct_branches":
                    full_stats[key][k] = __list_of_dis_avgs[idx]

    return full_stats


def create_list_of_trees_statistics(list_of_trees, out_stats_file):
    """
    Creates tab-separated csv file of statistics for list of trees.

    Args:
        list_of_trees: List of Trees (possibly loaded with 'load_list_of_trees(path)' method).
        out_stats_file: CSV comma-separated output file.
    """
    if type(list_of_trees) is str:
        list_of_trees = load_list_of_trees(list_of_trees)

    counter = 0
    with open(out_stats_file, 'w', encoding="utf-8") as out:
        header = True
        for tree in list_of_trees:
            #if tree['node']['author'] == '[deleted]' or tree['node']['author'] == '[removed]':
            #    continue
            #if tree['node']['text'] == '[deleted]' or tree['node']['text'] == '[removed]':
            #    continue
            stats = get_tree_stats(tree)
            counter += 1
            if counter % 100 == 0:
                print(counter)
            if header:
                header = False
                header_str = 'tree_id,submission_text'
                for key, value in stats.items():
                    if type(value) is dict:
                        for k, k_val in value.items():
                            header_str += ','+key+'_'+str(k)
                    else:
                        header_str += ',' + key
                header_str += '\n'
                out.write(header_str)

            stats_str = tree['node']['id']

            text = tree['node']['text']
            output = StringIO()
            wr = csv.writer(output, quoting=csv.QUOTE_ALL)
            wr.writerow([text])
            escaped_text = output.getvalue().strip()

            stats_str += ','+escaped_text
            for key, value in stats.items():
                if type(value) is dict:
                    for k, k_val in value.items():
                        stats_str += ',%.3f' % k_val
                else:
                    stats_str += ',%.3f' % value
            stats_str += '\n'
            out.write(stats_str)
        print('DONE: ' + out_stats_file)


def load_list_of_trees(trees_file):
    """
    Load a file of new-line separated trees into a Python list

    Args:
        trees_file: Path to file of new-line separated list of trees.

    Returns:
        Python list of trees loaded from given file.
    """
    if type(trees_file) is str:
        f = trees_file
        trees = []
        data = open(f)
        for line in data:
            tree = json.loads(line)
            trees.append(tree)
        data.close()
        return trees
    if type(trees_file) is list:
        return trees_file


def save_list_of_trees(list_of_trees, output_list_of_trees_filename):
    with open(output_list_of_trees_filename, 'w+') as out_file:
        for tree in list_of_trees:
            json.dump(tree, out_file)
            out_file.write('\n')
        print('DONE: ' + output_list_of_trees_filename)


def find_tree_in_list(list_of_trees, tree_id):
    """
    Finds and returns tree from list by tree-id (None if failed)
    :param list_of_trees: List of trees to search
    :param tree_id: tree_id
    :return: found tree or None
    """

    for tree in list_of_trees:
        if tree['node']['id'] == tree_id:
            return tree


def get_node(tree, branch, node):
    """
    Finds node in tree by given branch ordinal number in a tree and node ordinal number in a branch. All start with 0.
    If not such node exists, NONE returned. Otherwise, dictionary-object for relevant node returned.

    :param tree: tree
    :param branch: branch ordinal number in a tree (starts with 0)
    :param node: node ordinal number in a tree (starts with 0)
    """

    branches = get_branches(tree)
    if branch < len(branches) and node < len(branches[branch]):
        return branches[branch][node]
    else:
        return None


def get_nodes(tree):
    """
    Returns a list of nodes
    :param tree:
    :return: List of nodes in a tree
    """
    scanned_nodes = set()
    list_of_nodes = []
    branches = get_branches(tree)
    for branch in branches:
        for node in branch:
            if node['id'] not in scanned_nodes:
                list_of_nodes.append(node)
                scanned_nodes.add(node['id'])
    return list_of_nodes


def load_category_dict(fn="LIWC_Features.txt"):
    if not os.path.isfile(fn):
        return {}
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


liwc_regex_dict = create_cat_regex_dict(load_category_dict())
def get_liwc_cats(text):
    global liwc_regex_dict
    return Counter({cat: len(regex.findall(text.lower())) for cat, regex
            in liwc_regex_dict.items() if regex.findall(text.lower())})


def text_to_liwc(text):
    global liwc_regex_dict
    return ' '.join([' '.join([cat] * len(regex.findall(text.lower()))) for cat, regex
                     in liwc_regex_dict.items() if regex.findall(text.lower())])


def _traverse_and_gather_stats(sub_tree, stats=None):
    if stats is None:
        stats = {
            'users_count': 0,
            'nodes_count': 0,
            'user_stats': {}
        }
    node = sub_tree['node']
    text_to_filter = r'<quote>.*</quote>'
    if node['author'] not in stats['user_stats']:
        stats['user_stats'][node['author']] = defaultdict(int)
    stats['user_stats'][node['author']]['comments_count'] += 1  # calculate words count using 'str.split'
    stats['user_stats'][node['author']]['words_count'] += (len(re.sub(text_to_filter, '', node['text']).split()))
    stats['nodes_count'] += 1
    for child_node in sub_tree['children']:
        _traverse_and_gather_stats(child_node, stats)
    return stats


def _get_branches(tree, branch=list()):
    local_branch = list(branch)  # we need to copy the given branch,
    # since recursive calls will make a mess in the shared one
    local_branch.append(tree['node'])
    if len(tree['children']) == 0:
        return [local_branch]
    else:
        all_branches = []
        for child in tree['children']:
            child_branches = _get_branches(child, local_branch)
            all_branches += child_branches
        return all_branches


def get_full_branches(tree):
    return _get_full_branches(tree)


def get_full_nodes(tree):
    scanned_nodes = set()
    list_of_nodes = []
    branches = get_full_branches(tree)
    for branch in branches:
        for node in branch:
            if node['node']['id'] not in scanned_nodes:
                list_of_nodes.append(node)
                scanned_nodes.add(node['node']['id'])
    return list_of_nodes


def _get_full_branches(tree, branch=list()):
    local_branch = list(branch)  # we need to copy the given branch,
    # since recursive calls will make a mess in the shared one
    local_branch.append(tree)
    if len(tree['children']) == 0:
        return [local_branch]
    else:
        all_branches = []
        for child in tree['children']:
            child_branches = _get_full_branches(child, local_branch)
            all_branches += child_branches
        return all_branches


def remove_duplicate_children(tree_node):
    children = tree_node['children']
    ids = [child['node']['id'] for child in children]
    sizes = [len(json.dumps(child)) for child in children]
    to_delete = []
    to_save = []
    if len(ids) == len(set(ids)):
        return
    for k, child in enumerate(children):
        if k in to_delete or k in to_save:
            continue
        duplicates = [i for i, _id in enumerate(ids) if child['node']['id'] == _id]
        if len(duplicates) == 1:
            continue
        duplicate_sizes = [sizes[i] for i in duplicates]
        biggest_duplicates = [i for i in duplicates if sizes[i] == max(duplicate_sizes)]
        to_delete += [i for i in duplicates if i != biggest_duplicates[0]]
        to_save += [biggest_duplicates[0]]
    if to_delete:
        print("NodeID %s has detected the following delete candidates: " % tree_node['node']['id'])
        print(to_delete)
        print("NodeID %s these nodes will remain, as largest originals to their duplicates: " % tree_node['node']['id'])
        print(to_save)

    new_children = [child for i, child in enumerate(children) if i not in to_delete]
    tree_node['children'] = new_children


def traverse_tree_preorder(tree, func):
    func(tree)
    for child in tree['children']:
        traverse_tree_preorder(child, func)


def remove_duplicate_nodes(input_trees_path, out_trees_path):
    trees = load_list_of_trees(input_trees_path)
    for tree in trees:
        traverse_tree_preorder(tree, remove_duplicate_children)
    with open(out_trees_path, 'w+') as out_file:
        for tree in trees:
            json.dump(tree, out_file)
            out_file.write('\n')
        print('DONE ' + out_trees_path)


def check_tree_integrity(golden_trees, test_trees):
    for test_tree in test_trees:
        golden_tree = None
        for any_tree in golden_trees:
            if any_tree['node']['id'] == test_tree['node']['id']:
                golden_tree = any_tree
                break
        if not golden_tree:
            print("Not found Golden Tree for Test Tree with id: %s" % test_tree['node']['id'])
            break
        br1 = get_branches(golden_tree)
        br2 = get_branches(test_tree)
        print("Got two trees with ids (golden, test): %s, %s" % (golden_tree['node']['id'], test_tree['node']['id']))
        print("Branches count (golden, test): %d , %d" % (len(br1), len(br2)))
        nodes1 = get_nodes(golden_tree)
        nodes2 = get_nodes(test_tree)
        print("Nodes count (golden, test): %d , %d" % (len(nodes1), len(nodes2)))

        print("Computing Golden Tree Statistics: ")
        print(get_tree_stats(golden_tree))
        print("Computing Test Tree Statistics: ")
        print(get_tree_stats(test_tree))
        node_ids = []
        traverse_tree_preorder(golden_tree, lambda tree_node : node_ids.append(tree_node['node']['id']))
        print('Golden Tree Preorder Nodes: ')
        print('Total nodes: %d' % len(node_ids))
        print('Unique nodes: %d' % len(set(node_ids)))
        print(node_ids)
        node_ids = []
        traverse_tree_preorder(test_tree, lambda tree_node: node_ids.append(tree_node['node']['id']))
        print('Test Tree Preorder Nodes: ')
        print('Total nodes: %d' % len(node_ids))
        print('Unique nodes: %d' % len(set(node_ids)))
        print(node_ids)
        print('\n')


def create_bundles(trees_file_fn, required_tree_ids, out_path):
    """
    Extracts branches of the specified trees and saves them as flat files bundles (for annotation).
    ADD a check if output path exists, if not - create it.

    :param trees_file_fn:  a file containing trees in a json format
    :param required_tree_ids: a list of the tree ids to extract
    :param out_path: path to save the branches
    """

    trees = load_list_of_trees(trees_file_fn)  # this returns a list of json-like trees, like in the readme on GitHub,
    # where I describe the json format
    for tree in trees:
        if tree['node']['id'] in required_tree_ids:
            branches = get_branches(tree)  # this returns a list of branches, where each branch is a list of nodes,
            # the nodes like in GitHub readme where I describe Json tree
            for i, branch in enumerate(branches):
                out_fn = '%s_%d_%d.txt' %(tree['node']['id'],i,len(branch))
                outfile = open(out_path+ out_fn, 'w')
                for j, node in enumerate(branch):
                    text = '%d.\t%s\t:>>>\t%s\n' %(j,node['author'], node['text'])
                    outfile.write(text.encode('utf-8'))
                outfile.close()
                print('DONE ' + out_fn)


def get_longest_branhces_by_fn(path, suffix='txt', n=1, tree_ids=None):
    """
    :param path: full path for the branches flat files.
    :param suffix: suffix (file type) of the branches flat files
        (assuming that all files of that type is the specified directory are branches).
    :param n: the number of longest branches to return from each tree, Default=1.
    :param tree_ids: if not None, gets longest branches only for the specified trees. Default=None.
    :return: a list of file names, each file corresponds to a branch.
    """
    files = [fn for fn in os.listdir(path) if fn[-len(suffix):]==suffix]
    branches_d = defaultdict(list)
    for fn in files:   #6zq9km_392_7.txt
        tree_id, branch_id, l = fn.split('.')[0].split('_')
        if not tree_ids or tree_id in tree_ids:
            branches_d[tree_id]+=[(branch_id, int(l))]
    for k, a in branches_d.items():
        branches_d[k] = sorted(a, key=lambda x: x[1], reverse=True)
    return ['%s_%s_%d.%s' %(k,a[i][0],a[i][1],suffix) \
                for k, a in branches_d.items() for i in range(0,n)]


def translate_list_of_trees(trees, out_file = None):
    """
    Another preprocessing step.
    Is used to go over a Trees in 'in_file', which are the result of 'CleanAndUnify' function, and translates
    tree-by-tree into a format which is to be used by the TreeTools package.
    :param in_file: Input list of trees, that are produced by 'CleanAndUnify' function.
    :param out_file: Output list of trees, that is to be consistent and good to use with the TreeTools package.
    :return: None
    """
    def translate(d):
        nodeName = 'node'
        if nodeName in d:
            for key in list(d[nodeName].keys()):
                if key in d[nodeName]:
                    # double check needed because we dynamically modify the dictionary, however .keys() returns a static copy
                    # see if need to rename anything
                    rename = {"from": ["created_utc"],"to": "timestamp"}
                    move = {"not" : ["text", "timestamp", "id", "created_utc", "author", "extra_data", "labels"], "to" : "extra_data"}
                    for src in rename['from']:
                        if (key == src):
                            dest = rename['to']
                            d[nodeName][dest] = d[nodeName].pop(key)
                    if move["to"] not in d[nodeName]: d[nodeName][move["to"]] = {}
                    if (key not in move["not"]): d[nodeName][move["to"]][key] = d[nodeName].pop(key)
        return
    trees = load_list_of_trees(trees)
    if out_file:
        with open(out_file, 'w') as out:
            for tree in trees:
                traverse(tree, translate)
                json.dump(tree, out)
                out.write('\n')
            print('DONE: ' + out_file)
        return
    else:
        for tree in trees:
            traverse(tree, translate)
        return trees


def traverse(t, func):
    """
    Internal.
    Traverses over the Dict 't' and applies 'func' to each of it sub-dicts.
    :param t: Object 't'
    :param func: Function 'func'
    :return: None
    """
    if type(t) is dict:
        func(t)
        for key in list(t.keys()):
            # double check needed because we dynamically modify the dictionary, however .keys() returns a static copy
            if key in t:
                if type(t[key]) is dict: traverse(t[key], func)
                if type(t[key]) is list:
                    for elem in t[key]: traverse(elem, func)
    return


def main():
    print("NOTING HERE YET")


if __name__ == "__main__":
    # execute only if run as a script
    main()
