# tree-tools



## Each tree should be of the following JSON format:
```javascript
  {
    "node": {
      "text": text,
      "author": author /*nickname*/,
      "timestamp": 1531233634 /*  https: //www.epochconverter.com/  */ ,
      "id": id /* unique node id */ ,
      "extra_data": {
        /* source-specific data */
        "subreddit": bla,
        "twitter": bla2
      }
    },
    "children": [{
      "node": {
        "text": text1,
        "author": author2,
        "timestamp": 1531233834,
        "id": id2,
        "extra_data": {
          "subreddit": bla,
          "twitter": bla2
        }
      }"children": []
    }]
  }
  ```

## FUNCTIONS LIST

```python

def load_list_of_trees(trees_file_path):
    """
    Load a file of new-line separated trees into a Python list

    Args:
        trees_file_path: Path to file of new-line separated list of trees.

    Returns:
        Python list of trees loaded from given file.
    """
    
def create_list_of_trees_statistics(list_of_trees, out_stats_file):
    """
    Creates tab-separated csv file of statistics for list of trees.

    Args:
        list_of_trees: List of Trees (possibly loaded with 'load_list_of_trees(path)' method).
        out_stats_file: CSV tab-separated output file.
    """
    
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

def get_tree_stats(tree, params=None):
    """
    Calculate Statistics for a given Tree, according to Parameters.

    Args:
        tree: The Tree in Unified JSON-like Format.
        params: The Parameters describing a desired statistics. If None, defaults will be used.

    Returns:
        The Statistics - a Dictionary.

    """

def get_branches(tree):
    """
    Calculates all the branches in a Tree.

    Args:
        tree: The Tree in Unified JSON-like Format.

    Returns:
        A list of branches, where each branch is a list of nodes.
    """
    
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

def mentions_matrix(tree):
    """
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
```
