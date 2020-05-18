"""
Kamin Tools 2019.

Usage:
  KaminGo.py aggregate_labels <bundles_dir> <all_trees.txt> <out_dir> <prefix>
  KaminGo.py create_bundles <all_trees.txt> <out_dir> <tree_ids>...
  KaminGo.py rework_labels <labeled_trees.txt> <rework_settings.txt> <out_trees.txt>
  KaminGo.py create_df <labeled_trees.txt> <out_df.csv> [--ignore_deleted --rework_settings=<rework_settings.txt>]
  KaminGo.py rename_labels <in_file> <out_file>
  KaminGo.py rename_tags_in_trees <trees.txt> <out_trees.txt>
  KaminGo.py rename_tags_in_df <df.csv> <out_df.csv>
  KaminGo.py print tags_npmi <labeled_trees.txt> <out.csv> [--just_count --log=<log.txt>]
  KaminGo.py print label_details <labeled_trees.txt> <out.csv>
  KaminGo.py print label_priors <labeled_trees.txt> <out.csv> [--per_tree]
  KaminGo.py print label_ngrams <labeled_trees.txt> <out_dir> <n1,n2,n3>
  KaminGo.py print label_ngram_lists <labeled_trees.txt> <out_file> <n1,n2,n3> <min_count>
  KaminGo.py print label_cooc_lists <labeled_trees.txt> <out.csv>
  KaminGo.py print label_passes <labeled_trees.txt> <out_dir>
  KaminGo.py print trees_statistics <trees.txt> <stats.csv>
  KaminGo.py print label_stats <trees.txt> <stats_dir>
  KaminGo.py print forward_backward_transitions <trees.txt> <out_dir> <n1,n2,n3>
  KaminGo.py prefit_adders <data_prefix> [--adders=<bglptdr>]
  KaminGo.py grid_search <data_prefix> <pipe_prefix> <params_prefix> [--cv=<cv> --split_tags=<k,n> --target_tags=<t1,t2,t3> --adders=<bgltspdr>]
  KaminGo.py train_test <train> <test> <pipe> <params> <mode> [--target_tags=<t1,t2,t3> --adders=<bgltspdr> --cls=<NB>]
  KaminGo.py aggregate_train_test <train> <test> <pipe> <params> <mode> [--target_tags=<t1,t2,t3> --adders=<bgltspdr> --cls=<NB>]
  KaminGo.py lstmsa <data_prefix> <pipes> <hidden_dims> [--target_tags=<t1,t2,t3>]
  KaminGo.py aggregate_scores <data_prefix> <pipe_prefix>
  KaminGo.py aggregate_scores_lstmsa <data_prefix> [--tags]
  KaminGo.py create_doc2vec_train_file <trees.txt> <out_file.txt>
  KaminGo.py train_doc2vec <train_file.txt> <out_model.txt> <epochs> <dim>
  KaminGo.py prepare_dissent <trees.txt> <out_path>
  KaminGo.py merge_disprobas <probas_dir> <out_file.dispr>
  KaminGo.py print dis_branch <trees.txt> <probas.dispr> <branch_atlas_id> <out_file.csv>
  KaminGo.py print dis_tags_npmi <trees.txt> <probas.dispr> <out.csv> [--just_count --just_pmi]


Info:
  just_count: rather than npmi, print only counts.
  log: print all the evidences into a log file.
  aggregate_labels: goes over the bundles directory (it should contain .atlproj files only),
                    all_trees file should contain non-labeled trees (typically a big file named 'all_trees.txt')
                    prefix will be added to output files
  create_bundles: creates bundles to work in atlas
  rework_labels: Consolidating, Ignoring & Merging labels according to the given settings:
                {
                    "dont_consolidate": [list_of_labels],
                    "ignore_labels": [list_of_labels],
                    "merge_labels": [   (  [from_list_of_labels] , to_label  )   ]
                }
  create_df: creating data frame to be used for basic classification tasks, with optional 'rework_labels' preprocessing
  print label_details: prints detailed table describing all labels for given trees, per node & annotator
  print label_priors: prints label distribution over the given trees, --per_tree - with per-tree detalization
  print label_ngrams: prints & counts label n-grams over the given trees, one file per ngram & each tree, one for all trees
  print tree_statistics: tree stats (node count, branches count, avg branches, etc...)


Known issues:
  (scrapping.py) BeautifulSoup 'xml' unknown feature? Try: pip install -U lxml
"""

from docopt import docopt
import scrapping, TreeTools as tt, LabelTools as lt, ClassTools as ct, FeatureTools as ft
from DisSentTools import DisSentTools as dst
import sys, tempfile

def main(argv):
    if len(argv) == 0:
        # print 'Usage: my_program command --option <argument>'
        print(__doc__)
    else:
        args = docopt(__doc__, argv=argv)
        if args['aggregate_labels']:
            with tempfile.TemporaryDirectory() as tmpdir:
                scrapping.scrap(args['<bundles_dir>'], tmpdir)
                notcut_trees = args['<out_dir>'] + '/' + args['<prefix>'] + '_notcut_trees.txt'
                notcut_priors = args['<out_dir>'] + '/' + args['<prefix>'] + '_notcut_priors.csv'
                cut_trees = args['<out_dir>'] + '/' + args['<prefix>'] + '_cut_trees.txt'
                split_trees = args['<out_dir>'] + '/' + args['<prefix>'] + '_split_trees.txt'
                withdummy_trees = args['<out_dir>'] + '/' + args['<prefix>'] + '_withdummy_trees.txt'
                cut_priors = args['<out_dir>'] + '/' + args['<prefix>'] + '_cut_priors.csv'
                split_priors = args['<out_dir>'] + '/' + args['<prefix>'] + '_split_priors.csv'
                out_df = args['<out_dir>'] + '/' + args['<prefix>'] + '_df.csv'
                lt.aggregate_labels_from_bundles_to_trees(args['<all_trees.txt>'], tmpdir, notcut_trees)
                tt.remove_duplicate_nodes(notcut_trees, notcut_trees)
                print('Duplicate nodes were removed.')
                tt.translate_list_of_trees(notcut_trees, notcut_trees)
                print('Trees were translated.')
                lt.remove_2nd_tags(notcut_trees)
                print('2nd tags were removed.')
                lt.print_label_priors(notcut_trees, notcut_priors)
                lt.cut_non_labeled_branches(notcut_trees, cut_trees)
                print('Non-labeled branches were cut.')
                lt.apply_split_labels_to_trees(cut_trees, split_trees)
                lt.print_label_priors(split_trees, split_priors)
                print('Split labels were applied.')
                lt.apply_start_split_end_labels_to_trees(cut_trees, withdummy_trees)
                print('Start-Split-End labels were applied.')
                lt.print_label_priors(cut_trees, cut_priors)
                lt.create_data_csv(split_trees, out_df)
                print('Labels aggregated successfully.')
        if args['rename_labels']:
            lt.rename_labels(args['<in_file>'], args['<out_file>'])
        if args['rename_tags_in_trees']:
            lt.rename_tags_in_trees(args['<trees.txt>'], args['<out_trees.txt>'])
        if args['rename_tags_in_df']:
            lt.rename_tags_in_df(args['<df.csv>'], args['<out_df.csv>'])
        if args['create_bundles']:
            tt.create_bundles(args['<all_trees.txt>'], args['<tree_ids>'], args['<out_dir>'])
        if args['rework_labels']:
            lt.rework_labels(args['<labeled_trees.txt>'], args['<rework_settings.txt>'], args['<out_trees.txt>'])
        if args['create_df']:
            lt.create_data_csv(args['<labeled_trees.txt>'], args['<out_df.csv>'], args['--rework_settings'], args['--ignore_deleted'])
        if args['print']:
            if args['dis_branch']:
                dst.print_branch(trees_path=args['<trees.txt>'], probas_path=args['<probas.dispr>'],
                                 branch_atlas_id=args['<branch_atlas_id>'], out_file=args['<out_file.csv>'])

            if args['dis_tags_npmi']:
                dst.print_dis_tags_npmi(trees_path=args['<trees.txt>'], probas_path=args['<probas.dispr>'],
                                        out_file=args['<out.csv>'], just_count=args['--just_count'], just_pmi=args['--just_pmi'])
            if args['tags_npmi']:
                lt.print_tags_npmi_table(args['<labeled_trees.txt>'], args['<out.csv>'],
                                                 args['--just_count'], args['--log'])
            if args['label_priors']:
                lt.print_label_priors(args['<labeled_trees.txt>'], args['<out.csv>'], args['--per_tree'])

            if args['label_details']:
                lt.print_label_details(args['<labeled_trees.txt>'], args['<out.csv>'])

            if args['label_cooc_lists']:
                lt.print_label_cooc_lists(args['<labeled_trees.txt>'], args['<out.csv>'])

            if args['label_passes']:
                lt.print_label_passes(args['<labeled_trees.txt>'], args['<out_dir>'])

            if args['label_ngrams']:
                lt.print_label_ngrams(args['<labeled_trees.txt>'], args['<out_dir>'], [int(n) for n in args['<n1,n2,n3>'].split(',')])

            if args['forward_backward_transitions']:
                lt.print_forward_backward_transitions(args['<labeled_trees.txt>'], args['<out_dir>'], [int(n) for n in args['<n1,n2,n3>'].split(',')])

            if args['label_ngram_lists']:
                lt.print_label_ngram_lists(args['<labeled_trees.txt>'], args['<out_file>'], [int(n) for n in args['<n1,n2,n3>'].split(',')], args['<min_count>'])

            if args['trees_statistics']:
                tt.create_list_of_trees_statistics(args['<trees.txt>'], args['<stats.csv>'])

            if args['label_stats']:
                trees_path = args['<trees.txt>']
                out_dir = args['<stats_dir>']
                priors_csv = out_dir+'/priors.csv'
                npmi = out_dir + '/npmi.csv'
                correlation_log = out_dir+'/corr_log.txt'
                pmi = out_dir + '/pmi.csv'
                matthews = out_dir + '/matthews_correlation.csv'
                together_counts = out_dir + '/together_counts.csv'
                general_stats = out_dir + '/general_stats.csv'
                lt.print_label_ngrams(trees_path, out_dir, [2, 3, 4, 5, 6, 7])
                lt.print_label_priors(trees_path, priors_csv)
                lt.print_tags_npmi_table(trees_path, npmi, log_file = correlation_log)
                lt.print_tags_npmi_table(trees_path, pmi, just_pmi = True)
                lt.print_tags_npmi_table(trees_path, together_counts, just_count=True)
                tt.create_list_of_trees_statistics(trees_path, general_stats)
                lt.print_tags_matthews(trees_path, matthews)

        if args['grid_search']:
            cv = None
            if args['--cv']:
                cv = int(args['--cv'])
            target_tags = None
            if args['--target_tags']:
                target_tags = args['--target_tags'].split(',')
            split_tags = None
            if args['--split_tags']:
                split_tags = [int(i) for i in args['--split_tags'].split(',')]
            adders = args['--adders']
            ct.grid_search(args['<data_prefix>'], args['<pipe_prefix>'], args['<params_prefix>'], adders=adders, cv=cv,split_tags=split_tags,target_tags=target_tags)

        if args['train_test']:
            target_tags = None
            if args['--target_tags']:
                target_tags = args['--target_tags'].split(',')
            adders = args['--adders']
            cls = args['--cls']
            ct.train_test(args['<train>'], args['<test>'], args['<pipe>'], args['<params>'], args['<mode>'], classifier=cls, target_tags=target_tags,adders=adders )

        if args['aggregate_train_test']:
            target_tags = None
            if args['--target_tags']:
                target_tags = args['--target_tags'].split(',')
            adders = args['--adders']
            cls = args['--cls']
            ct.aggregate_traintest(args['<train>'], args['<test>'], args['<pipe>'], args['<params>'], args['<mode>'], classifier=cls, target_tags=target_tags,adders=adders)


        if args['lstmsa']:
            ct.lstm_sa(data_prefix=args['<data_prefix>'], target_tags=args['--target_tags'],
                       pipes=args['<pipes>'], hidden_dims=args['<hidden_dims>'])

        if args['aggregate_scores']:
            ct.aggregate_scores(args['<data_prefix>'], args['<pipe_prefix>'])

        if args['aggregate_scores_lstmsa']:
            ct.aggregate_scores_lstmsa(args['<data_prefix>'], args['--tags'])

        if args['create_doc2vec_train_file']:
            ft.create_d2vtrain_lines(args['<trees.txt>'], args['<out_file.txt>'])

        if args['train_doc2vec']:
            ft.train_doc2vec(args['<train_file.txt>'], args['<out_model.txt>'], int(args['<epochs>']), int(args['<dim>']))

        if args['prefit_adders']:
            ft.prefit_adders(args['<data_prefix>'], args['--adders'])

        if args['prepare_dissent']:
            dst.run_trees(args['<trees.txt>'], args['<out_path>'])

        if args['merge_disprobas']:
            dst.merge_probas(probas_dir=args['<probas_dir>'], out_path=args['<out_file.dispr>'])

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
