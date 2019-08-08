import os

import numpy as np
import pandas as pd


def count_args(x):
    """Counts number of unique arguments (org and response combined)."""
    return x[['org', 'response']].stack().nunique()


def count_values(x, labels):
    """Count how many rows have a label contained in labels."""
    return x['label'].loc[x['label'].isin(labels)].count()


def count_pairs(x):
    """Count all pairs and return as a Series."""
    return pd.Series({'Total pairs': count_values(x, ['attack', 'support', 'unrelated', 'agreement', 'disagreement']),
                      'Attack': count_values(x, ["attack"]), 'Support': count_values(x, ["support"]),
                      'Unrelated': count_values(x, ['unrelated']), 'Agreement': count_values(x, ['agreement']),
                      'Disagreement': count_values(x, ['disagreement'])})


def calc_tokens(x):
    """Calculate median and max tokens and return as Series."""
    return pd.Series({'Median combined tokens': x['complete_len'].median(),
                      'Max combined tokens': x['complete_len'].max()})


def count_topic_pairs_tokens(x):
    """Count topics, pairs and tokens and return as Series."""
    topic = x.name if x.name != 'irr' else 'Total'
    return pd.concat([pd.Series({'Topic': topic, 'Unique arguments': count_args(x)}),count_pairs(x), calc_tokens(x)])


def count_author_pairs_tokens(x):
    """Count author, pairs and tokens and returns as Series."""
    if x.name == 'irr':
        author_resp = 'Total'
        author_org = ''
    else:
        author_resp = x.name[0]
        author_org = x.name[1]
    return pd.concat([pd.Series({'Author resp': author_resp, 'Author org':
        author_org, 'Unique arguments': count_args(x)}), count_pairs(x), calc_tokens(x)])


def count_dataset_pairs_tokens(x):
    """Count dataset, pairs and tokens and returns as Series."""
    dataset = x.name if x.name != 'irr' else 'Total'
    return pd.concat([pd.Series({'Dataset': dataset, 'Unique arguments': count_args(x)}),
                      count_pairs(x), calc_tokens(x)])


if __name__ == '__main__':
    """Calculate several statistic over the datasets and saves them."""

    # Read the complete data tsv and create empty dicts.
    df = pd.read_csv('./complete_data.tsv', sep='\t')
    data_stats_topic = {}
    data_stats_org = {}
    data_stats_resp = {}
    data_stats_author = {}
    # Column where every row has the same value to be able to use groupby apply on the complete df.
    df['irr'] = 'irr'
    # List to rename some columns in dataframes.
    rename_list = [{}, {"Attack": "Attacked", "Support": "Supported"}, {"Attack": "Attacks", "Support": "Supports"}, {}]

    # Create stats by different features for every dataset.
    for data_set in ['debate_test', 'debate_train', 'procon', 'political', 'agreement']:
        # Select the data for the dataset.
        du = df.loc[df["org_dataset"].isin([data_set])]
        # Loop over all features (topic, orgs, resp, author).
        for i, curr_dict, func in [(0, data_stats_topic, count_topic_pairs_tokens),
                                   (1, data_stats_org, count_pairs), (2, data_stats_resp, count_pairs),
                                   (3, data_stats_author, count_author_pairs_tokens)]:
            # Stats by topic
            if i == 0:
                cd = du.groupby('topic').apply(func).reset_index(drop=True)
            # Stats by org
            elif i == 1:
                cd = du.groupby(['org'], as_index=False).apply(func).reset_index(drop=True)
            # Stats by resp
            elif i == 2:
                cd = du.groupby(['response'], as_index=False).apply(func).reset_index(drop=True)
            # By author
            else:
                # Only for political
                if not data_set == 'political':
                    continue
                else:
                    cd = du.groupby(["response_stance", "org_stance"]).apply(func)
                    # Unique arguments for Nixon and Kennedy.
                    data_use1 = du[['org_stance', 'org']].rename(index=str, columns={
                        "org_stance": "author", "org": "text"})
                    data_use2 = du[['response_stance', 'response']].rename(index=str, columns={
                        "response_stance": "author", "response": "text"})
                    data_nix_ken = data_use1.append(data_use2)

            # Total row
            cd = cd.append(du.groupby('irr').apply(func).reset_index(drop=True)).reset_index(drop=True)
            # Renaming and deleting empty columns.
            cd = cd.rename(columns=rename_list[i])
            cd = cd.loc[:, (cd != 0).any(axis=0)]
            # Save in the correct dict.
            curr_dict[data_set] = cd

    # Overall stats of the different datasets.
    data_stats_total = df.groupby('org_dataset').apply(
        count_dataset_pairs_tokens)
    # Total row
    data_stats_total = data_stats_total.append(
        df.groupby('irr').apply(
            count_dataset_pairs_tokens).reset_index(drop=True)).reset_index(
        drop=True)

    # Create folder to save, if they do not exist already.
    if not os.path.exists('stats'):
        os.makedirs('stats')
    if not os.path.exists('thesis'):
        os.makedirs('thesis')

    # Save dataframes to analyze in results_showcase.ipynb.
    data_stats_author['political'].to_csv('./stats/data_stats_author.tsv', encoding='utf-8', sep='\t', index=False)
    data_nix_ken.to_csv('./stats/data_nix_ken.tsv', encoding='utf-8', sep='\t', index=False)
    data_stats_total.to_csv('./stats/data_stats_total.tsv', encoding='utf-8', sep='\t', index=False)
    np.save('./stats/data_stats_org.npy', data_stats_org)
    np.save('./stats/data_stats_resp.npy', data_stats_resp)
    np.save('./stats/data_stats_topic.npy', data_stats_topic)

    # Save topic dataframes for direct usage in thesis.
    data_stats_topic['debate_train'].to_csv('./thesis/debate_train_topics.csv', encoding='utf-8', index=False)
    data_stats_topic['debate_test'].to_csv('./thesis/debate_test_topics.csv', encoding='utf-8', index=False)
    data_stats_topic['procon'].to_csv('./thesis/procon_topics.csv', encoding='utf-8', index=False)
    data_stats_topic['political'].to_csv('./thesis/political_topics.tsv', encoding='utf-8', sep='\t', index=False)
    data_stats_topic['agreement'].to_csv('./thesis/agreement_topics.csv', encoding='utf-8', index=False)
