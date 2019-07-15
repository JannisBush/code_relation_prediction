import os
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def count_args(x):
    return x[['org', 'response']].stack().nunique()


def count_values(x, labels):
    return x['label'].loc[x['label'].isin(labels)].count()


def add_sum(x, labels):
    sums = x.sum(numeric_only=True)
    x = x.append(sums, ignore_index=True)
    x[labels] = x[labels].apply(np.int64)
    return x


def disc_pol(x):
    if x >= 0.05:
        return 'positive'
    elif x <= -0.05:
        return 'negative'
    else:
        return 'neutral'


if __name__ == '__main__':

    df = pd.read_csv('./complete_data.tsv', sep='\t')
    data_stats_topic = {}
    data_stats_org = {}
    data_stats_resp = {}

    # Create stats by topic, by org argument and by resp argument
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended']:
        data_stats_topic[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby('topic').apply(
            lambda r: pd.Series({'topic': r['topic'].iloc[0], 'args': count_args(r),
                                 'tot': count_values(r, ['attack', 'support']),
                                 'attack': count_values(r, ["attack"]), 'support': count_values(r, ["support"]),
                                 'mean_total_len': r['complete_len'].mean(), 'median_total_len': r['complete_len'].median(),
                                 'max_total_len': r['complete_len'].max()}))

        data_stats_topic[data_set] = add_sum(data_stats_topic[data_set], ['tot', 'args', 'attack', 'support'])

        data_stats_org[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby(['org'],
                                                                                                        as_index=False).apply(
            lambda r: pd.Series({"attacked": count_values(r, ["attack"]), "supported": count_values(r, ["support"]),
                                 'tot': count_values(r, ['attack', 'support'])}))

        data_stats_org[data_set] = add_sum(data_stats_org[data_set], ["attacked", "supported", "tot"])

        data_stats_resp[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby(['response'],
                                                                                                         as_index=False).apply(
            lambda r: pd.Series({"attacks": count_values(r, ["attack"]), "supports": count_values(r, ["support"]),
                                 'tot': count_values(r, ['attack', 'support'])}))

        data_stats_resp[data_set] = add_sum(data_stats_resp[data_set], ["attacks", "supports", "tot"])

    # For political include unrelated!
    data_set = 'political'
    data_stats_topic[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby('topic').apply(
        lambda r: pd.Series(
            {'topic': r['topic'].iloc[0], 'args': count_args(r), 'tot': count_values(r, ['attack', 'support', 'unrelated']),
             'attack': count_values(r, ["attack"]), 'support': count_values(r, ["support"]),
             'unrelated': count_values(r, ['unrelated']),
             'mean_total_len': r['complete_len'].mean(), 'median_total_len': r['complete_len'].median(),
             'max_total_len': r['complete_len'].max()}))

    data_stats_topic[data_set] = add_sum(data_stats_topic[data_set], ['tot', 'args', 'attack', 'support', 'unrelated'])

    data_stats_org[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby(['org'],
                                                                                                    as_index=False).apply(
        lambda r: pd.Series({"attacked": count_values(r, ["attack"]), "supported": count_values(r, ["support"]),
                             "unrelated": count_values(r, ["unrelated"]),
                             'tot': count_values(r, ['attack', 'support', 'unrelated']),
                             }))

    data_stats_org[data_set] = add_sum(data_stats_org[data_set], ["attacked", "supported", "tot"])

    data_stats_resp[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby(['response'],
                                                                                                     as_index=False).apply(
        lambda r: pd.Series({"attacks": count_values(r, ["attack"]), "supports": count_values(r, ["support"]),
                             "unrelated": count_values(r, ["unrelated"]),
                             'tot': count_values(r, ['attack', 'support', 'unrelated']),
                             }))

    data_stats_resp[data_set] = add_sum(data_stats_resp[data_set], ["attacks", "supports", "tot", "unrelated"])

    # For political groupby Nixon-Kennedy in addition to by topic
    # print(df.loc[df["org_dataset"].isin([data_set])].groupby(["response_stance", "org_stance"])["label"].value_counts())
    data_stats_author = df.loc[df["org_dataset"].isin([data_set])].groupby(
        ["response_stance", "org_stance"]).apply(
        lambda r: pd.Series(
            {'author_resp': r['response_stance'].iloc[0],
             "author_org": r['org_stance'].iloc[0],
             'args': count_args(r),
             'tot': count_values(r, ['attack', 'support', 'unrelated']),
             'attack': count_values(r, ["attack"]), 'support': count_values(r, ["support"]),
             'unrelated': count_values(r, ['unrelated']),
             'mean_total_len': r['complete_len'].mean(), 'median_total_len': r['complete_len'].median(),
             'max_total_len': r['complete_len'].max()}))
    data_stats_author = add_sum(data_stats_author, ['tot', 'args', 'attack', 'support', 'unrelated'])

    # Unique arguments for Nixon and Kennedy
    data_use = df.loc[df["org_dataset"].isin([data_set])]
    data_use1 = data_use[['org_stance', 'org']]
    data_use1 = data_use1.rename(index=str, columns={"org_stance": "author", "org": "text"})
    data_use2 = data_use[['response_stance', 'response']]
    data_use2 = data_use2.rename(index=str, columns={"response_stance": "author", "response": "text"})
    data_nix_ken = data_use1.append(data_use2)
    print(data_nix_ken.groupby("author").nunique())

    # Overall stats of the different datasets
    data_stats_total = df.groupby('org_dataset').apply(
        lambda r: pd.Series(
            {'dataset': r['org_dataset'].iloc[0], 'args': count_args(r),
             'tot': count_values(r, ['attack', 'support', 'unrelated', 'agreement', 'disagreement']),
             'attack/disagreement': count_values(r, ["attack", "disagreement"]),
             'support/agreement': count_values(r, ["support", "agreement"]),
             'unrelated': count_values(r, ['unrelated']),
             'mean_total_len': r['complete_len'].mean(), 'median_total_len': r['complete_len'].median(),
             'max_total_len': r['complete_len'].max()}))
    data_stats_total = add_sum(data_stats_total,
                               ['tot', 'args', 'support/agreement', 'attack/disagreement', 'unrelated'])

    # List of all stats tables
    data_stats = [data_stats_org, data_stats_author, data_stats_resp, data_stats_topic, data_stats_total]

    # How many individual args (orgs and response) are there (responses can also be used as orgs)
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
        df_check = df[df['org_dataset'] == data_set]
        print(data_set)
        print('org', df_check['org'].nunique(), df_check.shape[0])
        print('response', df_check['response'].nunique(), df_check.shape[0])
        print()

    # Sentiment Analysis (Correlation between attack/support and sentiment)
    # Maybe use another sentiment analyser? (With only positive and negative class, without neutral class)
    sid = SentimentIntensityAnalyzer()
    sent_stats = {}
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
        df_sent = df.loc[df['org_dataset'] == data_set]
        df_sent.loc[:, 'polarity'] = df_sent['response'].apply(lambda r: sid.polarity_scores(r)['compound'])
        df_sent.loc[:, 'discrete_polarity'] = df_sent['polarity'].apply(lambda r: disc_pol(r))
        sent_stats['resp' + data_set] = pd.crosstab(index=df_sent['discrete_polarity'], columns=df_sent['label'],
                                                    margins=True)
        # print(pd.crosstab(index=df_sent['discrete_polarity'], columns=df_sent['label'], margins=True))
        # Sentiment Analysis (Correlation between sentiment of org and response and label)
        df_sent.loc[:, 'polarity_org'] = df_sent['org'].apply(lambda r: sid.polarity_scores(r)['compound'])
        df_sent.loc[:, 'discrete_polarity_org'] = df_sent['polarity_org'].apply(lambda r: disc_pol(r))
        # print(pd.crosstab(index=df_sent['label'],
        #                  columns=[df_sent['discrete_polarity'],df_sent['discrete_polarity_org']]))
        sent_stats['org/resp' + data_set] = pd.crosstab(index=df_sent['label'],
                                                        columns=[df_sent['discrete_polarity'],
                                                                 df_sent['discrete_polarity_org']])
        # Merge to same sentiment and not same sentiment
        df_sent.loc[:, 'discrete_polarity_both'] = df_sent.apply(
            lambda r: 'Same' if r['discrete_polarity'] == r['discrete_polarity_org'] else 'Different', axis=1)
        sent_stats['both' + data_set] = pd.crosstab(df_sent['label'], df_sent['discrete_polarity_both'])
        # Correlation coeff etc.

    # Ideas for the future:?

    # Maybe POS tags?

    # Maybe add wordcloud attack vs support (or difference between attack and support)

    # Maybe Plot distribution of top 1,2,3-grams with and without unigrams?
    # https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
    # Or using Scattertext

    # Save all dataframes
    if not os.path.exists('stats'):
        os.makedirs('stats')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    data_stats_author.to_csv('./stats/data_stats_author.tsv', encoding='utf-8', sep='\t', index=False)
    data_nix_ken.to_csv('./stats/data_nix_ken.tsv', encoding='utf-8', sep='\t', index=False)
    data_stats_total.to_csv('./stats/data_stats_total.tsv', encoding='utf-8', sep='\t', index=False)
    np.save('./stats/sent_stats.npy', sent_stats)
    np.save('./stats/data_stats_org.npy', data_stats_org)
    np.save('./stats/data_stats_resp.npy', data_stats_resp)
    np.save('./stats/data_stats_topic.npy', data_stats_topic)