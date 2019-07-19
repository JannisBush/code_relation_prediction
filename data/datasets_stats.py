import os
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def count_args(x):
    return x[['org', 'response']].stack().nunique()


def count_values(x, labels):
    return x['label'].loc[x['label'].isin(labels)].count()


def add_sum(x, labels):
    sums = x[labels].sum()
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
    df['irr'] = 'irr' # Column where every row has the same value to be able to use groupby apply

    # Create stats by topic, by org argument and by resp argument
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended']:
        data_stats_topic[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby('topic').apply(
            lambda r: pd.Series({'Topic': r['topic'].iloc[0], 'Unique arguments': count_args(r),
                                 'Total pairs': count_values(r, ['attack', 'support']),
                                 'Attack': count_values(r, ["attack"]), 'Support': count_values(r, ["support"]),
                                 'Mean total tokens': r['complete_len'].mean(), 'Median total tokens': r['complete_len'].median(),
                                 'Max total tokens': r['complete_len'].max()})).reset_index(drop=True)

        data_stats_topic[data_set] = data_stats_topic[data_set].append(df.loc[df["org_dataset"].isin([data_set])].groupby('irr').apply(
            lambda r: pd.Series({'Topic': 'Total', 'Unique arguments': count_args(r),
                                 'Total pairs': count_values(r, ['attack', 'support']),
                                 'Attack': count_values(r, ["attack"]), 'Support': count_values(r, ["support"]),
                                 'Mean total tokens': r['complete_len'].mean(), 'Median total tokens': r['complete_len'].median(),
                                 'Max total tokens': r['complete_len'].max()})).reset_index(drop=True)).reset_index(drop=True)
        #data_stats_topic[data_set] = add_sum(data_stats_topic[data_set], ['Total pairs', 'Unique arguments', 'attack', 'support'])

        data_stats_org[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby(['org'], as_index=False).apply(
            lambda r: pd.Series({'Attacked': count_values(r, ["attack"]), 'Supported': count_values(r, ["support"]),
                                 'Total pairs': count_values(r, ['attack', 'support'])}))

        data_stats_org[data_set] = add_sum(data_stats_org[data_set], ["Attacked", "Supported", "Total pairs"])

        data_stats_resp[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby(['response'], as_index=False).apply(
            lambda r: pd.Series({'Attacks': count_values(r, ["attack"]), 'Supports': count_values(r, ["support"]),
                                 'Total pairs': count_values(r, ['attack', 'support'])}))

        data_stats_resp[data_set] = add_sum(data_stats_resp[data_set], ["Attacks", "Supports", "Total pairs"])

    # For political include unrelated!
    data_set = 'political'
    data_stats_topic[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby('topic').apply(
        lambda r: pd.Series(
            {'Topic': r['topic'].iloc[0], 'Unique arguments': count_args(r), 'Total pairs': count_values(r, ['attack', 'support', 'unrelated']),
             'Attack': count_values(r, ["attack"]), 'Support': count_values(r, ["support"]),
             'Unrelated': count_values(r, ['unrelated']),
             'Mean total tokens': r['complete_len'].mean(), 'Median total tokens': r['complete_len'].median(),
             'Max total tokens': r['complete_len'].max()}))

    data_stats_topic[data_set] = data_stats_topic[data_set].append(
        df.loc[df["org_dataset"].isin([data_set])].groupby('irr').apply(
            lambda r: pd.Series({'Topic': "Total", 'Unique arguments': count_args(r), 'Total pairs': count_values(r, ['attack', 'support', 'unrelated']),
             'Attack': count_values(r, ["attack"]), 'Support': count_values(r, ["support"]),
             'Unrelated': count_values(r, ['unrelated']),
             'Mean total tokens': r['complete_len'].mean(), 'Median total tokens': r['complete_len'].median(),
             'Max total tokens': r['complete_len'].max()})).reset_index(drop=True)).reset_index(
        drop=True)
    #data_stats_topic[data_set] = add_sum(data_stats_topic[data_set], ['Total pairs', 'Unique arguments', 'attack', 'support', 'unrelated'])
    #data_stats_topic[data_set].loc[data_stats_topic[data_set]['topic'].isna(), 'topic'] = 'Total'

    data_stats_org[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby(['org'],
                                                                                                    as_index=False).apply(
        lambda r: pd.Series({'Attacked': count_values(r, ["attack"]), 'Supported': count_values(r, ["support"]),
                             'Unrelated': count_values(r, ["unrelated"]),
                             'Total pairs': count_values(r, ['attack', 'support', 'unrelated']),
                             }))

    data_stats_org[data_set] = add_sum(data_stats_org[data_set], ["Attacked", "Supported", "Total pairs"])

    data_stats_resp[data_set] = df.loc[df["org_dataset"].isin([data_set])].groupby(['response'],
                                                                                                     as_index=False).apply(
        lambda r: pd.Series({'Attacks': count_values(r, ["attack"]), 'Supports': count_values(r, ["support"]),
                             'Unrelated': count_values(r, ["unrelated"]),
                             'Total pairs': count_values(r, ['attack', 'support', 'unrelated']),
                             }))

    data_stats_resp[data_set] = add_sum(data_stats_resp[data_set], ["Attacks", "Supports", "Total pairs", "Unrelated"])

    # For political groupby Nixon-Kennedy in addition to by topic
    # print(df.loc[df["org_dataset"].isin([data_set])].groupby(["response_stance", "org_stance"])["label"].value_counts())
    data_stats_author = df.loc[df["org_dataset"].isin([data_set])].groupby(
        ["response_stance", "org_stance"]).apply(
        lambda r: pd.Series(
            {'Author resp': r['response_stance'].iloc[0],
             'Author org': r['org_stance'].iloc[0],
             'Unique arguments': count_args(r),
             'Total pairs': count_values(r, ['attack', 'support', 'unrelated']),
             'Attack': count_values(r, ["attack"]), 'Support': count_values(r, ["support"]),
             'Unrelated': count_values(r, ['unrelated']),
             'Mean total tokens': r['complete_len'].mean(), 'Median total tokens': r['complete_len'].median(),
             'Max total tokens': r['complete_len'].max()}))

    #data_stats_author = add_sum(data_stats_author, ['Total pairs', 'Unique arguments', 'attack', 'support', 'unrelated'])
    data_stats_author = data_stats_author.append(
        df.loc[df["org_dataset"].isin([data_set])].groupby('irr').apply(
            lambda r: pd.Series({'Author resp': 'Total',
             'Author org': "",
             'Unique arguments': count_args(r),
             'Total pairs': count_values(r, ['attack', 'support', 'unrelated']),
             'Attack': count_values(r, ["attack"]), 'Support': count_values(r, ["support"]),
             'Unrelated': count_values(r, ['unrelated']),
             'Mean total tokens': r['complete_len'].mean(), 'Median total tokens': r['complete_len'].median(),
             'Max total tokens': r['complete_len'].max()})).reset_index(drop=True)).reset_index(
        drop=True)

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
            {'Dataset': r['org_dataset'].iloc[0], 'Unique arguments': count_args(r),
             'Total pairs': count_values(r, ['attack', 'support', 'unrelated', 'agreement', 'disagreement']),
             'Attack/Disagreement': count_values(r, ["attack", "disagreement"]),
             'Support/Agreement': count_values(r, ["support", "agreement"]),
             'Unrelated': count_values(r, ['unrelated']),
             'Mean total tokens': r['complete_len'].mean(), 'Median total tokens': r['complete_len'].median(),
             'Max total tokens': r['complete_len'].max()}))

    data_stats_total = data_stats_total.append(
        df.groupby('irr').apply(
            lambda r: pd.Series({'Dataset': 'Total', 'Unique arguments': count_args(r),
             'Total pairs': count_values(r, ['attack', 'support', 'unrelated', 'agreement', 'disagreement']),
             'Attack/Disagreement': count_values(r, ["attack", "disagreement"]),
             'Support/Agreement': count_values(r, ["support", "agreement"]),
             'Unrelated': count_values(r, ['unrelated']),
             'Mean total tokens': r['complete_len'].mean(), 'Median total tokens': r['complete_len'].median(),
             'Max total tokens': r['complete_len'].max()})).reset_index(drop=True)).reset_index(
        drop=True)
    #data_stats_total = add_sum(data_stats_total,['Total pairs', 'Unique arguments', 'support/agreement', 'attack/disagreement', 'unrelated'])

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
    if not os.path.exists('thesis'):
        os.makedirs('thesis')
    data_stats_author.to_csv('./stats/data_stats_author.tsv', encoding='utf-8', sep='\t', index=False)
    data_nix_ken.to_csv('./stats/data_nix_ken.tsv', encoding='utf-8', sep='\t', index=False)
    data_stats_total.to_csv('./stats/data_stats_total.tsv', encoding='utf-8', sep='\t', index=False)
    np.save('./stats/sent_stats.npy', sent_stats)
    np.save('./stats/data_stats_org.npy', data_stats_org)
    np.save('./stats/data_stats_resp.npy', data_stats_resp)
    np.save('./stats/data_stats_topic.npy', data_stats_topic)

    # Save for thesis
    data_stats_topic['debate_train'].to_csv('./thesis/debate_train_topics.csv', encoding='utf-8', index=False)
    data_stats_topic['debate_test'].to_csv('./thesis/debate_test_topics.csv', encoding='utf-8', index=False)
    data_stats_topic['procon'].to_csv('./thesis/procon_topics.csv', encoding='utf-8', index=False)

