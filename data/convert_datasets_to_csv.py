import re
import copy
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from bert import tokenization
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import scattertext as st
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


arg_dict = {'id': [], 'org': [], 'org_stance': [], 'response': [], 'response_stance': [], 'label': [], 'topic': []}


def create_data_from_xml_com_arg(path):
    """

    :param path: Path to the ComArg dataset
    :return: the ComArg dataset as a pandas dataframe
    """
    root = ET.parse(path).getroot()
    argument_dict = copy.deepcopy(arg_dict)
    topic = re.search('comarg/(.*).xml', path).group(1)

    for unit in root.findall('unit'):
        # print(unit.tag, unit.attrib)
        argument_dict['id'].append(unit.attrib['id'])
        argument_dict['org'].append(unit.find('argument').find('text').text)
        argument_dict['org_stance'].append(unit.find('argument').find('stance').text)
        argument_dict['response'].append(unit.find('comment').find('text').text)
        argument_dict['response_stance'].append(unit.find('comment').find('stance').text)
        argument_dict['label'].append(unit.find('label').text)
        argument_dict['topic'].append(topic)

    df = pd.DataFrame(data=argument_dict)
    df = df.replace({'label': {'1': 'A', '2': 'a', '3': 'N', '4': 's', '5': 'S'}})
    return df


def create_data_from_xml_debatepedia(path, replacement):
    """
    Works for the original dataset. Not the extended ones

    :param replacement:
    :param path: Path to the NoDe debatepedia or procon dataset
    :return: the debatepedia dataset as a pandas dataframe
    """
    root = ET.parse(path).getroot()
    argument_dict = copy.deepcopy(arg_dict)

    for unit in root.findall('pair'):
        # print(unit.tag, unit.attrib)
        argument_dict['id'].append(unit.attrib['id'])
        argument_dict['org'].append(unit.find('h').text)
        argument_dict['org_stance'].append('unknown')
        argument_dict['response'].append(unit.find('t').text)
        argument_dict['response_stance'].append('unknown')
        argument_dict['label'].append(unit.attrib['entailment'])
        argument_dict['topic'].append(unit.attrib['topic'])

    df = pd.DataFrame(data=argument_dict)
    df = df.replace(replacement)
    return df


def create_data_from_xml_debatepedia_extended(path):
    """
    Works for the extended ones

    :param path: Path to the NoDe debatepedia datasets
    :return: the debatepedia dataset as a pandas dataframe
    """
    root = ET.parse(path).getroot()
    argument_dict = copy.deepcopy(arg_dict)
    for unit in root.findall('pair'):
        # There are some malformatted arguments in the dataset (t,t instead of t,h)
        # Ignore them for now
        if unit.find('h') is None:
            print("Malformatted: {}".format(unit.attrib))
            continue
        try:
            argument_dict['id'].append(unit.attrib['id'])
            argument_dict['org'].append(unit.find('h').text)
            argument_dict['org_stance'].append('unknown')
            argument_dict['response'].append(unit.find('t').text)
            argument_dict['response_stance'].append('unknown')
            argument_dict['topic'].append(unit.attrib['topic'])
            argument_dict['label'].append(unit.attrib['argument'])
        # There is a typo in the dataset (aargument instead of argument)
        # Add explicitly
        except KeyError as e:
            print("Error: {}, malformatted {}".format(e, unit.attrib))
            argument_dict['label'].append(unit.attrib['aargument'])

    df = pd.DataFrame(data=argument_dict)
    return df


def create_data_from_tsv(path):
    """

    :param path: Path to the PoliticalArgumentation dataset
    :return: the politicalArgumentation dataset as a pandas dataframe
    """
    df = pd.read_csv(path, sep='\t')
    df = df.rename(index=str, columns={'pair_id': 'id', 'relation': 'label', 'argument1': 'response', 'argument2': 'org',
                                       'source_arg_1': 'response_stance', 'source_arg_2': 'org_stance'})
    df = df.replace({'label': {'no_relation': 'unrelated'}})
    #df = df.replace({'response_stance': r'.*'}, {'response_stance': 'unknown'}, regex=True)
    #df = df.replace({'org_stance': r'.*'}, {'org_stance': 'unknown'}, regex=True)
    df = df.replace({'org_stance': r'Kennedy.*'}, {'org_stance': 'Kennedy'}, regex=True)
    df = df.replace({'response_stance': r'Kennedy.*'}, {'response_stance': 'Kennedy'}, regex=True)
    df = df.replace({'org_stance': r'Nixon.*'}, {'org_stance': 'Nixon'}, regex=True)
    df = df.replace({'response_stance': r'Nixon.*'}, {'response_stance': 'Nixon'}, regex=True)

    return df


def create_data_from_tsv_agreement(path):
    """

    :param path: Path to the PoliticalArgumentation dataset
    :return: the politicalArgumentation dataset as a pandas dataframe
    """
    df = pd.read_csv(path, sep='\t', header=None, names=['id', 'label', 'topic', 'org_stance', 'response_stance',
                                                         'org', 'response'])
    df = df.replace({'label': {'no_relation': 'unrelated'}})
    df = df.replace({'response_stance': r'.*'}, {'response_stance': 'unknown'}, regex=True)
    df = df.replace({'org_stance': r'.*'}, {'org_stance': 'unknown'}, regex=True)

    return df


def print_summary(dataframe):
    """

    :param dataframe: Dataframe for which some summary statistics are printed
    :return: Nothing
    """
    print(dataframe.head())
    print(dataframe.groupby('label').nunique())
    print(dataframe.describe())


if __name__ == '__main__':
    pd.set_option('display.max_columns', 7)
    pd.set_option('display.max_rows', 999)
    paths_comarg = ['../../data/good_ComArg/comarg/GM.xml','../../data/good_ComArg/comarg/UGIP.xml']
    paths_debatepedia = ['../../data/good_depatepedia/debatepedia/debatepedia_test.xml',
                         '../../data/good_depatepedia/debatepedia/debatepedia_train.xml']
    # procon
    paths_procon = ['../../data/good_depatepedia/debatepedia/procon.xml']
    # extended debatepedia
    paths_extended = ['../../data/good_depatepedia/extended/debatepediaExtended.xml',
                      '../../data/good_depatepedia/extended_attacks/extended_attacks.xml',
                      '../../data/good_depatepedia/extended_attacks/mediated_attacks.xml',
                      '../../data/good_depatepedia/extended_attacks/secondary_attacks.xml',
                      '../../data/good_depatepedia/extended_attacks/supported_attacks.xml']
    # political
    paths_political = ['../../data/good_PoliticalArgumentation/balanced_dataset.tsv']

    # other datasets?
    paths_agreement = ['../../data/maybe_agreement_disagreement/debatepedia_agreement_dataset.tsv']

    data = []
    for path in paths_comarg:
        data.append(create_data_from_xml_com_arg(path))
    for path in paths_debatepedia:
        data.append(create_data_from_xml_debatepedia(path,
                                                        {'label': {'NO': 'attack', 'YES': 'support'}}))
    for path in paths_procon:
        data.append(create_data_from_xml_debatepedia(path,
                                                         {'label': {'NONENTAILMENT': 'attack', 'ENTAILMENT': 'support'}}))
    for path in paths_extended:
        data.append(create_data_from_xml_debatepedia_extended(path))

    for path in paths_political:
        data.append(create_data_from_tsv(path))

    for path in paths_agreement:
        data.append(create_data_from_tsv_agreement(path))

    #for data_set in data:
        #print_summary(data_set)

    # join all datasets
    df_complete = pd.concat(data, keys=['comargGM', 'comargUGIP', 'debate_test',
                                        'debate_train', 'procon', 'debate_extended', 'debate_ext_attacks',
                                        'debate_ext_media', 'debate_ext_second', 'debate_ext_supp', 'political',
                                        'agreement'],
                            sort=False, names=['org_dataset'])
    # Remove the index (org_dataset) and use a column instead
    df_complete = df_complete.reset_index(level='org_dataset')
    print_summary(df_complete)

    # remove all duplicates
    # There are some in ComArg, Political and in the extended version of NoDe (the ids are mixed up, therefore ignore)
    # In two cases even the labels are incorrect, therefore ignore
    # print(df_complete[(df_complete.duplicated(subset=['org', 'response'], keep=False)) & (df_complete['org_dataset'].isin(['political']))])
    #df_complete = df_complete.drop_duplicates(subset=['org', 'response'])


    # Here, we only select the rows with correct labels (ignore ComArg)
    data_to_use = df_complete.loc[df_complete['label'].isin(['attack','support','unrelated'])]
    # Select everything except for 'political'
    print_summary(data_to_use.loc[~data_to_use['org_dataset'].isin(['political'])])

    ### 3 methods to do the same, if we have not converted the dataset index to a column
    # print_summary(data_to_use.query("org_dataset != 'political'"))
    # print_summary(data_to_use[~data_to_use.index.get_level_values('org_dataset').isin(['political'])])
    # print_summary(data_to_use.loc[data_to_use.index.get_level_values('org_dataset') != 'political'])
    ###

    print(data_to_use.keys())
    print(data_to_use.index)

    #df_complete.to_csv('complete_data.tsv', encoding='utf-8', sep='\t', index=False)

    #### NEW

    # Filter only relevant datasets
    df_complete = df_complete[df_complete["org_dataset"].isin(['debate_test',
                                        'debate_train', 'procon', 'debate_extended', 'political'])]

    BERT_VOCAB = './bert/uncased_L-12_H-768_A-12/vocab.txt'
    BERT_INIT_CHKPNT = './bert/uncased_L-12_H-768_A-12/bert_model.ckpt'

    # Create tokenizer
    tokenization.validate_case_matches_checkpoint(True, BERT_INIT_CHKPNT)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=BERT_VOCAB, do_lower_case=True)

    # Count length in WordPiece tokens
    # How many tokens (wordpiece tokenizer in org and response)
    df_complete["org_len"] = df_complete.apply(lambda row: len(tokenizer.tokenize(row.org)), axis=1)
    df_complete["response_len"] = df_complete.apply(lambda row: len(tokenizer.tokenize(row.response)), axis=1)
    df_complete["complete_len"] = df_complete.apply(lambda row: row.org_len + row.response_len, axis=1)

    def count_args(x):
        return x[['org', 'response']].stack().nunique()

    def count_values(x, labels):
        return x['label'].loc[x['label'].isin(labels)].count()

    def add_sum(x, labels):
        sums = x.sum(numeric_only=True)
        x = x.append(sums, ignore_index=True)
        x[labels] = x[labels].apply(np.int64)
        return x

    data_stats_topic = {}
    data_stats_org = {}
    data_stats_resp = {}

    # Create stats by topic, by org argument and by resp argument
    for data_set in ['debate_test','debate_train', 'procon', 'debate_extended']:

        data_stats_topic[data_set] = df_complete.loc[df_complete["org_dataset"].isin([data_set])].groupby('topic').apply(
            lambda r: pd.Series({'topic': r['topic'].iloc[0], 'args': count_args(r),
                                 'tot': count_values(r, ['attack', 'support']),
                                 'no': count_values(r, ["attack"]), 'yes': count_values(r, ["support"]),
                                 'mean_total_len': r['complete_len'].mean(), 'median_total_len': r['complete_len'].median(), 'max_total_len': r['complete_len'].max()}))

        data_stats_topic[data_set] = add_sum(data_stats_topic[data_set], ['tot', 'args', 'yes', 'no'])

        data_stats_org[data_set] = df_complete.loc[df_complete["org_dataset"].isin([data_set])].groupby(['org'], as_index=False).apply(
            lambda r: pd.Series({"attacked": count_values(r, ["attack"]), "supported": count_values(r, ["support"]),
                                 'tot': count_values(r, ['attack', 'support'])}))

        data_stats_org[data_set] = add_sum(data_stats_org[data_set], ["attacked", "supported", "tot"])

        data_stats_resp[data_set] = df_complete.loc[df_complete["org_dataset"].isin([data_set])].groupby(['response'],
                                                                                                        as_index=False).apply(
            lambda r: pd.Series({"attacks": count_values(r, ["attack"]), "supports": count_values(r, ["support"]),
                                 'tot': count_values(r, ['attack', 'support'])}))

        data_stats_resp[data_set] = add_sum(data_stats_resp[data_set], ["attacks", "supports", "tot"])

    # For political include unrelated!
    data_set = 'political'
    data_stats_topic[data_set] = df_complete.loc[df_complete["org_dataset"].isin([data_set])].groupby('topic').apply(
        lambda r: pd.Series(
            {'topic': r['topic'].iloc[0], 'args': count_args(r), 'tot': count_values(r, ['attack', 'support', 'unrelated']),
             'no': count_values(r, ["attack"]), 'yes': count_values(r, ["support"]), 'unrelated': count_values(r, ['unrelated']),
             'mean_total_len': r['complete_len'].mean(), 'median_total_len': r['complete_len'].median(), 'max_total_len': r['complete_len'].max()}))

    data_stats_topic[data_set] = add_sum(data_stats_topic[data_set], ['tot', 'args', 'yes', 'no', 'unrelated'])

    data_stats_org[data_set] = df_complete.loc[df_complete["org_dataset"].isin([data_set])].groupby(['org'],
                                                                                                    as_index=False).apply(
        lambda r: pd.Series({"attacked": count_values(r, ["attack"]), "supported": count_values(r, ["support"]),
                             "unrelated": count_values(r, ["unrelated"]), 'tot': count_values(r, ['attack', 'support', 'unrelated']),
                             }))

    data_stats_org[data_set] = add_sum(data_stats_org[data_set], ["attacked", "supported", "tot"])

    data_stats_resp[data_set] = df_complete.loc[df_complete["org_dataset"].isin([data_set])].groupby(['response'],
                                                                                                    as_index=False).apply(
        lambda r: pd.Series({"attacks": count_values(r, ["attack"]), "supports": count_values(r, ["support"]),
                             "unrelated": count_values(r, ["unrelated"]), 'tot': count_values(r, ['attack', 'support', 'unrelated']),
                             }))

    data_stats_resp[data_set] = add_sum(data_stats_resp[data_set], ["attacks", "supports", "tot", "unrelated"])

    # For political groupby Nixon-Kennedy in addition to by topic
    # print(df_complete.loc[df_complete["org_dataset"].isin([data_set])].groupby(["response_stance", "org_stance"])["label"].value_counts())
    data_stats_author = df_complete.loc[df_complete["org_dataset"].isin([data_set])].groupby(["response_stance", "org_stance"]).apply(
        lambda r: pd.Series(
            {'author_resp': r['response_stance'].iloc[0],
             "author_org": r['org_stance'].iloc[0],
            'args': count_args(r),
             'tot': count_values(r, ['attack', 'support', 'unrelated']),
             'no': count_values(r, ["attack"]), 'yes': count_values(r, ["support"]),
             'unrelated': count_values(r, ['unrelated']),
             'mean_total_len': r['complete_len'].mean(), 'median_total_len': r['complete_len'].median(), 'max_total_len': r['complete_len'].max()}))
    data_stats_author = add_sum(data_stats_author, ['tot', 'args', 'yes', 'no', 'unrelated'])

    # Unique arguments for Nixon and Kennedy
    data_use = df_complete.loc[df_complete["org_dataset"].isin([data_set])]
    data_use1 = data_use[['org_stance', 'org']]
    data_use1 = data_use1.rename(index=str, columns={"org_stance": "author", "org": "text"})
    data_use2 = data_use[['response_stance', 'response']]
    data_use2 = data_use2.rename(index=str, columns={"response_stance": "author", "response": "text"})
    data_nix_ken = data_use1.append(data_use2)
    print(data_nix_ken.groupby("author").nunique())

    # Overall stats of the different datasets
    data_stats_total = df_complete.groupby('org_dataset').apply(
        lambda r: pd.Series(
            {'dataset': r['org_dataset'].iloc[0], 'args': count_args(r), 'tot': count_values(r, ['attack', 'support', 'unrelated']),
             'no': count_values(r, ["attack"]), 'yes': count_values(r, ["support"]), 'unrelated': count_values(r, ['unrelated']),
             'mean_total_len': r['complete_len'].mean(), 'median_total_len': r['complete_len'].median(), 'max_total_len': r['complete_len'].max()}))
    data_stats_total = add_sum(data_stats_total, ['tot', 'args', 'yes', 'no', 'unrelated'])

    # List of all stats tables
    data_stats = [data_stats_org, data_stats_author, data_stats_resp, data_stats_topic, data_stats_total]

    # How many individual args (orgs and response) are there (responses can also be used as orgs)
    for data_set in ['debate_test','debate_train', 'procon', 'debate_extended', 'political']:
        df_check = df_complete[df_complete['org_dataset'] == data_set]
        print(data_set)
        print('org', df_check['org'].nunique(), df_check.shape[0])
        print('response', df_check['response'].nunique(), df_check.shape[0])
        print()

    # Plot distribution of length of org, resp and combined over the different datasets
    for data_set in ['debate_test','debate_train', 'procon', 'debate_extended', 'political']:
        df_plot = df_complete[df_complete['org_dataset'] == data_set]
        #axes = df_plot.hist(density=True, sharey=True)
        axes = df_plot.boxplot()
        plt.suptitle(data_set)
        plt.show()

    # Plot how many arguments attack an argument (attack-ratio), also exclude arguments only answered to once
    for data_set in ['debate_test','debate_train', 'procon', 'debate_extended', 'political']:
        df_plot = data_stats_org[data_set].iloc[:-1].apply(
            lambda r: pd.Series({"Attack-ratio": r.attacked/r.tot,
            "Attack-ratio (exluding arguments only attacked/supported once)": np.nan if r.tot == 1 else r.attacked/r.tot}), axis=1)
        # Ratio broken?
        axes = df_plot.hist(density=True)
        plt.suptitle(data_set)
        plt.show()

    # Plot how often an org argument is used
    for data_set in ['debate_test','debate_train', 'procon', 'debate_extended', 'political']:
        df_plot = data_stats_org[data_set].iloc[:-1]
        df_plot = df_plot['tot']
        # Ratio broken?
        axes = df_plot.hist(density=True)
        plt.suptitle(data_set)
        plt.show()

    # Plot how often an response argument is used
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
        df_plot = data_stats_resp[data_set].iloc[:-1]
        df_plot = df_plot['tot']
        # Ratio broken?
        axes = df_plot.hist(bins=np.arange(0,10))
        plt.suptitle(data_set)
        plt.show()

    # Repair debate_test + debate_train (internetaccess -> train, groundzero -> test)
    # Still not the same as described in paper/website (internetaccess 2 missing, military service 2 too much)
    df_complete.loc[(df_complete['topic'] == 'Groundzeromosque') &
                    (df_complete['org_dataset'].isin(['debate_train','debate_test'])), 'org_dataset'] = 'debate_test'
    df_complete.loc[(df_complete['topic'] == 'Internetaccess') & (
        df_complete['org_dataset'].isin(['debate_train', 'debate_test'])), 'org_dataset'] = 'debate_train'

    # Print duplicates
    # Duplicates in political?!
    # 481, 1310 (one labelled as attack one labelled as unrelated) and 1225 + 1393 (both unrelated)
    # What to do with it?
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
        print(data_set + " Duplicates:")
        df_check = df_complete[df_complete['org_dataset'] == data_set]
        print(df_check[df_check.duplicated(subset=['org','response'], keep=False)])

    # Visualize content?
    # WordClouds for orgs and responses (mainly the entities of the debated issues!)
    for data_set in ['debate_test','debate_train', 'procon', 'debate_extended', 'political']:
        df_plot = df_complete[df_complete['org_dataset'] == data_set]
        # Stopwords?
        stopwords = set() # set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords).generate(" ".join(text for text in df_plot['response']))
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(data_set + " Response WordCloud")
        plt.axis("off")
        plt.show()
        wordcloud = WordCloud(stopwords=stopwords).generate(" ".join(text for text in df_plot['org']))
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(data_set + " Org WordCloud")
        plt.axis("off")
        plt.show()

    # wordcloud for kennedy and for nixon
    stopwords = set(STOPWORDS)  # set(STOPWORDS)
    wordcloud = WordCloud(
        stopwords=stopwords).generate(" ".join(text for text in data_nix_ken.loc[data_nix_ken["author"]=='Nixon','text']))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Nixon WordCloud")
    plt.axis("off")
    plt.show()

    stopwords = set(STOPWORDS)  # set(STOPWORDS)
    wordcloud = WordCloud(
        stopwords=stopwords).generate(
        " ".join(text for text in data_nix_ken.loc[data_nix_ken["author"] == 'Kennedy', 'text']))
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Kennedy WordCloud")
    plt.axis("off")
    plt.show()

    # Scattertext attack vs support (only responses)
    # https://kanoki.org/2019/03/17/text-data-visualization-in-python/
    # https://github.com/JasonKessler/scattertext
    nlp = spacy.load('en_core_web_sm')
    for data_set in ['debate_test','debate_train', 'procon', 'debate_extended', 'political']:

        df_plot = df_complete.loc[df_complete['org_dataset'] == data_set]
        df_plot['parsed'] = df_plot['response'].apply(nlp)
        corpus = st.CorpusFromParsedDocuments(df_plot, category_col='label', parsed_col='parsed').build()
        html = st.produce_scattertext_explorer(
            corpus, category='attack', not_category_name='support',
            width_in_pixels=1000, minimum_term_frequency=5, transform=st.Scalers.log_scale_standardize, use_full_doc=True)
        file_name = 'plots/scattertext_attack_support' + data_set + '.html'
        with open(file_name, 'wb') as file:
            file.write(html.encode('utf-8'))

    # Scattertext Nixon vs Kennedy
    df_plot = data_nix_ken
    df_plot['parsed'] = df_plot['text'].apply(nlp)
    corpus = st.CorpusFromParsedDocuments(df_plot, category_col='author', parsed_col='parsed').build()
    html = st.produce_scattertext_explorer(
        corpus, category='Kennedy', not_category_name='Nixon',
        width_in_pixels=1000, minimum_term_frequency=5, transform=st.Scalers.log_scale_standardize, use_full_doc=True)
    file_name = 'plots/scattertext_nixon_kennedy.html'
    with open(file_name, 'wb') as file:
        file.write(html.encode('utf-8'))

    def disc_pol(x):
        if x >= 0.05:
            return 'positive'
        elif x <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    # Sentiment Analysis (Correlation between attack/support and sentiment)
    # Maybe use another sentiment analyser? (With only positive and negative class, without neutral class)
    sid = SentimentIntensityAnalyzer()
    sent_stats = {}
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
        df_sent = df_complete.loc[df_complete['org_dataset'] == data_set]
        df_sent['polarity'] = df_sent['response'].apply(lambda r: sid.polarity_scores(r)['compound'])
        df_sent['discrete_polarity'] = df_sent['polarity'].apply(lambda r: disc_pol(r))
        sent_stats['resp' + data_set] = pd.crosstab(index=df_sent['discrete_polarity'], columns=df_sent['label'], margins=True)
        # print(pd.crosstab(index=df_sent['discrete_polarity'], columns=df_sent['label'], margins=True))
        # Sentiment Analysis (Correlation between sentiment of org and response and label)
        df_sent['polarity_org'] = df_sent['org'].apply(lambda r: sid.polarity_scores(r)['compound'])
        df_sent['discrete_polarity_org'] = df_sent['polarity_org'].apply(lambda r: disc_pol(r))
        # print(pd.crosstab(index=df_sent['label'],
        #                  columns=[df_sent['discrete_polarity'],df_sent['discrete_polarity_org']]))
        sent_stats['org/resp' + data_set] = pd.crosstab(index=df_sent['label'],
                          columns=[df_sent['discrete_polarity'],df_sent['discrete_polarity_org']])
        # Merge to same sentiment and not same sentiment
        df_sent['discrete_polarity_both'] = df_sent.apply(
            lambda r: 'Same' if r['discrete_polarity'] == r['discrete_polarity_org'] else 'Different', axis=1)
        sent_stats['both' + data_set] = pd.crosstab(df_sent['label'], df_sent['discrete_polarity_both'])
        # Correlation coeff etc.

    # Ideas for the future:?

    # Maybe POS tags?

    # Maybe add wordcloud attack vs support (or difference between attack and support)

    # Maybe Plot distribution of top 1,2,3-grams with and without unigrams?
     # https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
    # Or using Scattertext