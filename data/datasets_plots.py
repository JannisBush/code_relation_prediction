import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import scattertext as st
import spacy


if __name__ == '__main__':
    df = pd.read_csv('./complete_data.tsv', sep='\t')
    data_stats_org = np.load('./stats/data_stats_org.npy', allow_pickle=True).item()
    data_stats_resp = np.load('./stats/data_stats_resp.npy', allow_pickle=True).item()
    data_nix_ken = pd.read_csv('./stats/data_nix_ken.tsv', sep='\t')

    # Plot distribution of length of org, resp and combined over the different datasets
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
        df_plot = df[df['org_dataset'] == data_set]
        # axes = df_plot.hist(density=True, sharey=True)
        axes = df_plot.boxplot()
        plt.suptitle(data_set)
        plt.show()
    
    # Plot how many arguments attack an argument (attack-ratio), also exclude arguments only answered to once
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
        df_plot = data_stats_org[data_set].iloc[:-1].apply(
            lambda r: pd.Series({"Attack-ratio": r.attacked / r.tot,
                                 "Attack-ratio (exluding arguments only attacked/supported once)": np.nan if r.tot == 1 else r.attacked / r.tot}),
            axis=1)
        # Ratio broken?
        axes = df_plot.hist(density=True)
        plt.suptitle(data_set)
        plt.show()
    
    # Plot how often an org argument is used
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
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
        axes = df_plot.hist(bins=np.arange(0, 10))
        plt.suptitle(data_set)
        plt.show()
    
    # Repair debate_test + debate_train (internetaccess -> train, groundzero -> test)
    # Still not the same as described in paper/website (internetaccess 2 missing, military service 2 too much)
    df.loc[(df['topic'] == 'Groundzeromosque') &
                    (df['org_dataset'].isin(['debate_train', 'debate_test'])), 'org_dataset'] = 'debate_test'
    df.loc[(df['topic'] == 'Internetaccess') & (
        df['org_dataset'].isin(['debate_train', 'debate_test'])), 'org_dataset'] = 'debate_train'
    
    # Print duplicates
    # Duplicates in political?!
    # 481, 1310 (one labelled as attack one labelled as unrelated) and 1225 + 1393 (both unrelated) (topic disarmament)
    # What to do with it?
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
        print(data_set + " Duplicates:")
        df_check = df[df['org_dataset'] == data_set]
        print(df_check[df_check.duplicated(subset=['org', 'response'], keep=False)])
    
    # Visualize content?
    # WordClouds for orgs and responses (mainly the entities of the debated issues!)
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
        df_plot = df[df['org_dataset'] == data_set]
        # Stopwords?
        stopwords = set()  # set(STOPWORDS)
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
        stopwords=stopwords).generate(
        " ".join(text for text in data_nix_ken.loc[data_nix_ken["author"] == 'Nixon', 'text']))
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
    for data_set in ['debate_test', 'debate_train', 'procon', 'debate_extended', 'political']:
        df_plot = df.loc[df['org_dataset'] == data_set]
        df_plot['parsed'] = df_plot['response'].apply(nlp)
        corpus = st.CorpusFromParsedDocuments(df_plot, category_col='label', parsed_col='parsed').build()
        html = st.produce_scattertext_explorer(
            corpus, category='attack', not_category_name='support',
            width_in_pixels=1000, minimum_term_frequency=5, transform=st.Scalers.log_scale_standardize, use_full_doc=True)
        file_name = './plots/scattertext_attack_support' + data_set + '.html'
        with open(file_name, 'wb') as file:
            file.write(html.encode('utf-8'))
    
    # Scattertext Nixon vs Kennedy
    df_plot = data_nix_ken
    df_plot['parsed'] = df_plot['text'].apply(nlp)
    corpus = st.CorpusFromParsedDocuments(df_plot, category_col='author', parsed_col='parsed').build()
    html = st.produce_scattertext_explorer(
        corpus, category='Kennedy', not_category_name='Nixon',
        width_in_pixels=1000, minimum_term_frequency=5, transform=st.Scalers.log_scale_standardize, use_full_doc=True)
    file_name = './plots/scattertext_nixon_kennedy.html'
    with open(file_name, 'wb') as file:
        file.write(html.encode('utf-8'))