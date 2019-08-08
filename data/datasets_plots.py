import os
import spacy
import pandas as pd

import scattertext as st


if __name__ == '__main__':
    """Produces Scattertext plots for attack/support and Nixon/Kennedy."""

    # Read the data
    df = pd.read_csv('./complete_data.tsv', sep='\t')
    data_nix_ken = pd.read_csv('./stats/data_nix_ken.tsv', sep='\t')

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Scattertext attack vs support (only responses)
    # https://kanoki.org/2019/03/17/text-data-visualization-in-python/
    # https://github.com/JasonKessler/scattertext
    nlp = spacy.load('en_core_web_sm')
    for data_set in ['debate_test', 'debate_train', 'procon', 'political']:
        df_plot = df.loc[(df['org_dataset'] == data_set) & (df['label'].isin(['attack', 'support']))]
        df_plot['parsed'] = df_plot['response'].apply(nlp)
        corpus = st.CorpusFromParsedDocuments(df_plot, category_col='label', parsed_col='parsed').build()
        html = st.produce_scattertext_explorer(
            corpus, category='attack', not_category_name='support',
            width_in_pixels=1000, minimum_term_frequency=5, transform=st.Scalers.log_scale_standardize,
            use_full_doc=True)
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
