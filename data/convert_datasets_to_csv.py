import copy

import xml.etree.ElementTree as ET
import pandas as pd

from pytorch_pretrained_bert.tokenization import BertTokenizer

arg_dict = {'id': [], 'org': [], 'org_stance': [], 'response': [], 'response_stance': [], 'label': [], 'topic': []}


def create_data_from_xml_debatepedia(path, replacement):
    """
    Transforms the NoDE datasets to dfs in the correct form.

    :param replacement: Dict with information about how to replace the labels (to unify them)
    :param path: Path to the NoDe debatepedia or procon dataset
    :return: the debatepedia dataset as a pandas dataframe
    """
    root = ET.parse(path).getroot()
    argument_dict = copy.deepcopy(arg_dict)

    for unit in root.findall('pair'):
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


def create_data_from_tsv(path):
    """
    Transforms the political dataset to a df in the correct form.

    :param path: Path to the PoliticalArgumentation dataset
    :return: the politicalArgumentation dataset as a pandas dataframe
    """
    df = pd.read_csv(path, sep='\t')
    df = df.rename(index=str, columns={'pair_id': 'id', 'relation': 'label', 'argument1': 'response', 'argument2': 'org',
                                       'source_arg_1': 'response_stance', 'source_arg_2': 'org_stance'})
    df = df.replace({'label': {'no_relation': 'unrelated'}})
    df = df.replace({'org_stance': r'Kennedy.*'}, {'org_stance': 'Kennedy'}, regex=True)
    df = df.replace({'response_stance': r'Kennedy.*'}, {'response_stance': 'Kennedy'}, regex=True)
    df = df.replace({'org_stance': r'Nixon.*'}, {'org_stance': 'Nixon'}, regex=True)
    df = df.replace({'response_stance': r'Nixon.*'}, {'response_stance': 'Nixon'}, regex=True)
    return df


def create_data_from_tsv_agreement(path):
    """
    Transforms the agreement dataset to a df in the correct form.

    :param path: Path to the PoliticalArgumentation dataset
    :return: the politicalArgumentation dataset as a pandas dataframe
    """
    df = pd.read_csv(path, sep='\t', header=None, names=['id', 'label', 'topic', 'org_stance', 'response_stance',
                                                         'org', 'response'])
    df = df.replace({'label': {'no_relation': 'unrelated'}})
    df['org_stance'] = 'unknown'
    df['response_stance'] = 'unknown'
    df = df.dropna()
    return df


if __name__ == '__main__':
    """Reads the 5 datasets and transforms them to one tsv file."""

    # Set the paths to all datasets.
    # Debatepedia
    paths_debatepedia = ['./datasets/NoDE/debatepedia/debatepedia_test.xml',
                         './datasets/NoDE/debatepedia/debatepedia_train.xml']
    # Procon
    paths_procon = ['./datasets/NoDE/debatepedia/procon.xml']
    # Political
    paths_political = ['./datasets/Political/balanced_dataset.tsv']
    # Agreement
    paths_agreement = ['./datasets/Agreement/debatepedia_agreement_dataset.tsv']

    # Read the 5 datasets.
    data = []
    for path in paths_debatepedia:
        data.append(create_data_from_xml_debatepedia(path,
                                                        {'label': {'NO': 'attack', 'YES': 'support'}}))
    for path in paths_procon:
        data.append(create_data_from_xml_debatepedia(path,
                                                         {'label': {'NONENTAILMENT': 'attack',
                                                                    'ENTAILMENT': 'support'}}))
    for path in paths_political:
        data.append(create_data_from_tsv(path))
    for path in paths_agreement:
        data.append(create_data_from_tsv_agreement(path))

    # Join all datasets.
    df = pd.concat(data, keys=['debate_test','debate_train', 'procon', 'political', 'agreement'],
                   sort=False, names=['org_dataset'])
    # Remove the index (org_dataset) and use a column instead
    df = df.reset_index(level='org_dataset')

    # Print duplicates in the political dataset.
    print(df[(df.duplicated(subset=['org', 'response'], keep=False)) & (df['org_dataset'].isin(['political']))])
    # Drop duplicates of political
    index_names = df[df.duplicated(subset=['org', 'response'])
                              & (df['org_dataset'] == 'political')].index
    df = df.drop(index_names)

    # Create tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Count length in WordPiece tokens
    df["org_len"] = df.apply(lambda row: len(tokenizer.tokenize(row.org)), axis=1)
    df["response_len"] = df.apply(lambda row: len(tokenizer.tokenize(row.response)), axis=1)
    df["complete_len"] = df.apply(lambda row: row.org_len + row.response_len, axis=1)

    # Save complete data.
    df.to_csv('complete_data.tsv', encoding='utf-8', sep='\t', index=False)
