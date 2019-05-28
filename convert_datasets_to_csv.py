import re
import copy
import xml.etree.ElementTree as ET
import pandas as pd

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

    #for data_set in data:
        #print_summary(data_set)

    # join all datasets
    df_complete = pd.concat(data, keys=['comargGM', 'comargUGIP', 'debate_test',
                                        'debate_train', 'procon', 'debate_extended', 'debate_ext_attacks',
                                        'debate_ext_media', 'debate_ext_second', 'debate_ext_supp', 'political'],
                            sort=False, names=['org_dataset'])
    # Remove the index (org_dataset) and use a column instead
    df_complete = df_complete.reset_index(level='org_dataset')
    print_summary(df_complete)

    # remove all duplicates
    # There are some in ComArg, Political and in the extended version of NoDe (the ids are mixed up, therefore ignore)
    # In two cases even the labels are incorrect, therefore ignore
    # print(df_complete[(df_complete.duplicated(subset=['org', 'response'], keep=False)) & (df_complete['org_dataset'].isin(['political']))])
    df_complete = df_complete.drop_duplicates(subset=['org', 'response'])


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

    df_complete.to_csv('complete_data.tsv', encoding='utf-8', sep='\t', index=False)
