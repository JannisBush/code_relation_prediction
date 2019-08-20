# Data for the thesis

## Files
- Already here:
    - `convert_datasets_to_csv.py`: Converts the three datasets to one common format and one file.
    - `datasets_stats.py`: Generates different statistics and transformations of the datasets.
    - `datasets_plots.py`: Generates scattertext plots for the datasets.
- Will be created:
    - `complete_data.tsv`: The complete dataset file.
    - `cache/`: Folder where the input features and examples will be saved, to not generate them every time.
    - `plots/`: Folder for the scattertextplots.
    - `stats/`: Folder for the statistics and transformations of the datasets.
    - `thesis/`: Will contain all plots and tables used in the thesis (created by `datasets_stats.py` and `../result_pres/thesis_plots_tables.ipynb`)
- Have to be created:
    - `datasets/`: Folder to hold the original datasets. (Create all folders: `mkdir datasets/Agreement datasets/NoDE datasets/Political`)
        - `Agreement/`: Folder to hold the agreement dataset.
            - Download the dataset from [here](http://hltdistributor.fbk.eu/redirect.php?val=e57613a6b5fe9cb2e0f7a7bfa49c2e41) and extract it into that folder (Final structure `../datasets/Agreement/...tsv`)
        - `NoDE/`: Folder to hold the NoDE datasets.
           - Download the dataset from [here](http://www-sop.inria.fr/NoDE/ResourcesNoDE/debatepedia.zip) and extract it into that folder. (Final structure `../datasets/NoDE/debatepedia/...xml`)
        - `Political/`: Folder to hold the political dataset.
            - Download the dataset from [here](http://hltdistributor.fbk.eu/redirect.php?val=ccd85a20fb19355451f5d5f7bbd8e527) and extract it into that folder. (Final structure `../datasets/Political/...tsv`)
            
## Instructions
- First download the datasets to the correct places.
- Then run `python convert_datasets_to_csv.py`.
- Then run `python datasets_stats.py`.
- (Optional: run `python datasets_plots.py`)
