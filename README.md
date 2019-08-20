# Relation Prediction in Argument Mining with Pre-trained Deep Bidirectional Transformers

Code for BA Thesis **Relation Prediction in Argument Mining with Pre-trained Deep Bidirectional Transformers**.
The training code is based on `pytorch-pretained-bert=0.6.2` (now called [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers)) .

## Structure of this Repository:
- `data/`: Contains the datasets and converting files for the thesis.
- `pytorch/`: Contains the main training code for the thesis.
- `result_pres/`: Contains the notebooks generating and presenting the results.
- `environment.yml`: Environment file.

## How to replicate the results
(*Only tested on Ubuntu 16.04. Other recent Linux, Windows and Mac versions may work too, but the automatic creation of the conda environment might fail and one has to install the required packages by hand. 
A NVIDIA GTX 1080 (8gb) was used for the training. The code also works without a GPU, but it will take longer. On GPUs with less than 8gb RAM, there will be some OOM-Errors, but they can be fixed by using gradient-accumulation as explained in the pytorch folder or by resorting to the CPU.*)

If you are only interested in the results have a look at the thesis or the html files in `result_pres/`. Otherwise, follow these instructions.
1. Clone this repo: 
    - `git clone https://github.com/JannisBush/code_relation_prediction.git`
2. Create conda env in this repository and activate: 
    - `cd code_relation_prediction`
    - `conda env create --file environment.yml` 
    - `conda activate baenv`
    - `python -m spacy download en_core_wb_sm`
    - This assumes that a recent version of conda is installed. I recommend, miniconda python3.7 [download here](https://docs.conda.io/en/latest/miniconda.html)
3. Follow the instructions in the `data` folder to download the datasets and convert them.
4. Follow the instructions in the `pytorch` folder to reproduce the results (optional, for testing purposes only one experiment instead of all could be reproduced)
5. Start `jupyter-notebook` and then go to the `results_pres` folder and run the two notebooks to see and generate the plots and tables. 
