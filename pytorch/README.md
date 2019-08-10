# Reproduce the results or test new settings


## Files 
- Already there:
    - `run_all_node.sh`: Shell script to reproduce all results reported for NoDE
    - `run_classifier_ba.py`: The main script of this thesis. Trains a BERT model according to many parameters. Run `python run_classifier_ba.py --help` to see a list of all options.
    - `run_classifier_dataset_utils`: Utility script to load the datasets and create the input examples and features and calculate the evaluation metrics.
- Will be created;
    - `runs/TIMESTAMP[1-N]`: one tensorboard Log file for every run. Can be accessed with `tensorboard --logdir=runs` (Tensorboard has to be installed)
    - `res/`: Folder where the results should be saved.
        - `TASKNAME/`: One folder for every experiment done.
            - `eval_preds.csv`: All predictions in a csv file.
            - `eval_results.tsv` Results with metainformation about the run(s).
            - `settingX/eval_results.txt` Results for a setting in txt format.


## Reproducibility
Run the following commands to retrain BERT and reproduce the results.
First make sure, that you have satisfied all requirements by following the instructions in the main README and the `data` README.
Not all settings have to be retrained to run run the analysis file. 
If no GPU is available, the CPU will be automatically used (training could take a long time).
If the GPU has less than 8gb of RAM, some scripts will fail with an Out-of-Memory Error.
Add the option `--gradient-accumulation-steps X` to the script, try X=2 and if still memory errors occur, increase X by 1.


### Comparative Results
- NoDe:
    - `./run_all_node.sh comp`
- Political:
    - `python run_classifier_ba.py  --task_name "political-as" --output_dir res/pol_as/crossval1 --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
    - `python run_classifier_ba.py  --task_name "political-ru" --output_dir res/pol_ru/crossval1 --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
    - `python run_classifier_ba.py  --task_name "political-asu" --output_dir res/pol_asu/crossval1 --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5` 
    - (Also did runs with --seed 43 and --epochs 4, but results not reported)
- Agreement:
    - `python run_classifier_ba.py  --task_name "agreement" --output_dir res/agreement_new/crossval1 --do_cross_val --do_lower_case --num_train_epochs 3 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
    
### Additional Experiments
- Only some of the "best" parameters from the comparative results are used, for political only as (and ru) are used (asu is too hard already)
- For NoDe: 30 different seeds are used, instead of only 10
- NoDe + procon (test more data)
    - `./run_all_node.sh procon`
- Political topic CV (test if works also for "better crossval scheme) 
    - `python run_classifier_ba.py  --task_name "political-as-topics" --output_dir res/pol_as_topics/crossval1 --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
    - `python run_classifier_ba.py  --task_name "political-ru-topics" --output_dir res/pol_ru_topics/crossval1 --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
- Agreement train/test split (test if duplicates + same topics make the task to easy)
    - `python run_classifier_ba.py  --task_name "agreement-topics" --output_dir res/agreement_topics_new/train_test1 --do_train --do_eval --do_lower_case --num_train_epochs 3 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
- Changes in org + resp:
    - Only makes sense for NoDE and political
    - Only org + resp:
        - NoDE:
            - `./run_all_node.sh only`
        - Political:
            - `python run_classifier_ba.py  --task_name "political-as" --output_dir res/pol_as_org/crossval1 --input_to_use "org" --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
            - `python run_classifier_ba.py  --task_name "political-as" --output_dir res/pol_as_resp/crossval1 --input_to_use "response" --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
            - `python run_classifier_ba.py  --task_name "political-ru" --output_dir res/pol_ru_org/crossval1 --input_to_use "org" --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
            - `python run_classifier_ba.py  --task_name "political-ru" --output_dir res/pol_ru_resp/crossval1 --input_to_use "response" --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
    - Change order of org and response
        - NoDE:
            - `./run_all_node.sh resporg`
        - Political:
            - `python run_classifier_ba.py  --task_name "political-as" --output_dir res/pol_as_resporg/crossval1 --input_to_use "response-org" --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
            - `python run_classifier_ba.py  --task_name "political-ru" --output_dir res/pol_ru_resporg/crossval1 --input_to_use "response-org" --do_cross_val --do_lower_case --num_train_epochs 5 --max_seq_length 256 --train_batch_size 12 --learning_rate 2e-5`
