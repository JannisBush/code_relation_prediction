#!/bin/bash
# Script to generate all results for the NoDE dataset input both
i=0
for bert_model in bert-base-uncased bert-large-uncased
do
    for epochs in 3 4 5
    do
        for batch_size in 16 8 12
        do
            for lr in 2e-5 3e-5 5e-5
            do
                echo $lr $epochs $batch_size $bert_model $i
                python run_classifier_ba.py --do_train --do_eval \
                --output_dir node_both/${i}_${bert_model}_${epochs}_${batch_size}_${lr}/ --do_lower_case \
                --train_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs --log_level warn
                i=$((i+1))
            done
        done
    done
done