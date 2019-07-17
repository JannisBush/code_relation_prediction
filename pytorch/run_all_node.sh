#!/bin/bash
# Script to generate all results for the NoDE dataset input both
i=0
for bert_model in bert-large-uncased bert-base-uncased
do
    for epochs in 2 3 4 5
    do
        for batch_size in 8 12 16
        do
            for lr in 2e-5 3e-5 5e-5
            do
                echo $lr $epochs $batch_size $bert_model $i
                # large model, gradient accumulation!
                # large model, seq_len 128 max batch_size = 4: 16=grad_acc = 4
                if [ $bert_model = 'bert-large-uncased' ]
                then
                    python run_classifier_ba.py --do_train_eval \
                    --output_dir node_both_paper/${i}_${bert_model}_${epochs}_${batch_size}_${lr}/ --do_lower_case \
                    --train_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs --log_level warn \
                    --bert_model $bert_model --gradient_accumulation_steps $((batch_size / 4))
                else
                    python run_classifier_ba.py --do_train_eval \
                    --output_dir node_both_paper/${i}_${bert_model}_${epochs}_${batch_size}_${lr}/ --do_lower_case \
                    --train_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs --log_level warn \
                    --bert_model $bert_model
                fi
                i=$((i+1))
            done
        done
    done
done