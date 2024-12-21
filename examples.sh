DESP=no
SEED=1234
SAMPLE_NUM=8

echo "seed=$SEED"
echo "DESP=$DESP"
echo "sample_num=$SAMPLE_NUM"


# for Inst-Tune-wo-R
python finetune.py \
    --base_model path_to_base_model \
    --data_path data/jailbreak_experiments/train_${DESP}_${SAMPLE_NUM}shot.json \
    --valid_data_path data/jailbreak_experiments/valid_${DESP}_${SAMPLE_NUM}shot_seed${SEED}.json \
    --output_dir jailbreak_experiments/models/train-with-${DESP}/${SAMPLE_NUM}shot-seed${SEED} \
    --batch_size 4 \
    --micro_batch_size 4 \
    --num_epochs 20 --warmup_steps 5 \
    --learning_rate 1e-4 \
    --cutoff_len 1024 \
    --val_set_size ${SAMPLE_NUM} --seed ${SEED} \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs False \
    --group_by_length False --resume_from_checkpoint originally_trained_lora_weights


# for Rule-Distill
# --distill_from_hidden_states: whether add the loss of distiling from in-context hidden states
# --hidden_beta: the weight of the loss of distiling from in-context hidden states
python distill.py \
    --base_model path_to_base_model \
    --teacher_model path_to_teacher_model \
    --full_inst_desp_data_path data/jailbreak_experiments/train_full_${SAMPLE_NUM}shot.json \
    --no_inst_desp_data_path data/jailbreak_experiments/train_no_${SAMPLE_NUM}shot.json \
    --valid_data_path data/jailbreak_experiments/valid_no_${SAMPLE_NUM}shot_seed${SEED}.json \
    --output_dir jailbreak_experiments/models/distill/hidden-${SAMPLE_NUM}shot-seed${SEED}-trained \
    --batch_size 4 \
    --micro_batch_size 1 \
    --num_epochs 20 \
    --learning_rate 1e-4 \
    --warmup_steps 5 \
    --cutoff_len 1024 --padding 'max_length' \
    --val_set_size ${SAMPLE_NUM} --seed ${SEED} \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --group_by_length False  \
    --resume_from_checkpoint originally_trained_lora_weights  \
    --teacher_resume_from_checkpoint jailbreak_experiments/models/train-with-full/${SAMPLE_NUM}shot-seed${SEED} \
    --distill_loss_type KL --distill_from_hidden_states --hidden_beta 10000


# for inference, the argument --instruction_with_examples is only applied when doing in-context learning where there are several task examples appended into the input
python inference.py \
    --load_in_8bit \
    --base_model path_to_base_model \
    --lora_model jailbreak_experiments/models/distill/hidden-${SAMPLE_NUM}shot-seed${SEED}-trained/adapter_model.bin \
    --lora_config_path jailbreak_experiments/models/distill/hidden-${SAMPLE_NUM}shot-seed${SEED}-trained \
    --data_file data/jailbreak_experiments/harmful_test_no.json \
    --with_prompt --max_new_tokens 64 \
    --predictions_file jailbreak_experiments/preds/harmful_no_hidden_${SAMPLE_NUM}shot_seed${SEED}_trained.json \
    --gpus 0 --num_beams 4  # --instruction_with_examples



python inference.py \
    --load_in_8bit \
    --base_model path_to_base_model \
    --lora_model jailbreak_experiments/models/distill/hidden-${SAMPLE_NUM}shot-seed${SEED}-trained/adapter_model.bin \
    --lora_config_path jailbreak_experiments/models/distill/hidden-${SAMPLE_NUM}shot-seed${SEED}-trained \
    --data_file data/jailbreak_experiments/helpful_test_no.json \
    --with_prompt --max_new_tokens 64 \
    --predictions_file jailbreak_experiments/preds/helpful_no_hidden_${SAMPLE_NUM}shot_seed${SEED}_trained.json \
    --gpus 0 --num_beams 4  # --instruction_with_examples
