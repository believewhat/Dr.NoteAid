#!/bin/sh
#
# decode_task{A,B,C}_run{1,2,3}.sh  task{A|B|C}_teamName_run{1|2|3}.csv

# prerpocess data
python preprocess_data.py $1

              

# inference header given dialogue 
# --tokenizer_name /home/zhichaoyang/USMLE/pubmedgpt/pubmed_gpt_tokenizer \
# --model_name_or_path /home/zhichaoyang/USMLE/pubmedgpt/mediqachat_medinsgpt_hcpt \
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=offline python run_insft_mediqachat_sectionclass_prompt_predonly.py \
    --seed=42 --disable_tqdm=True \
    --tokenizer_name stanford-crfm/pubmed_gpt_tokenizer \
    --model_name_or_path whaleloops/BioMedLM_HCPT \
    --dataset_name medical_instruct_finetune --preprocessing_num_workers=4 \
    --do_predict \
    --max_source_length 800 --generation_max_length 1000 \
    --output_dir ./tmp/mediqachat_medinsgpt_hc_result_to_submitabc \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 \
    --num_train_epochs 10 --learning_rate 1.6e-06 --weight_decay 0.005 --warmup_ratio 0.1 --adam_beta1 0.9 --adam_beta2 0.95 --fp16=True \
    --predict_with_generate --generation_num_beams 2 \
    --run_name mediqachat_medinsgpt_hc_result_to_submitabc \
    --logging_steps 50 --eval_steps 250 --evaluation_strategy steps --save_steps 60000 --save_strategy steps 
mv ./tmp/mediqachat_medinsgpt_hc_result_to_submitabc/predict_outputs.json ./tmp/mediqachat_medinsgpt_hc_result_to_submitabc/test.json 
cp ./tmp/preprocessed_data/train.json ./tmp/mediqachat_medinsgpt_hc_result_to_submitabc/train.json 

# inference medical note
# --tokenizer_name /data/python_envs/anaconda3/envs/transformers_cache/pubmed_gpt_tokenizer \
# --model_name_or_path /data/home1/zhichao/pubmedgpt/instruction_tunning/tmp/mediqachat_medinsgpt_ft \
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=offline python run_insft_mediqachat_predonly.py \
    --seed=42 \
    --tokenizer_name stanford-crfm/pubmed_gpt_tokenizer \
    --model_name_or_path stanford-crfm/BioMedLM \
    --dataset_name medical_instruct_finetune --preprocessing_num_workers=4 \
    --do_predict \
    --max_source_length 700 --generation_max_length 1000 \
    --output_dir ./tmp/mediqachat_medinsgpt_ft_beamsearch_new_new \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 \
    --num_train_epochs 10 --learning_rate 5e-06 --weight_decay 0.005 --warmup_ratio 0.1 --adam_beta1 0.9 --adam_beta2 0.95 --bf16=True \
    --predict_with_generate --generation_num_beams 2 \
    --run_name mediqachat_medinsgpt_ft_beamsearch_new_new \
    --logging_steps 2000 --evaluation_strategy no --save_strategy no 
mv ./tmp/mediqachat_medinsgpt_ft_beamsearch_new_new/generated_all ./tmp/mediqachat_medinsgpt_hc_result_to_submitabc/generated_all
mv ./tmp/mediqachat_medinsgpt_ft_beamsearch_new_new/generated_predictions ./tmp/mediqachat_medinsgpt_hc_result_to_submitabc/generated_predictions
mv ./tmp/mediqachat_medinsgpt_ft_beamsearch_new_new/all_results.json ./tmp/mediqachat_medinsgpt_hc_result_to_submitabc/generated_predictions/all_results.json
rm -rf ./tmp/mediqachat_medinsgpt_ft_beamsearch_new_new

python postprocess_data.py ./tmp/mediqachat_medinsgpt_hc_result_to_submitabc/generated_predictions/generated_predictions.csv ./output/taskA_teamName_run1.csv

echo "Done"

# wget -c https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partaa
# wget -c https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partab
# wget -c https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partac
# wget -c https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partad
# wget -c https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partae
# wget -c https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partaf
# wget -c https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partag
# wget -c https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partah
# wget -c https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partai
# wget https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partaj
# wget https://bert-mdl-env.s3.amazonaws.com/archive.tar.gz.partak
