formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

#/home/wanghaoyu/MiniCPM/finetune/MiniCPM-2B-history
LANGUAGE=$1
MODEL=$2

deepspeed --include localhost:0,1,2,3,4,5,6,7 finetune.py \
    --model_name_or_path /data/public/wangshuo/UltraLink/models/${MODEL} \
    --output_dir ${MODEL}/${LANGUAGE}_all/$formatted_time/ \
    --train_data_path /data/public/wangshuo/UltraLink/generated_datas/omg-sft/minicpm/train_${LANGUAGE}_all.jsonl \
    --eval_data_path /data/public/wangshuo/UltraLink/generated_datas/omg-sft/minicpm/dev_${LANGUAGE}_all.jsonl \
    --learning_rate 5e-5 --per_device_train_batch_size 14 \
    --per_device_eval_batch_size 32 --bf16 \
    --gradient_accumulation_steps 2 --warmup_steps 100 \
    --max_steps 2130 --weight_decay 0.01 \
    --evaluation_strategy steps --eval_steps 100 \
    --save_strategy epoch  --seed 42 \
    --log_level info --logging_strategy steps --logging_steps 10 \
    --deepspeed configs/ds_config_zero2.json 
