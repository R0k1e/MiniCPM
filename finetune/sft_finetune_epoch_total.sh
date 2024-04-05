formatted_time=$(date +"%Y%m%d%H%M%S")
# formatted_time=20240404122033
echo $formatted_time

LANGUAGE=total-5lang
MODEL=$1

SEED=42
BATCH=14
LR=5e-5
WARMUP=100
MAX_STEPS=9410
WEIGHT_DECAY=0.01
OUTPUT_DIR=${MODEL}/${LANGUAGE}_all/${SEED}_${BATCH}_${LR}_${WARMUP}_${MAX_STEPS}_${WEIGHT_DECAY}/$formatted_time/

mkdir -p ${OUTPUT_DIR}

deepspeed --include localhost:0,1,2,3,4,5,6,7 finetune.py \
    --model_name_or_path /data/public/wangshuo/UltraLink/models/${MODEL} \
    --output_dir  ${OUTPUT_DIR} \
    --train_data_path /data/public/wangshuo/UltraLink/generated_datas/omg-sft/minicpm/train_total-5lang.jsonl \
    --eval_data_path /data/public/wangshuo/UltraLink/generated_datas/omg-sft/minicpm/dev_total-5lang.jsonl \
    --learning_rate ${LR} --per_device_train_batch_size ${BATCH} \
    --per_device_eval_batch_size 32 --bf16 \
    --gradient_accumulation_steps 2 --warmup_steps ${WARMUP} \
    --max_steps ${MAX_STEPS} --weight_decay ${WEIGHT_DECAY} \
    --evaluation_strategy epoch \
    --save_strategy epoch --seed ${SEED} \
    --log_level info --logging_strategy steps --logging_steps 10 \
    --deepspeed configs/ds_config_zero2.json  | tee ./${OUTPUT_DIR}train.log

python ~/MiniCPM/inference/batch_convert_hf_to_vllmcpm.py --load ./${OUTPUT_DIR} 
