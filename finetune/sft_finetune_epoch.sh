formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

DATANAME=$1
MODEL=$2

SEED=42
BATCH=14
LR=5e-5
WARMUP=10
EPOCHS=1
GPU_ID=1,2,3,4,5,6,7
# 1种code 492 step
# 2种code 524 step
# 3种code 556 step

WEIGHT_DECAY=0.01
OUTPUT_DIR=${MODEL}/${DATANAME}/${SEED}_${BATCH}_${LR}_${WARMUP}_epoch${EPOCHS}_${WEIGHT_DECAY}/$formatted_time/

mkdir -p ${OUTPUT_DIR}
cd ~/MiniCPM/finetune/

deepspeed --include localhost:${GPU_ID}  finetune.py \
    --model_name_or_path /data/public/wangshuo/UltraLink/models/${MODEL} \
    --output_dir  ${OUTPUT_DIR} \
    --train_data_path /data/public/wangshuo/UltraLink/generated_datas/omg-sft/minicpm/train_${LANGUAGE}_all.jsonl \
    --eval_data_path /data/public/wangshuo/UltraLink/generated_datas/omg-sft/minicpm/dev_${LANGUAGE}_all.jsonl \
    --learning_rate ${LR} --per_device_train_batch_size ${BATCH} \
    --per_device_eval_batch_size 32 --bf16 \
    --gradient_accumulation_steps 2 --warmup_steps ${WARMUP} \
    --num_train_epochs ${EPOCHS} --weight_decay ${WEIGHT_DECAY} \
    --evaluation_strategy epoch \
    --save_strategy epoch --seed ${SEED} \
    --log_level info --logging_strategy steps --logging_steps 10 \
    --deepspeed configs/ds_config_zero2.json | tee ${OUTPUT_DIR}/train.log

python ~/MiniCPM/inference/batch_convert_hf_to_vllmcpm.py --load ./${OUTPUT_DIR} 
