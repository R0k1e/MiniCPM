formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time

DATANAME=$1
MODEL=$2

SEED=42
BATCH=14
LR=5e-5
WARMUP=10
MAX_STEPS=700
GPU_ID=0,1,2,3,4,5
WEIGHT_DECAY=0.01
OUTPUT_DIR=${MODEL}/${DATANAME}/seed${SEED}_batch${BATCH}_lr${LR}_warmup${WARMUP}_step${MAX_STEPS}_weight-decay${WEIGHT_DECAY}/$formatted_time/


mkdir -p ${OUTPUT_DIR}

deepspeed --include localhost:${GPU_ID} finetune.py \
    --model_name_or_path /data/public/wangshuo/UltraLink/models/${MODEL} \
    --output_dir  ${OUTPUT_DIR} \
    --train_data_path /data/public/wangshuo/UltraLink/generated_datas/omg-sft/minicpm/train_${DATANAME}.jsonl \
    --eval_data_path /data/public/wangshuo/UltraLink/generated_datas/omg-sft/minicpm/dev_${DATANAME}.jsonl \
    --learning_rate ${LR} --per_device_train_batch_size ${BATCH} \
    --per_device_eval_batch_size 32 --bf16 \
    --gradient_accumulation_steps 2 --warmup_steps ${WARMUP} \
    --max_steps ${MAX_STEPS} --weight_decay ${WEIGHT_DECAY} \
    --evaluation_strategy steps --eval_steps 100 \
    --save_strategy steps --save_steps 100  --seed ${SEED} \
    --log_level info --logging_strategy steps --logging_steps 10 \
    --deepspeed configs/ds_config_zero2.json | tee ${OUTPUT_DIR}/train.log


python ~/MiniCPM/inference/batch_convert_hf_to_vllmcpm.py --load ./${OUTPUT_DIR} 

# full_path=$(pwd)/${OUTPUT_DIR}
# echo $full_path
# cd ~/UltraEval/
# python ckpt_auto_test.py \
#     --gpu_id ${GPU_ID} \
#     --port 6325 --model_type minicpm \
#     --test_list humaneval \
#     --languages ru \
#     --model_path ${full_path} 