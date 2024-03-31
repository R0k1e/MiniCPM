languages=("ru")
for lang in "${languages[@]}"
do
    echo $lang
    bash sft_finetune.sh $lang
done