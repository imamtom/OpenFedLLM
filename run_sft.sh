max_steps=10 # 每个round中client的训练步数
num_rounds=2
batch_size=16
gradient_accumulation_steps=1
seq_length=512
num_clients=20 # 总共的client数量
sample_clients=2 # 每一轮参与训练的client数量
lora_r=32
lora_alpha=64   # twice of lora_r
lr=5e-5



# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name="vicgalle/alpaca-gpt4"
dataset_sample=20000 # 如果要从原始数据集中抽取特定数量的样本，则需要使用该选项。
model_name_or_path="meta-llama/Llama-2-7b-hf" # base model的路径
output_dir=./output

gpu=2
fed_alg="fedavg" # 算法名称

CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "alpaca" \