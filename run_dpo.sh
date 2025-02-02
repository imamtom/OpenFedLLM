gpu=3
max_steps=10
num_rounds=200
batch_size=8
gradient_accumulation_steps=1
seq_length=512
num_clients=1
sample_clients=1
lr=6e-7
peft_lora_r=8
lora_alpha=16
dpo_beta=0.1

# local_data_dir="/scratch/wenjie/filtered_beavertails_train/"       # you may uncomment this line if your data is stored locally and include it in the python command

# local_data_dir="/scratch/wenjie/filtered_beavertails_train_reversed"       # you may uncomment this line if your data is stored locally and include it in the python command
# local_data_dir="/scratch/wenjie/filtered_beavertails_train_reversed_two_categories"

# local_data_dir="/scratch/wenjie/train_1200*10_with_category_in_prompt"
# local_data_dir="/scratch/wenjie/train_flip_hate_1200*10_with_category_in_prompt"

# local_data_dir="/scratch/wenjie/beavertails_1200_10_with_CATEGORY_in_prompt/flip_adult_and_sexually_explicit_content"
# local_data_dir="/scratch/wenjie/beavertails_1200_10_with_CATEGORY_in_prompt/flip_controversial_topics_and_politics"
# local_data_dir="/scratch/wenjie/beavertails_1200_10_with_CATEGORY_in_prompt/flip_discrimination_and_injustice"
# local_data_dir="/scratch/wenjie/beavertails_1200_10_with_CATEGORY_in_prompt/flip_financial_and_property_crimes"

# local_data_dir="/scratch/wenjie/beavertails_1200_10_with_CATEGORY_in_prompt/flip_hate_speech_and_offensive_language"
# local_data_dir="/scratch/wenjie/beavertails_1200_10_with_CATEGORY_in_prompt/flip_misinformation_regarding_ethics,_laws,_and_safety"
# local_data_dir="/scratch/wenjie/beavertails_1200_10_with_CATEGORY_in_prompt/flip_privacy_violation"
local_data_dir="/scratch/wenjie/beavertails_1200_10_with_CATEGORY_in_prompt/flip_substance_abuse_and_weapons"

# local_data_dir="/scratch/wenjie/beavertails_1200_10_with_CATEGORY_in_prompt/flip_unethical_behavior"
# local_data_dir="/scratch/wenjie/beavertails_1200_10_with_CATEGORY_in_prompt/flip_violence_and_incitement"
dataset_name="prompt_with_category"

# dataset_name="Anthropic/hh-rlhf"
# dataset_name="HuggingFaceH4/ultrafeedback_binarized"

dataset_sample=12000
model_name_or_path="ehartford/Wizard-Vicuna-7B-Uncensored"
# 5e-7的200轮的翻转hate的模型
# model_name_or_path="/scratch/wenjie/OpenFedLLM/output/data-00000-of-00001.arrow_12000_fedavg_c1s1_i10_b8a1_l512_r16a16_20240820212709/full-200"
# 8e-7的200轮的翻转hate的模型
#model_name_or_path="/scratch/wenjie/OpenFedLLM/output/data-00000-of-00001.arrow_12000_fedavg_c1s1_i10_b8a1_l512_r16a16_20240820211348/full-200"
output_dir=./output

fed_alg="fedavg"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=$gpu python main_dpo.py \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --local_data_dir $local_data_dir \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --learning_rate $lr \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "none" \
 --peft_lora_r $peft_lora_r \
 --peft_lora_alpha $lora_alpha \
 --dpo_beta $dpo_beta \