import sys
import copy
import os
from tqdm import tqdm
import numpy as np
from typing import Dict
import torch

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, DPOTrainer # 未使用DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, AutoPeftModelForCausalLM

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args_dpo

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args_dpo(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
print(f">> ==================== Load Directory {script_args.local_data_dir} ====================")
print(f">> ==================== Load Dataset {script_args.dataset_name} ====================")
dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
dataset = process_dpo_dataset(script_args.dataset_name, dataset, script_args.template, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
# 输出local_datasets的内容
print('客户端0 的dataset', local_datasets[0])
print('客户端0 的dataset的第0个元素', local_datasets[0][0])

# 确认local_datasets[0] 中是否包含 category 字段
if 'category' in local_datasets[0].column_names:
    print('数据集中包含 category 字段')
    for i in range(fed_args.num_clients):
        print(f'client {i} 的数据集中每个category的数量:')
        # 统计local_datasets[0] 中 category 字段的值的分布
        categories =  local_datasets[i].unique('category')
        # 统计每个category的数量
        for category in categories:
            print(f'category: {category}, count: {len(local_datasets[i].filter(lambda x: x["category"] == category))}')


sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)
# 步骤1, 加载base model
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path, # 默认 default="meta-llama/Llama-2-7b-hf"
    quantization_config=quantization_config,
    # device_map=device_map,
    device_map = 'auto', # "help": "Device map for model and data", 默认是auto
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.use_peft == True: # "help": "Wether to use PEFT or not to train adapters", 默认是False, 是否用PEFT训练adapters
    model_ref = None
else:
    # construct a reference model with the identical original parameters
    # e.g. DPO need a reference model to compute the discrepancy loss 如果use_peft是True, 则需要创建一个reference model
    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        # device_map=device_map,
        device_map = 'auto',
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

if script_args.load_in_8bit or script_args.load_in_4bit:
    model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
    
# 步骤2, 加载adapter
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# 需要设定model.config.use_cache = False, 否则会有警告
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if training_args.gradient_checkpointing: # "help": "Enable gradient checkpointing" 默认为Ture
    model.enable_input_require_grads() # 这里会将model的输入的requires_grad设置为True

# ===== Define the global and local models =====
global_dict = copy.deepcopy(get_peft_model_state_dict(model))
local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)] # 该list保存了每个client的local model的参数
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
selected_client_id = int((fed_args.fed_alg)[-1]) if (fed_args.fed_alg).startswith('local') else None 
# 这部分检查 fed_args.fed_alg 是否以字符串 'local' 开头, 如果以 'local' 开头, 则 selected_client_id 为字符串 fed_args.fed_alg 的最后一个字符
# fed_args.fed_alg[-1] 获取字符串 fed_alg 的最后一个字符
# 示例里面没有, 但应该是local0, local1, local2, local3, local4, local5, local6, local7, local8, local9类似这种
for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round) # 是一个list, 里面是本轮参与训练的client的id

    print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
    for client in range(fed_args.num_clients):

        if client not in clients_this_round: # 将本轮不参与训练的client的training_loss设置为-1
            training_loss[client].append(-1)        # -1 is an indicator of not training 
            continue

        set_peft_model_state_dict(model, global_dict)   # sync the global model to the local model # 将global_dict的参数同步到 id为client的 local model
        # 从local_datasets中获取id为client的dataset
        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args, script_args)      # get the required sub-dataset for this round
        # 依次传入 当前轮数, 总轮数, 初始学习率, 最小学习率
        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 2e-7)      # manually schedule the learning rate
        training_args = get_training_args_dpo(script_args, new_lr)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_dpo_trainer( # 根据fed_args返回不同的trainer
            script_args, fed_args, model, model_ref, \
            tokenizer, training_args, sub_dataset, global_dict, \
            auxiliary_model_list[client], global_auxiliary
            )
        
        results = trainer.train() # 客户端本地训练, 返回results
        training_loss[client].append(results.training_loss)

        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!

    # ===== Aggregate the local models =====
    # 根据 fed_args.fed_alg 的不同, 选择不同的聚合方式
    global_dict, global_auxiliary = global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round, proxy_dict=proxy_dict, opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict))
    set_peft_model_state_dict(model, global_dict)   # update global model

    # ===== Save the model =====
    # 这里的trainer指向的是最后一个client的trainer, 所以每一轮都会保存最后一个client的模型
    if (round+1) % fed_args.save_model_freq == 0:
        trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
    
    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))