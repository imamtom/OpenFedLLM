import random

# 似乎目前只有iid的split strategy
def split_dataset(fed_args, script_args, dataset):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(fed_args.num_clients):
            local_datasets.append(dataset.shard(fed_args.num_clients, i))
    
    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args):
    # num2sample 是根据 batch_size(bath_size是脚本传入的), gradient_accumulation_steps 是指累积几个batch才更新一次, max_steps训练的steps数量,应该是指更新几次模型
    num2sample = script_args.batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset)) # 采样的数量不能超过dataset的大小
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample) # 数据的索引是从0到len(dataset), 从中采样num2sample个
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round