import json
import os

def format_data(item: dict):
    messages = {
        "messages_foreignization": [{"role": "user", "content": f"Translate the following text to Malay:\n\n{item['source']}"}, {"role": "assistant", "content": item['foreignization']}],
        "messages_domestication": [{"role": "user", "content": f"Translate the following text to Malay:\n\n{item['source']}"}, {"role": "assistant", "content": item['domestication']}]
    } 

    item.update(messages)
    return item

data_dir = "data/synthetic/enms/parallel_asian_treebank_qwen/"
data_files = os.listdir(data_dir)


data_files = [data_dir + data_file for data_file in data_files]

for data_file in data_files:
    if data_file.endswith("jsonl"):
        continue
    data = json.load(open(data_file, "r"))
    data = list(map(lambda item: format_data(item), data))
    with open(f"{data_file.split('.')[0]}.jsonl", "w") as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')