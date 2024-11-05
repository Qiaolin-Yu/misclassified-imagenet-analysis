import json
import yaml
import asyncio
from utils import openai_generate_text, openai_generate_json

IMAGENET_CLASS_INDEX_PATH = "./data/ImageNetV2/imagenet_class_index.json"
CONFIG_PATH = "./src/prompt_superclassing.yaml"
SUPER_CLASS_INDEX_PATH = "./data/ImageNetV2/superclass/"+'superclass_index.json'

superclass_names = [
    "Bird",
    "Boat",
    "Car",
    "Cat",
    "Dog",
    "Fruit",
    "Fungus",
    "Insect",
    "Monkey"
]

with open(IMAGENET_CLASS_INDEX_PATH, 'r') as f:
    imagenet_class_index = json.load(f)

with open(CONFIG_PATH, "r") as config_file:
    """
    Load the configuration file.
    """    
    config = yaml.safe_load(config_file)

# Example Usage
async def generate_response_async(config, input_text):
    """
    Generate a response asynchronously.
    """
    return await openai_generate_json(config, input_text)

async def map_label_to_superclass(idx_to_class_desc):
    """
    Classify original label into superclassing labels.
    """
    tasks = []
    label_to_superclass = {}
    superclass_to_idx = {name: idx for idx, name in enumerate(superclass_names)}

    for index, item in enumerate(idx_to_class_desc.items()):
        label_idx=item[0]
        label=item[1]

        input_text = config["input_format"].format(
            label=label
        )

        task = asyncio.create_task(generate_response_async(config, input_text))
        tasks.append((index, task))

    responses = await asyncio.gather(*[task for _, task in tasks])

    # process responses
    for (index, _), response in zip(tasks, responses):
        superclass=response.get('result')
        if superclass is not None and superclass in superclass_names:
            label_to_superclass[index]=superclass_to_idx[superclass]

    return label_to_superclass

if __name__ == "__main__":

    # Map label indices to class descriptions, replacing spaces with underscores
    idx_to_class_desc = {}
    for key, value in imagenet_class_index.items():
        label = int(key)
        class_desc = value[1]
        if " " in class_desc:
            class_desc = class_desc.replace(" ", "_")
        idx_to_class_desc[label] = class_desc

    label_to_superclass = asyncio.run(map_label_to_superclass(idx_to_class_desc))

    with open(SUPER_CLASS_INDEX_PATH, 'w') as json_file:
        json.dump(label_to_superclass, json_file, indent=4)
