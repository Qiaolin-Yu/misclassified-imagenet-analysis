import os
import logging
import backoff
import json

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@backoff.on_exception(
    backoff.expo,
    (
            openai.RateLimitError,
            openai.ConflictError,
            openai.UnprocessableEntityError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
            json.JSONDecodeError,
    ),
    max_tries=5,
    jitter=backoff.random_jitter
)
async def openai_generate_json(config, input_text) -> None:
    messages = [{"role": "system", "content": config["system_prompts"]["prompt"]}, 
                {"role": "user", "content": input_text}]
    
    response = await client.chat.completions.create(
        model=config["model_type"],
        max_tokens=config["max_tokens"],
        temperature=config["model_temperature"],
        response_format={"type": "json_object"},
        messages=messages
    )
    json_response = json.loads(response.choices[0].message.content)
    
    logger.info("Successfully generate response with OpenAI")
    return json_response

async def openai_generate_text(config, input_text) -> None:
    messages = [{"role": "system", "content": config["system_prompt"]}, 
                {"role": "user", "content": input_text}]
    
    response = await client.chat.completions.create(
        model=config["model_type"],
        max_tokens=config["max_tokens"],
        temperature=config["model_temperature"],
        response_format={"type": "text"},
        messages=messages
    )

    text_resposne = response.choices[0].message.content

    logger.info("Successfully generate response with OpenAI")
    return text_resposne
    
def create_label_mapping(dataset, superclass_id):
    # Extract all unique labels for the given superclass
    unique_labels = set()
    for _, label, super_label in dataset:
        if super_label == superclass_id:
            unique_labels.add(label)
    
    # Create a mapping from original labels to new labels (0 to n-1)
    label_mapping = {original_label: new_label for new_label, original_label in enumerate(sorted(unique_labels))}
    return label_mapping