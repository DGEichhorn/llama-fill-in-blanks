import os
import re
import random
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

def generate_blanks(text, blanking_prob=0.25):
    tokens = re.findall(r"\w+|[.,!?;:]|\s+", text)
    blanked_tokens = []

    for tok in tokens:
        if re.match(r"\w+", tok):
            if random.random() < blanking_prob:
                blanked_tokens.append("<blank>")
            else:
                blanked_tokens.append(tok)
        else:
            blanked_tokens.append(tok)

    return "".join(blanked_tokens)


load_dotenv()

hf_token = os.getenv("HF_TOKEN")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

client = InferenceClient(
    model=model_id,
    token=hf_token,
)

response = client.chat_completion(
    messages=[
        {"role": "user", "content": "Wo wurde Marcel Reich Ranicki? Was hat er studiert? Wo lebt er heute?"}
    ],
    max_tokens=200,        # ACHTUNG: NICHT max_new_tokens
    temperature=0.9,
    top_p=0.9,
)

print(response.choices[0].message["content"])

print(generate_blanks("Hi, my name is Dominik. What is your name? Where are you from?"))