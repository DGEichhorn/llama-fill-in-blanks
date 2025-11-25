import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

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