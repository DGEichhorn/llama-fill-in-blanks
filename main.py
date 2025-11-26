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



def fill_blanks(blanked_text, model, token):
    client = InferenceClient(model=model, token=token)

    system_prompt = f"""
    You are a strict text reconstruction engine.
    
    You must adhere to the following rules:
    1. You must replace each <blank> by exactly one word that fits the context of the text.
    2. Do not rephrase, rewrite, reorder, or improve the sentence.
    3. Do not change any existing words.
    4. Preserve punctuation and spacing exactly.
    5. Do not output any extra text. Do just return the final reconstructed sentence.
    """

    user_prompt = f"""
    Fill in the blanks marked by <blank> in the following text:
    {blanked_text}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat_completion(
        messages=messages,
        max_tokens=200,
        temperature=0.9,
        top_p=0.9,
    )
    return response.choices[0].message["content"]



load_dotenv()

hf_token = os.getenv("HF_TOKEN")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"


#text = "Hi, my name is Dominik. What is your name? Where are you from?"
text = "It is now Advent, and the festive season has begun with a special sense of anticipation. People all around are preparing their homes with lights, candles, and decorations. The streets feel warmer and more joyful despite the cold weather. Everyone is looking forward to celebrating Christmas with family and friends soon. The atmosphere is filled with hope, tradition, and holiday spirit."
blanked_text = generate_blanks(text)

print(blanked_text)
print(fill_blanks(blanked_text, model_id, hf_token))