import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def ask_ai(prompt, system_role = "You are a helpful assistant"):
    """
    A helper that accepts a 'system_role' to give the AI a personality
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":system_role},
            {"role":"user","content":prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


## Input
user_input = input("How can I help you?\n")
print(f"User Input : {user_input}\n")

#Step 1: The Router
router_prompt = f"""
Analyze the user's input and output ONLY one of the following words:
- MATH (if the user ask for calculation)
- CREATIVE (if the user ask for writing , stories, or poem)
- OTHER (for everything else)

Input: {user_input}
"""

#We ask the AI to be a "Classifier"
route = ask_ai(router_prompt, system_role="You are a classification machine.")
print(f"Router decision: {route}\n")

## STEP 2
if "MATH" in route:
    print(">>>Routing to the Mathematician.......")
    math_response = ask_ai(
        user_input,
        system_role = "You are a strict mathematician. Output only the formula and the result. No chit-chat."
    )
    print(f"Result: {math_response}")

elif "CREATIVE" in route:
    print(">>> Routing to be a Poet.....")
    creative_response = ask_ai(
        user_input,
        system_role="You are a Shakespearean poet. Answer in rhyme"
    )
    print(f"Result : {creative_response}\n")

else:
    print(">>>Routing to General Support....")
    general_response = ask_ai(
        user_input
    )
    print(f"Result : {general_response}")



