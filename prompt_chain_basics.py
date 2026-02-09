# Social Media Manager Chain
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def ask_ai(prompt):
    """
    Sends a prompt to OpenAI and returns just the text response.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"user","content":prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


#----THE CHAIN----

raw_product_data = """
Product : SuperBottle 3000
It keeps water cold for 24 hours. Made of steel.
Price is $20. It has a hancle.
Color: Nenon Green.
"""

print(f"---INPUT DATA ---\n{raw_product_data}\n")

#LINK 1: Extraction
print("Executing Step 1: Extraction...")
prompt_1= f"""
Read the following product text and extract 3 key bullet points:
{raw_product_data}
"""

step_1_output= ask_ai(prompt_1)
print(f"---STEP 1 OUTPUT (Key Features)---\n{step_1_output}\n")


# LINK 2:Creative writing
print("Executing step 2: Creative writing....")

prompt_2= f"""
Write an enthusiastic Instagram caption using these key features:{step_1_output}
Make sure to include emojis and hastags
"""

step_2_output = ask_ai(prompt_2)

print(f"---STEP 2 OUTPUT (Final Caption)---\n{step_2_output}\n")


