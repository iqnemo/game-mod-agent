from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
r = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role":"user","content":"Say OK"}],
)
print(r.choices[0].message.content)
