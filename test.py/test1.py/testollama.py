import ollama
resp = ollama.chat(
    model="mistral",
    messages=[{"role": "user", "content": "Say hello"}]
)
msg = resp["message"]
print(msg.content)
