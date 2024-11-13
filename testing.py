from transformers import pipeline

from args import get_args

def test():
    message = input("user: ")
    messages = [{ "role": "user", "content": message }]
    pipe = pipeline("text-generation", model=get_args().llm_generating_model_name, trust_remote_code=True, device_map="auto")
    results = pipe(messages, max_length=1024)
    for result in results[0]['generated_text'][1:]:
        print(f"{result['role']}: {result['content']}")