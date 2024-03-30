import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка токенизатора и модели GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")
model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")

# Проверка доступности CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Генерация текста
prompt_text = "Какие штрафы предусмотрены за неисполнение контракта?"
input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
output = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_return_sequences=5, temperature=None, do_sample=True)


# Декодирование и вывод сгенерированного текста
for i, sample_output in enumerate(output):
    print(f"Example {i+1}: {tokenizer.decode(sample_output, skip_special_tokens=True)}")
