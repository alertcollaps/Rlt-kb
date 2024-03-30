import torch

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device: {torch.cuda.current_device()} ")

print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)} ")
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Инициализация токенизатора и модели GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Добавляем новый токен для заполнения
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained("gpt2")

train_dataset = TextDataset(tokenizer, file_path="fz44.txt", block_size=128)

# Инициализация аргументов для обучения
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Инициализация трейнера для обучения модели
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=train_dataset,
)

# Запуск обучения
trainer.train()

# Сохранение модели
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")
