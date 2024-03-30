from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
# Загрузка предварительно обученной модели GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Предположим, что у вас есть некоторый корпус текстовых данных, хранящийся в переменной texts
from docx import Document

# Путь к файлу .doc
docx_file = "fz44.docx"

# Чтение текста из файла .doc
doc = Document(docx_file)
texts = []
for paragraph in doc.paragraphs:
    texts.append(paragraph.text)

# Преобразование списка текстов в единый текст
text_data = "\n".join(texts)

# Использование текста для обучения модели
# (вместо предполагаемой переменной texts)
# Пример:
inputs = tokenizer(text_data, return_tensors="pt", truncation=True, padding=True)
dataset = TextDataset(inputs["input_ids"], inputs["attention_mask"])

# Создание аргументов обучения
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
)


# Создание трейнера для обучения модели
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=dataset,
)

# Обучение модели
trainer.train()

# Сохранение обученной модели
model.save_pretrained("./trained_model")
