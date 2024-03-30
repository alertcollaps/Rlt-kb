import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm
from docx import Document


# Загрузка и предобработка данных из документа Word
class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length_per_fragment=512):
        self.fragments = []
        doc = Document(data_path)
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            # Разделение параграфа на фрагменты заданной длины
            for i in range(0, len(text), max_length_per_fragment):
                fragment = text[i:i + max_length_per_fragment]
                # Дополнение фрагмента пробелами до нужной длины
                fragment += " " * (max_length_per_fragment - len(fragment))
                self.fragments.append(fragment)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.fragments)

    def __getitem__(self, idx):
        text = self.fragments[idx]

        inputs = self.tokenizer.encode(
            text, add_special_tokens=True, truncation=True)
        inputs = torch.tensor(inputs, dtype=torch.long)
        return inputs


# Конфигурация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.train()

# Подготовка данных
dataset = CustomDataset(data_path='fz44.docx', tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Обучение
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        batch = batch.to(device)
        model.zero_grad()
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Average Loss: {total_loss / len(dataloader)}')

# Сохранение модели
torch.save(model.state_dict(), 'trained_gpt2_model.pth')
