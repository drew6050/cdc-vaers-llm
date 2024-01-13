# %%
#!jupyter nbconvert --to script config_template.ipynb
#jupyter: create interactive window

# %%
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader


# Load and preprocess the VAERS data and symptoms
vaers_data_path = 'data/2023VAERSDATA.csv'
vaers_symptoms_path = 'data/2023VAERSSYMPTOMS.csv'
vaers_data = pd.read_csv(vaers_data_path, encoding='ISO-8859-1')
vaers_symptoms = pd.read_csv(vaers_symptoms_path, encoding='ISO-8859-1')

# Merge datasets on VAERS_ID
merged_data = vaers_data.merge(vaers_symptoms, on='VAERS_ID')
merged_data['SYMPTOM_TEXT'] = merged_data['SYMPTOM_TEXT'].astype(str)

# Get the unique labels count
number_of_symptom_codes = len(vaers_symptoms['SYMPTOM1'].unique())  

# %%
# Reduce to just some rows for testing
merged_data = merged_data[0:200]

# %%
# Preprocess the data for DistilBERT
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(merged_data['SYMPTOM_TEXT'].tolist(), truncation=True, padding=True)

# %%
# Split the data
train_texts, val_texts = train_test_split(merged_data['SYMPTOM_TEXT'].tolist(), test_size=0.1)
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# %%
from torch.utils.data import DataLoader

# PyTorch Dataset
class VAERSSymptomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = VAERSSymptomDataset(train_encodings)
val_dataset = VAERSSymptomDataset(val_encodings)

# Load Pretrained DistilBERT Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=number_of_symptom_codes)

# DataLoader for validation set
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# %%
# Evaluation Function
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += len(batch)
            correct += (predicted == batch['labels']).sum().item()
    return correct / total



# %%
# Evaluate the Model Without Fine-Tuning
print("Evaluating Pretrained Model...")
pretrained_accuracy = evaluate_model(model, val_loader)
print(f'Pretrained Model Accuracy: {pretrained_accuracy:.4f}')

# %%
# Fine-Tuning the Model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("Starting Fine-Tuning...")
trainer.train()

# Evaluate the Fine-Tuned Model
print("Evaluating Fine-Tuned Model...")
fine_tuned_accuracy = evaluate_model(model, val_loader)
print(f'Fine-Tuned Model Accuracy: {fine_tuned_accuracy:.4f}')

