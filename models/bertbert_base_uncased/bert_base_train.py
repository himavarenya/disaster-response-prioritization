# train.py
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load datasets
df_train = pd.read_csv('canada_wildfires_2016_train.tsv', sep='\t')
df_dev = pd.read_csv('canada_wildfires_2016_dev.tsv', sep='\t')
df_test = pd.read_csv('canada_wildfires_2016_test.tsv', sep='\t')

# Define urgent classes
URGENT_CLASSES = {
    'injured_or_dead_people',
    'missing_or_found_people',
    'requests_or_urgent_needs',
    'displaced_people_and_evacuations',
    'rescue_volunteering_or_donation_effort'
}

# Function to assign urgency labels
def label_urgency(row):
    return 1 if row in URGENT_CLASSES else 0

# Apply to datasets
for df in [df_train, df_dev, df_test]:
    df['urgency_label'] = df['class_label'].apply(label_urgency)

# Tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(df_train['tweet_text']), truncation=True, padding=True)
val_encodings = tokenizer(list(df_dev['tweet_text']), truncation=True, padding=True)

train_labels = list(df_train['urgency_label'])
val_labels = list(df_dev['urgency_label'])

# Dataset class
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = TweetDataset(train_encodings, train_labels)
val_dataset = TweetDataset(val_encodings, val_labels)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
