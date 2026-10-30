# train.py

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

# Load the training and validation datasets
df_train = pd.read_csv('canada_wildfires_2016_train.tsv', sep='\t')
df_dev = pd.read_csv('canada_wildfires_2016_dev.tsv', sep='\t')

# Define the classes considered "urgent"
URGENT_CLASSES = {
    'injured_or_dead_people',
    'missing_or_found_people',
    'requests_or_urgent_needs',
    'displaced_people_and_evacuations',
    'rescue_volunteering_or_donation_effort'
}

# Function to assign urgency label
def label_urgency(row):
    return 1 if row in URGENT_CLASSES else 0

# Apply urgency labels
for df in [df_train, df_dev]:
    df['urgency_label'] = df['class_label'].apply(label_urgency)

# Tokenizer for RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('facebookai/roberta-base')

# Tokenize datasets
train_encodings = tokenizer(list(df_train['tweet_text']), truncation=True, padding=True)
val_encodings = tokenizer(list(df_dev['tweet_text']), truncation=True, padding=True)

train_labels = list(df_train['urgency_label'])
val_labels = list(df_dev['urgency_label'])

# Define Dataset class for PyTorch
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

# Create train and validation datasets
train_dataset = TweetDataset(train_encodings, train_labels)
val_dataset = TweetDataset(val_encodings, val_labels)

# Load the pre-trained RoBERTa model with a classification head
model = RobertaForSequenceClassification.from_pretrained('facebookai/roberta-base', num_labels=2)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    report_to="none",  # Disable W&B logging
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./roberta_model")
tokenizer.save_pretrained("./roberta_model")
