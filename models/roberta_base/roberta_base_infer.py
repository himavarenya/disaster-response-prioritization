# infer.py

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('./roberta_model')
model = RobertaForSequenceClassification.from_pretrained('./roberta_model', num_labels=2)

# Move model to the appropriate device (GPU/CPU)
model.to(device)


# Function to predict urgency based on input text
def predict_urgency(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Make the prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to predicted labels
    predictions = torch.argmax(logits, dim=-1)

    # Map the predicted label to its meaning (0 = Not Urgent, 1 = Urgent)
    label_mapping = {0: "Not Urgent", 1: "Urgent"}
    predicted_label = label_mapping[predictions.item()]

    return predicted_label

# Function to take user input for prediction
def run_inference():
    tweet = input("Enter the tweet or text to classify: ")
    result = predict_urgency(tweet)
    print(f"Predicted Urgency: {result}")

if __name__ == "__main__":
    run_inference()
