# infer.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("./final_model")
tokenizer = BertTokenizer.from_pretrained("./final_model")

# Function to classify a tweet
def classify_tweet():
    # Ask for tweet input
    new_tweet = input("Enter the tweet: ")

    # Tokenize the input
    inputs = tokenizer(new_tweet, padding=True, truncation=True, return_tensors="pt")

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted class (0 or 1)
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Map predicted class to a label
    if predicted_class == 0:
        print("Not urgent")
    else:
        print("Urgent")

# Call the function for user input and prediction
classify_tweet()
