import torch
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from utils import tokenize_function, TextDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
import string
import pandas as pd
import os
from datetime import datetime

csv_path = r"D:\photos\Words\walmart\home\sdf\marketing_sample_for_walmart_com-product_details__20200101_20200331__30k_data.csv"
save_dir = r"D:\photos\Words\walmart\home\sdf"
model_dir = r"D:\Projects\reciept-scanner\BART"
checkpt_dir = os.path.join(model_dir, datetime.strftime(datetime.now(), "%Y%m%d%H%M"), "bart_model.pt").replace("\\","/")


def misspell(word):
    result = word
    repeat = 1
    if len(word) > 1:
        op = random.choice(['rmv', 'rpl'])
        pos = random.randint(0, len(word) - 1)
        if op == 'rmv':
            result = word[:pos] + word[pos+1:]
        elif op == 'rpl':
            result = word[:pos] + random.choice(string.ascii_letters) + word[pos+1:]
        
        if len(word) > 5:
            repeat = 3
        
        i = 0
        while i < repeat:
            op = random.choice(['rmv', 'rpl'])
            pos = random.randint(0, len(word) - 1)
            if op == 'rmv':
                result = result[:pos] + result[pos+1:]
            elif op == 'rpl':
                result = result[:pos] + random.choice(string.ascii_letters) + result[pos+1:]
            i+=1
        
    return result


def add_noise(name):
    words = name.split()

    random.shuffle(words)

    noisy_words = []
    for word in words:
        if random.random() < 0.5:
            word = misspell(word)
        if random.random() < 0.2:
            word = ''.join([word, ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=random.randint(1, 5)))])
        noisy_words.append(word)

    return ' '.join(noisy_words)

csv = pd.read_csv(csv_path)

data_set = []
for i in range(0, csv.shape[0]):
    grd_truth = csv.loc[i].at["Product Name"]
    if len(grd_truth) < 61 and grd_truth != "":
        variations = 12 #5 + random.choice(range(0,6))
        id = 0;
        while id < variations:
            temp = {}
            temp["input"] = add_noise(grd_truth)
            temp["output"] = grd_truth
            data_set.append(temp)
            id += 1

print("dataset of " + str(len(data_set)))

train, val = train_test_split(data_set, test_size=0.1)

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#HYPERPARAMS-----------------------------------------------------------------------
batch_size = 8
learning_rate = 3e-5
epochs = 8
sentence_length = 60

# load model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

if torch.cuda.is_available():
    model = model.cuda()
    print("CUDA Enabled...Training On GPU")

# create tokens for each data input
# tokenize_function input is (data set input, model tokenizer, and max length of string)

tokenized_train = [tokenize_function(i, tokenizer, sentence_length) for i in train]
tokenized_val = [tokenize_function(i, tokenizer, sentence_length) for i in val]

# create training set
train_dataset = TextDataset(tokenized_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("### Training Dataset Created")

#create val set
val_dataset = TextDataset(tokenized_val)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

print("### Validation Dataset Created")

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

#set model to training
model.train()
print("### Begin Training---")
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Training phase
    total_train_loss = 0
    correct_predictions = 0
    total_predictions = 0
    correct_sentences = 0
    total_sentences = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        # Ensure the correct batch structure
        # print(batch)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Pass the inputs to the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Training Loss: {avg_train_loss}")
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    correct_sentences = 0
    total_sentences = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Pass the inputs to the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

            # Get model predictions (greedy decoding)
            predictions = outputs.logits.argmax(dim=-1)
            
            # Flatten the tensors for token-level comparison
            predictions_flat = predictions.view(-1).to(device)
            labels_flat = batch["labels"].view(-1).to(device)
            
            # Only consider non-padding tokens (ignore index 0 or whatever your pad token is)
            non_padding_mask = labels_flat != torch.tensor(tokenizer.pad_token_id, device=labels_flat.device)
            num_correct = (predictions_flat == labels_flat) & non_padding_mask
            
            correct_predictions += num_correct.sum().item()
            total_predictions += non_padding_mask.sum().item()

            # Calculate sentence-level accuracy
            for i in range(batch["input_ids"].size(0)):
                label_sentence = batch["labels"][i][non_padding_mask[i]].cpu().tolist()  # Get the label sentence excluding padding
                predicted_sentence = predictions[i][non_padding_mask[i]].cpu().tolist()  # Get the predicted sentence excluding padding
                
                if label_sentence == predicted_sentence:
                    correct_sentences += 1
                total_sentences += 1
    
    # token level accuracy
    token_level_accuracy = correct_predictions / total_predictions
    print(f"Token-Level Validation Accuracy: {token_level_accuracy:.4f}")

    # sentence level accuracy
    sentence_level_accuracy = correct_sentences / total_sentences
    print(f"Sentence-Level Validation Accuracy: {sentence_level_accuracy:.4f}")
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss}")
    
    # Set the model back to training mode for the next epoch
    model.train()
print("### Training Complete")

print("### Saving Model---")
model.eval()

os.makedirs(os.path.dirname(checkpt_dir))
torch.save(model.state_dict(), checkpt_dir)
tokenizer.save_pretrained(os.path.dirname(checkpt_dir))