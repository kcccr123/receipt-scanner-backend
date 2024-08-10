from torch.utils.data import Dataset

def tokenize_function(example, tokenizer, seq_max_length):
    inputs = tokenizer(example["input"], padding="max_length", max_length = seq_max_length, truncation=True, return_tensors="pt")
    outputs = tokenizer(example["output"], padding="max_length", max_length = seq_max_length, truncation=True, return_tensors="pt")

    return {
        "input_ids": inputs["input_ids"].squeeze(),  # Remove unnecessary dimensions
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": outputs["input_ids"].squeeze()  # The model uses input_ids as labels for the output
    }

class TextDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["input_ids"],
            "attention_mask": self.data[idx]["attention_mask"],
            "labels": self.data[idx]["labels"]
        }
