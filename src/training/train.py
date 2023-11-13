import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


# Assuming the data is in a simple TSV format: label followed by text
class CustomDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.tokenizer = tokenizer
        self.samples = []
        with open(file_path, "r") as f:
            for line in f:
                label, text = line.strip().split("\t")
                self.samples.append((int(label), text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text = self.samples[idx]
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        return label, input_ids, attention_mask


def train(model, dataloader, optimizer):
    model.train()
    total_acc, total_count = 0, 0
    for idx, (label, input_ids, attention_mask) in enumerate(dataloader):
        input_ids, attention_mask, label = (
            input_ids.to(device),
            attention_mask.to(device),
            label.to(device),
        )
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=label)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_acc += (outputs.logits.argmax(1) == label).sum().item()
        total_count += label.size(0)
    return total_acc / total_count


if __name__ == "__main__":
    # Paths for input data and output model specified by SageMaker
    input_data_path = os.environ["SM_CHANNEL_TRAIN"]
    output_model_path = os.environ["SM_MODEL_DIR"]

    # Load tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = CustomDataset(tokenizer, os.path.join(input_data_path, "train.tsv"))
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Define model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    for epoch in range(3):  # number of epochs
        train_acc = train(model, train_dataloader, optimizer)
        print(f"Epoch {epoch}, Training Accuracy: {train_acc}")

    # Save the model to the path specified by SageMaker
    model_save_path = os.path.join(output_model_path, "model.pth")
    torch.save(model.state_dict(), model_save_path)
