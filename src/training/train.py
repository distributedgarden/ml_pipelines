import os
import csv
import torch
import logging

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """
    Dataset class for handling CSV format data.
    """

    def __init__(self, tokenizer, file_path):
        self.tokenizer = tokenizer
        self.samples = self.load_data(file_path)

    def load_data(self, file_path):
        """
        Load and preprocess data from a CSV file.
        """
        try:
            samples = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip the header row if it exists
                for row in reader:
                    text, label = row
                    samples.append((int(label), text))
            logger.info("Data loaded successfully from %s", file_path)
            return samples
        except Exception as e:
            logger.error("Failed to load data from %s: %s", file_path, e)
            raise

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


def train(model, dataloader, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_acc, total_count, total_loss = 0, 0, 0

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
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Compute and log metrics
        total_loss += loss.item()
        total_acc += (outputs.logits.argmax(1) == label).sum().item()
        total_count += label.size(0)

        if (idx + 1) % 10 == 0:  # Log every 10 batches
            logger.info(
                f"Batch {idx + 1}: Loss: {total_loss / (idx + 1)}, "
                f"Accuracy: {total_acc / total_count}"
            )

    # Log epoch-level metrics
    logger.info(
        f"Training completed: Total Loss: {total_loss / len(dataloader)}, "
        f"Total Accuracy: {total_acc / total_count}"
    )

    return total_acc / total_count


def main():
    """
    Execute training.
    """
    try:
        input_data_path = os.environ.get(
            "SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"
        )
        output_model_path = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

        logger.info(f"input_data_path: {input_data_path}")
        logger.info(f"output_model_path: {output_model_path}")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = CustomDataset(tokenizer, os.path.join(input_data_path, "train.csv"))
        train_dataset, _ = train_test_split(dataset, test_size=0.1)

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=5e-5)

        for epoch in range(3):
            train_acc = train(model, train_dataloader, optimizer, device)
            logger.info(f"Epoch {epoch}, Training Accuracy: {train_acc}")

        model_save_path = os.path.join(output_model_path, "model.pth")
        torch.save(model.state_dict(), model_save_path)

    except Exception as e:
        logger.error("Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
