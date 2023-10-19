import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(model_dir)

    return model.to(device)


def train(args):
    # Load and preprocess data
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train model
    for epoch in range(args.epochs):
        for i, (input_features, labels) in enumerate(zip(X_train, y_train)):
            inputs = tokenizer(
                str(input_features.tolist()),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            labels = torch.tensor([labels]).unsqueeze(0)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")

    # Save model
    model_dir = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--model_dir", type=str)

    args, _ = parser.parse_known_args()

    train(args)
