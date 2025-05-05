#!/usr/bin/env python3
# filepath: /home/daan/repos/thesis/federated-learning-demo/main.py
"""
A simple logistic regression model for MNIST dataset with basic federated
learning. This script demonstrates training and evaluation of a basic logistic
regression classifier on the MNIST dataset using a simulated federated learning
approach with 3 clients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


class LogisticRegression(nn.Module):
    """Simple logistic regression model for MNIST classification."""

    def __init__(self):
        super(LogisticRegression, self).__init__()
        # MNIST images are 28x28 pixels = 784 features
        # Output has 10 classes (digits 0-9)
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        # Flatten the input from [batch_size, 1, 28, 28] to [batch_size, 784]
        x = x.view(-1, 28 * 28)
        # Apply linear transformation
        return self.linear(x)

    def get_weights(self):
        """Return the model's weights for aggregation."""
        return {
            "weight": self.linear.weight.data.clone(),
            "bias": self.linear.bias.data.clone(),
        }

    def set_weights(self, weights):
        """Set the model's weights from aggregated weights."""
        self.linear.weight.data = weights["weight"].clone()
        self.linear.bias.data = weights["bias"].clone()


def load_federated_data(batch_size=64, num_clients=3):
    """Load and prepare MNIST dataset for federated learning with multiple clients."""

    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std of MNIST
        ]
    )

    # Load full training data
    full_train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )

    # Calculate the size of each client's dataset
    dataset_size = len(full_train_dataset)
    client_dataset_size = dataset_size // num_clients

    # Create indices for each client's data subset
    indices = list(range(dataset_size))

    # Split indices for each client
    client1_indices = indices[0:client_dataset_size]
    client2_indices = indices[client_dataset_size : 2 * client_dataset_size]
    client3_indices = indices[2 * client_dataset_size : 3 * client_dataset_size]

    # Create client datasets
    client1_dataset = Subset(full_train_dataset, client1_indices)
    client2_dataset = Subset(full_train_dataset, client2_indices)
    client3_dataset = Subset(full_train_dataset, client3_indices)

    # Create data loaders for each client
    client1_loader = DataLoader(client1_dataset, batch_size=batch_size, shuffle=True)
    client2_loader = DataLoader(client2_dataset, batch_size=batch_size, shuffle=True)
    client3_loader = DataLoader(client3_dataset, batch_size=batch_size, shuffle=True)

    # Load test data (stays the same for evaluation)
    test_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return client1_loader, client2_loader, client3_loader, test_loader


def train_model(model, train_loader, epochs=5, learning_rate=0.01):
    """Train the logistic regression model."""

    # Use Cross Entropy Loss (combines softmax and NLL loss)
    criterion = nn.CrossEntropyLoss()

    # Use Stochastic Gradient Descent optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute predicted outputs
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass: compute gradient of loss with respect to parameters
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate batch loss
            running_loss += loss.item()

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model, losses


def aggregate_weights(client_weights_list):
    """Aggregate weights from multiple clients (simple averaging)."""

    # Initialize the aggregated weights with the structure of the first client
    aggregated_weights = {
        "weight": torch.zeros_like(client_weights_list[0]["weight"]),
        "bias": torch.zeros_like(client_weights_list[0]["bias"]),
    }

    # Sum all client weights
    for client_weights in client_weights_list:
        aggregated_weights["weight"] += client_weights["weight"]
        aggregated_weights["bias"] += client_weights["bias"]

    # Average the weights by dividing by the number of clients
    num_clients = len(client_weights_list)
    aggregated_weights["weight"] /= num_clients
    aggregated_weights["bias"] /= num_clients

    return aggregated_weights


def evaluate_model(model, test_loader):
    """Evaluate the model on test data."""

    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # No need to track gradients for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            # Forward pass
            outputs = model(images)

            # Get predicted class from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Update counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store for confusion matrix
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy, all_preds, all_labels


def plot_results(losses_clients, global_losses, final_accuracy, all_preds, all_labels):
    """Plot training loss curve and confusion matrix."""

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Plot client training losses
    for i, client_losses in enumerate(losses_clients):
        ax1.plot(range(1, len(client_losses) + 1), client_losses, label=f"Client {i+1}")
    ax1.set_title("Client Training Losses")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()

    # Plot global model losses
    ax2.plot(
        range(1, len(global_losses) + 1), global_losses, color="black", linestyle="--"
    )
    ax2.set_title("Global Model Loss")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.grid(True)

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3)
    ax3.set_title(f"Confusion Matrix\nFinal Accuracy: {final_accuracy:.4f}")
    ax3.set_xlabel("Predicted Label")
    ax3.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig("federated_mnist_results.png")
    plt.show()


def main_federated():
    """Main function to run the federated learning example with MNIST."""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Hyperparameters
    batch_size = 64
    local_epochs = 3  # Number of epochs for each client's local training
    federated_rounds = 5  # Number of federated learning rounds
    learning_rate = 0.01

    print("Loading MNIST data for federated learning...")
    client1_loader, client2_loader, client3_loader, test_loader = load_federated_data(
        batch_size, num_clients=3
    )

    # Create a global model to be sent to clients
    global_model = LogisticRegression()

    # Keep track of losses for visualization
    client1_losses = []
    client2_losses = []
    client3_losses = []
    global_losses = []

    # Federated learning rounds
    for federated_round in range(federated_rounds):
        print(
            f"\n--- Federated Learning Round {federated_round+1}/{federated_rounds} ---"
        )

        # Initialize client models with current global weights
        client1_model = LogisticRegression()
        client2_model = LogisticRegression()
        client3_model = LogisticRegression()

        # Copy global weights to client models
        global_weights = global_model.get_weights()
        client1_model.set_weights(global_weights)
        client2_model.set_weights(global_weights)
        client3_model.set_weights(global_weights)

        # Train client 1
        print("\nTraining Client 1...")
        client1_model, client1_round_losses = train_model(
            client1_model,
            client1_loader,
            epochs=local_epochs,
            learning_rate=learning_rate,
        )
        client1_losses.extend(client1_round_losses)

        # Train client 2
        print("\nTraining Client 2...")
        client2_model, client2_round_losses = train_model(
            client2_model,
            client2_loader,
            epochs=local_epochs,
            learning_rate=learning_rate,
        )
        client2_losses.extend(client2_round_losses)

        # Train client 3
        print("\nTraining Client 3...")
        client3_model, client3_round_losses = train_model(
            client3_model,
            client3_loader,
            epochs=local_epochs,
            learning_rate=learning_rate,
        )
        client3_losses.extend(client3_round_losses)

        # Get weights from all clients
        client1_weights = client1_model.get_weights()
        client2_weights = client2_model.get_weights()
        client3_weights = client3_model.get_weights()

        # Aggregate weights from all clients
        aggregated_weights = aggregate_weights(
            [client1_weights, client2_weights, client3_weights]
        )

        # Update the global model with aggregated weights
        global_model.set_weights(aggregated_weights)

        # Evaluate the global model after this round
        print("\nEvaluating global model...")
        accuracy, _, _ = evaluate_model(global_model, test_loader)
        global_losses.append(
            accuracy
        )  # Using accuracy as a proxy for global model performance

    # Final evaluation of the global model
    print("\n--- Final Evaluation of Global Model ---")
    final_accuracy, all_preds, all_labels = evaluate_model(global_model, test_loader)

    # Plot results
    print("Plotting results...")
    losses_clients = [client1_losses, client2_losses, client3_losses]
    plot_results(losses_clients, global_losses, final_accuracy, all_preds, all_labels)

    print("Done!")


if __name__ == "__main__":
    main_federated()
