#!/usr/bin/env python3
"""
PyTorch neural network for behavioral cloning.

Predicts [speed, steering_angle] from state features.
"""

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


class BCNetwork(nn.Module):
    """
    MLP for behavioral cloning.

    Input:  state features (e.g., [x, y, yaw, v_long, v_lat, omega])
    Output: [speed, steering_angle]

    TODO: You must implement:
        1. __init__  — Define your network architecture and optimizer
        2. forward   — Forward pass through the network
        3. train_step — Single training step (forward + loss + backprop)
    """

    def __init__(self, input_size=6, output_size=2, hidden_sizes=None, lr=1e-3):
        super().__init__()

        # TODO: Define your network architecture.
        # Use nn.Sequential, nn.Linear, and activation functions (e.g., nn.ReLU).
        # The network should map input_size -> output_size.
        #
        # Example (you can change this):
        #   self.network = nn.Sequential(
        #       nn.Linear(input_size, 64),
        #       nn.ReLU(),
        #       ...
        #       nn.Linear(??, output_size),
        #   )

        raise NotImplementedError("TODO: Define your network architecture")

        # TODO: Define your optimizer and loss function.
        # Example:
        #   self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Feel free to define or utilize other loss functions
        # https://docs.pytorch.org/docs/stable/nn.html#loss-functions
        #   self.loss_fn = nn.MSELoss() 

        raise NotImplementedError("TODO: Define your optimizer and loss function")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input tensor (N, input_size)

        Returns:
            output tensor (N, output_size)
        """
        # TODO: Pass x through your network and return the output.
        raise NotImplementedError("TODO: Implement forward pass")

    def train_step(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> float:
        """
        Single training step on a batch.

        Args:
            x: input features (N, input_size)
            y: target actions (N, output_size)
            w: optional per-sample weights (N,)

        Returns:
            loss value (float)

        TODO: Implement the training step:
            1. Convert numpy arrays to tensors using torch.FloatTensor()
            2. Forward pass
            3. Compute loss (MSE). If weights w are provided, use weighted MSE:
               loss = mean(w * (pred - target)^2)
            4. Zero gradients, backprop, optimizer step
            5. Return loss.item()
        """
        raise NotImplementedError("TODO: Implement training step")

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference. Do not modify this method.

        Args:
            x: input features (N, input_size) or (input_size,)

        Returns:
            predicted actions as numpy array
        """
        self.eval()
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        x_t = torch.FloatTensor(x)
        pred = self.forward(x_t).numpy()

        if single:
            return pred[0]
        return pred

    def train_epoch(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None, batch_size=64, shuffle=True):
        """
        Train for one full epoch over the dataset. Do not modify this method.

        Args:
            w: optional per-sample weights (N,)

        Returns:
            average loss over all batches
        """
        n = x.shape[0]
        if shuffle:
            idx = np.random.permutation(n)
            x, y = x[idx], y[idx]
            if w is not None:
                w = w[idx]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            w_batch = w[i : i + batch_size] if w is not None else None
            total_loss += self.train_step(x_batch, y_batch, w_batch)
            n_batches += 1

        return total_loss / n_batches

    def save_model(self, path):
        """Save model weights. Do not modify this method."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """Load model weights. Do not modify this method."""
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
