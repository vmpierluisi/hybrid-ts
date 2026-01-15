import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn.utils import weight_norm

import warnings
from src.models import Classic_TCN
from src.models import AdditiveHybrid_AR_TCN
from src.models import MultiplicativeHybrid_AR_TCN
warnings.filterwarnings('ignore')

def train_tcn_model(y_train, num_epochs=100, lr=0.001, seed=42):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train)

    model = Classic_TCN(kernel_size=3, num_filters=64, num_layers=5, dilation_base=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions, targets = model(y_train)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Classic TCN - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    return model


def train_additive_model(y_train, ar_order=1, num_epochs=100, lr=0.001, seed=42):
    """
    Train additive hybrid model with specified AR order

    Args:
        y_train: training data
        ar_order: AR order (e.g., 1 for AR(1), 5 for AR(5))
        num_epochs: number of training epochs
        lr: learning rate
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train)

    model = AdditiveHybrid_AR_TCN(
        ar_order=ar_order,
        kernel_size=3,
        num_filters=64,
        num_layers=3,
        dilation_base=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions, targets = model(y_train)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Additive Hybrid AR({ar_order}) + TCN - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    return model


def train_multiplicative_model(y_train, ar_order=1, num_epochs=100, lr=0.001, seed=42):
    """
    Train multiplicative hybrid model with specified AR order

    Args:
        y_train: training data
        ar_order: AR order (e.g., 1 for AR(1), 5 for AR(5))
        num_epochs: number of training epochs
        lr: learning rate
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train)

    model = MultiplicativeHybrid_AR_TCN(
        ar_order=ar_order,
        kernel_size=3,
        num_filters=64,
        num_layers=3,
        dilation_base=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions, targets = model(y_train)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Multiplicative Hybrid AR({ar_order})+TCN - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

    return model

