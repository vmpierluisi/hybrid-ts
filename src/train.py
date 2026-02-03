import torch
import torch.nn as nn
import numpy as np
import random


import warnings
import pandas as pd
from src.models import Classic_TCN
from src.models import AdditiveHybrid_AR_TCN
from src.models import MultiplicativeHybrid_AR_TCN

warnings.filterwarnings('ignore')

def _ensure_tensor(data):
    """ Helper to convert input to torch tensor, handling Darts TimeSeries objects """
    if isinstance(data, torch.Tensor):
        return data
    if hasattr(data, 'values'):
        # Handles Darts TimeSeries
        return torch.FloatTensor(data.values().flatten())
    if isinstance(data, (list, np.ndarray)):
        return torch.FloatTensor(np.array(data).flatten())
    return torch.FloatTensor(data)

def train_tcn_model(y_train, kernel_size=3, num_channels=[64]*5, dilations=None, num_epochs=100, lr=0.001, seed=42):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    y_train = _ensure_tensor(y_train)

    model = Classic_TCN(num_channels=num_channels, kernel_size=kernel_size, dilations=dilations)
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


def train_additive_model(y_train, ar_order=1, kernel_size=3, num_channels=[64]*3, dilations=None, num_epochs=100, lr=0.001, seed=42):
    """
    Train additive hybrid model with specified AR order
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    y_train = _ensure_tensor(y_train)
    model = AdditiveHybrid_AR_TCN(
        ar_order=ar_order,
        kernel_size=kernel_size,
        num_channels=num_channels,
        dilations=dilations
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


def train_multiplicative_model(y_train, ar_order=1, kernel_size=3, num_channels=[64]*3, dilations=None, num_epochs=100, lr=0.001, seed=42):
    """
    Train multiplicative hybrid model with specified AR order
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    y_train = _ensure_tensor(y_train)
    model = MultiplicativeHybrid_AR_TCN(
        ar_order=ar_order,
        kernel_size=kernel_size,
        num_channels=num_channels,
        dilations=dilations
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



def cross_validate_additive_hybrid(train_series, val_series, ar_order, hyperparameter_grid, num_epochs=100, seed=42):
    """
    Cross-validation for Additive Hybrid AR-TCN based on forecasting errors to avoid overfitting.

    Args:
        train_series: Training TimeSeries
        val_series: Validation TimeSeries
        ar_order: AR order (1 or 5 based on previous selection)
        hyperparameter_grid: Dictionary of hyperparameters to search
        num_epochs: Training epochs for each configuration
        seed: Random seed
    Returns:
        best_config: Best hyperparameter configuration
        results_df: DataFrame with all results
    """
    train_tensor = _ensure_tensor(train_series)
    val_tensor = _ensure_tensor(val_series)
    results = []

    #generate all combinations (similar to TCN)
    from itertools import product
    keys = hyperparameter_grid.keys()
    values = hyperparameter_grid.values()

    for config_values in product(*values):
        config = dict(zip(keys, config_values))

        print(f"\n{'='*60}")
        print(f"Testing Additive Hybrid AR({ar_order}) configuration: {config}")
        print(f"{'='*60}")

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Prepare model configuration
        num_channels = config.get('num_channels')
        if num_channels is None and 'num_filters' in config and 'num_layers' in config:
            num_channels = [config['num_filters']] * config['num_layers']
        
        dropout = config.get('dropout', 0.1)

        #Train model
        model = AdditiveHybrid_AR_TCN(
            ar_order=ar_order,
            kernel_size=config['kernel_size'],
            num_channels=num_channels,
            dropout=dropout
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.MSELoss()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            predictions, targets = model(train_tensor)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.6f}')

        #evaluate on validation test
        model.eval()
        with torch.no_grad():
            # Multi-step forecasting on validation set
            forecast_horizon = len(val_series)
            val_predictions = model.predict(train_tensor, steps=forecast_horizon)
            if val_predictions.dim() > 1:
                val_predictions = val_predictions.squeeze()
            val_mse = nn.MSELoss()(val_predictions, val_tensor).item()
            val_mae = torch.mean(torch.abs(val_predictions - val_tensor)).item()
            val_rmse = np.sqrt(val_mse)


        print(f"\n Validation Forecasting Metrics:")
        print(f"   MSE:  {val_mse:.6f}")
        print(f"   MAE:  {val_mae:.6f}")
        print(f"   RMSE: {val_rmse:.6f}")

        result = {
            'ar_order': ar_order,
            'kernel_size': config['kernel_size'],
            'num_filters': config.get('num_filters'),
            'num_layers': config.get('num_layers'),
            'num_channels': str(num_channels),
            'dropout': dropout,
            'lr': config['lr'],
            'val_mse': val_mse,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'training_loss': loss.item()
        }
        results.append(result)

    #convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_mse')

    print(f"\n{'='*60}")
    print(f" ADDITIVE HYBRID AR({ar_order}) CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    #Find the best model
    best_config = results_df.iloc[0].to_dict()

    print(f"\n{'='*60}")
    print(" BEST CONFIGURATION:")
    print(f"{'='*60}")
    for key, value in best_config.items():
        if key not in ['val_mse', 'val_mae', 'val_rmse', 'val_mape', 'training_loss']:
            print(f"   {key}: {value}")
    print(f"\n   Validation MSE:  {best_config['val_mse']:.6f}")
    print(f"   Validation MAE:  {best_config['val_mae']:.6f}")
    print(f"   Validation RMSE: {best_config['val_rmse']:.6f}")

    return best_config, results_df


def cross_validate_classic_tcn(train_series, val_series, hyperparameter_grid, num_epochs=100, seed=42):
    """
    Cross-validation for Classic TCN based on forecasting erros, to prevent overfitting

    Args:
        train_series: Training TimeSeries
        val_series: Validation TimeSeries
        hyperparameter_grid: Dictionary of hyperparameters to search
        num_epochs: Training epochs for each configuration
        seed: Random seed

    Returns:
        best_config: Best hyperparameter configuration
        results_df: DataFrame with all results
    """
    train_tensor = _ensure_tensor(train_series)
    val_tensor = _ensure_tensor(val_series)
    results = []

    #generate combinations
    from itertools import product
    keys = hyperparameter_grid.keys()
    values = hyperparameter_grid.values()

    for config_values in product(*values):
        config = dict(zip(keys, config_values))

        print(f"\n{'='*60}")
        print(f"Testing configuration: {config}")
        print(f"{'='*60}")

        #set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Prepare model configuration
        num_channels = config.get('num_channels')
        if num_channels is None and 'num_filters' in config and 'num_layers' in config:
            num_channels = [config['num_filters']] * config['num_layers']
        
        dropout = config.get('dropout', 0.1)

        #strain model with this configuration
        model = Classic_TCN(
            kernel_size=config['kernel_size'],
            num_channels=num_channels,
            dropout=dropout
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            predictions, targets = model(train_tensor)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.6f}')

        #evaluate on the validation set
        model.eval()
        with torch.no_grad():
            #multi-step forecasting on validation set
            forecast_horizon = len(val_series)
            val_predictions = model.predict(train_tensor, steps=forecast_horizon)
            if val_predictions.dim() > 1:
                val_predictions = val_predictions.squeeze()
            #compute forecasting errors
            val_mse = nn.MSELoss()(val_predictions, val_tensor).item()
            val_mae = torch.mean(torch.abs(val_predictions - val_tensor)).item()
            val_rmse = np.sqrt(val_mse)

        print(f"\n Validation Forecasting Metrics:")
        print(f"   MSE:  {val_mse:.6f}")
        print(f"   MAE:  {val_mae:.6f}")
        print(f"   RMSE: {val_rmse:.6f}")


        # Store results
        result = {
            'kernel_size': config['kernel_size'],
            'num_filters': config.get('num_filters'),
            'num_layers': config.get('num_layers'),
            'num_channels': str(num_channels),
            'dropout': dropout,
            'lr': config['lr'],
            'val_mse': val_mse,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'training_loss': loss.item()
        }
        results.append(result)

    #convert to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_mse')

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS (sorted by validation MSE)")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    #Show which is the optimal model
    best_config = results_df.iloc[0].to_dict()

    print(f"\n{'='*60}")
    print("Best model:")
    print(f"{'='*60}")
    for key, value in best_config.items():
        if key not in ['val_mse', 'val_mae', 'val_rmse', 'val_mape', 'training_loss']:
            print(f"   {key}: {value}")
    print(f"\n   Validation MSE:  {best_config['val_mse']:.6f}")
    print(f"   Validation MAE:  {best_config['val_mae']:.6f}")
    print(f"   Validation RMSE: {best_config['val_rmse']:.6f}")

    return best_config, results_df


class DartsBridge:
    def __init__(self, torch_model):
        self.torch_model = torch_model
    
    def fit(self, X, y):
        # We assume the model is already trained, so we do nothing here
        pass

    def predict(self, X):
        # Darts sends X as a 2D numpy array (samples, lags)
        # We convert to Torch, predict 1-step, and return numpy
        x_tensor = torch.from_numpy(X).float()
        self.torch_model.eval()
        with torch.no_grad():
            # Using your model's existing .predict() method
            # We predict 1 step because Darts handles the recursion for us
            preds = self.torch_model.predict(x_tensor, steps=1)
        return preds.numpy().reshape(-1, 1)