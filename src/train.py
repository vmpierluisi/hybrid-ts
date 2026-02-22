import torch
import torch.nn as nn
import numpy as np
import random
from itertools import product

import warnings
import pandas as pd
from src.models import Classic_TCN
from src.models import AdditiveHybrid_ARMA_TCN
from src.models import MultiplicativeHybrid_ARMA_TCN

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


class Trainer:
    """
    Unified trainer for TCN-based models
    """

    def __init__(
            self,
            model_type='classic',
            ar_order=1,
            kernel_size=3,
            num_channels=None,
            dilations=None,
            num_epochs=100,
            lr=0.001,
            seed=42,
            verbose=True
    ):
        """
        Args:
            model_type: 'classic', 'additive', or 'multiplicative'
            ar_order: AR order for hybrid models (ignored for classic)
            kernel_size: TCN kernel size
            num_channels: List of channel sizes per layer
            dilations: Dilation factors (if None, uses exponential)
            num_epochs: Training epochs
            lr: Learning rate
            seed: Random seed
            verbose: Print training progress
        """
        self.model_type = model_type.lower()
        self.ar_order = ar_order
        self.kernel_size = kernel_size
        self.num_channels = num_channels or ([64] * 5 if model_type == 'classic' else [64] * 3)
        self.dilations = dilations
        self.num_epochs = num_epochs
        self.lr = lr
        self.seed = seed
        self.verbose = verbose

        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.history = {'loss': []}

    def _set_seed(self):
        """Set random seeds for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _initialize_model(self):
        """Initialize model based on type"""
        if self.model_type == 'classic':
            self.model = Classic_TCN(
                num_channels=self.num_channels,
                kernel_size=self.kernel_size,
                dilations=self.dilations
            )
        elif self.model_type == 'additive':
            self.model = AdditiveHybrid_ARMA_TCN(
                ar_order=self.ar_order,
                kernel_size=self.kernel_size,
                num_channels=self.num_channels,
                dilations=self.dilations
            )
        elif self.model_type == 'multiplicative':
            self.model = MultiplicativeHybrid_ARMA_TCN(
                ar_order=self.ar_order,
                kernel_size=self.kernel_size,
                num_channels=self.num_channels,
                dilations=self.dilations
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _get_model_name(self):
        """Get display name for logging"""
        if self.model_type == 'classic':
            return 'Classic TCN'
        elif self.model_type == 'additive':
            return f'Additive Hybrid AR({self.ar_order}) + TCN'
        else:
            return f'Multiplicative Hybrid AR({self.ar_order}) + TCN'

    def fit(self, y_train):
        """
        Train the model

        Args:
            y_train: Training data (numpy array or torch tensor)

        Returns:
            self: Trained trainer instance
        """
        self._set_seed()
        y_train = _ensure_tensor(y_train)
        self._initialize_model()

        model_name = self._get_model_name()

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            predictions, targets = self.model(y_train)
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()

            # Track history
            self.history['loss'].append(loss.item())

            # Logging
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f'{model_name} - Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.6f}')

        return self

    def get_model(self):
        """Return the trained model"""
        return self.model

    def get_history(self):
        """Return training history"""
        return self.history


class CrossValidator:
    """
    Cross-validator for TCN-based models with hyperparameter search
    Supports AR/MA order search for hybrid models
    """

    def __init__(
            self,
            model_type='classic',
            ar_orders=None,
            ma_orders=None,
            hyperparameter_grid=None,
            num_epochs=100,
            patience=10,
            seed=42,
            verbose=True
    ):
        """
        Args:
            model_type: 'classic', 'additive', or 'multiplicative'
            ar_orders: List of AR orders to test for hybrid models (e.g., [1, 3, 5])
                      If None for hybrid models, defaults to [1]
            ma_orders: List of MA orders to test for hybrid models (e.g., [0, 1, 3])
                      If None, MA component is not used
            hyperparameter_grid: Dict of hyperparameters to search
                Example: {
                    'kernel_size': [3, 5],
                    'num_filters': [32, 64],
                    'num_layers': [3, 5],
                    'dropout': [0.1, 0.2],
                    'lr': [0.001, 0.0001]
                }
            num_epochs: Training epochs per configuration
            seed: Random seed
            verbose: Print progress
        """

        self.model_type = model_type.lower()

        # Handle AR/MA orders for hybrid models
        if self.model_type in ['additive', 'multiplicative']:
            self.ar_orders = ar_orders if ar_orders is not None else [1]
            self.ma_orders = ma_orders if ma_orders is not None else [0]
        else:
            self.ar_orders = None
            self.ma_orders = None

        self.hyperparameter_grid = hyperparameter_grid or {}
        self.num_epochs = num_epochs
        self.seed = seed
        self.verbose = verbose
        self.patience = patience

        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
             else 'mps' if torch.backends.mps.is_available()
             else 'cpu'
            )

        if self.verbose:
            print(f"Using device: {self.device}")

        self.best_config = None
        self.results_df = None

    def _set_seed(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _prepare_num_channels(self, config):
        """Convert num_filters + num_layers to num_channels list"""
        num_channels = config.get('num_channels')
        if num_channels is None and 'num_filters' in config and 'num_layers' in config:
            num_channels = [config['num_filters']] * config['num_layers']
        return num_channels

    def _initialize_model(self, config, ar_order=None, ma_order=None):
        """Initialize model based on type and config"""
        num_channels = self._prepare_num_channels(config)
        dropout = config.get('dropout', 0.1)

        if self.model_type == 'classic':
            return Classic_TCN(
                kernel_size=config['kernel_size'],
                num_channels=num_channels,
                dropout=dropout
            )
        elif self.model_type == 'additive':
            model_kwargs = {
                'ar_order': ar_order,
                'kernel_size': config['kernel_size'],
                'num_channels': num_channels,
                'dropout': dropout
            }
            # Add MA order if specified
            if ma_order is not None and ma_order > 0:
                model_kwargs['ma_order'] = ma_order
            return AdditiveHybrid_ARMA_TCN(**model_kwargs)

        elif self.model_type == 'multiplicative':
            model_kwargs = {
                'ar_order': ar_order,
                'kernel_size': config['kernel_size'],
                'num_channels': num_channels,
                'dropout': dropout
            }
            # Add MA order if specified
            if ma_order is not None and ma_order > 0:
                model_kwargs['ma_order'] = ma_order
            return MultiplicativeHybrid_ARMA_TCN(**model_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _get_model_name(self, ar_order=None, ma_order=None):
        """Get display name for logging"""
        if self.model_type == 'classic':
            return 'Classic TCN'
        elif self.model_type == 'additive':
            if ma_order is not None and ma_order > 0:
                return f'Additive Hybrid ARMA({ar_order},{ma_order})'
            return f'Additive Hybrid AR({ar_order})'
        else:
            if ma_order is not None and ma_order > 0:
                return f'Multiplicative Hybrid ARMA({ar_order},{ma_order})'
            return f'Multiplicative Hybrid AR({ar_order})'

    def _train_single_config(self, config, train_tensor, ar_order=None, ma_order=None):
        """Train model with a single configuration"""
        self._set_seed()

        train_tensor = train_tensor.to(self.device)
        model = self._initialize_model(config, ar_order, ma_order)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0
        final_loss = None

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            predictions, targets = model(train_tensor)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                patience = 0
            else:
                patience_counter += 1
            final_loss = current_loss

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Training Loss: {loss.item():.6f}')

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f'Early stopping after {patience_counter} epochs')
                break
        return model, loss.item()

    def _evaluate_on_validation(self, model, train_tensor, val_tensor):
        """Evaluate model on validation set with multi-step forecasting"""
        model.eval()
        train_tensor = train_tensor.to(self.device)
        val_tensor = val_tensor.to(self.device)
        with torch.no_grad():
            forecast_horizon = len(val_tensor)
            val_predictions = model.predict(train_tensor, steps=forecast_horizon)

            if val_predictions.dim() > 1:
                val_predictions = val_predictions.squeeze()

            val_mse = nn.MSELoss()(val_predictions, val_tensor).item()
            val_mae = torch.mean(torch.abs(val_predictions - val_tensor)).item()
            val_rmse = np.sqrt(val_mse)

        return val_mse, val_mae, val_rmse

    def _build_result_dict(self, config, val_metrics, training_loss, ar_order=None, ma_order=None):
        """Build result dictionary from config and metrics"""
        num_channels = self._prepare_num_channels(config)

        result = {
            'kernel_size': config['kernel_size'],
            'num_filters': config.get('num_filters'),
            'num_layers': config.get('num_layers'),
            'num_channels': str(num_channels),
            'dropout': config.get('dropout', 0.1),
            'lr': config['lr'],
            'val_mse': val_metrics[0],
            'val_mae': val_metrics[1],
            'val_rmse': val_metrics[2],
            'training_loss': training_loss
        }

        # Add AR/MA orders for hybrid models
        if self.model_type in ['additive', 'multiplicative']:
            result['ar_order'] = ar_order
            if ma_order is not None and ma_order > 0:
                result['ma_order'] = ma_order
            else:
                result['ma_order'] = 0

        return result

    def fit(self, train_series, val_series):
        """
        Perform cross-validation across hyperparameter grid

        Args:
            train_series: Training TimeSeries
            val_series: Validation TimeSeries

        Returns:
            self: Fitted cross-validator with results
        """
        train_tensor = _ensure_tensor(train_series)
        val_tensor = _ensure_tensor(val_series)
        results = []

        # Generate all hyperparameter combinations
        keys = list(self.hyperparameter_grid.keys())
        values = list(self.hyperparameter_grid.values())

        # For hybrid models, iterate over AR and MA orders as well
        if self.model_type in ['additive', 'multiplicative']:
            # Create cartesian product of AR orders, MA orders, and other hyperparameters
            for ar_order in self.ar_orders:
                for ma_order in self.ma_orders:
                    model_name = self._get_model_name(ar_order, ma_order)

                    for config_values in product(*values):
                        config = dict(zip(keys, config_values))

                        if self.verbose:
                            print(f"\n{'=' * 60}")
                            print(f"Testing {model_name} configuration: {config}")
                            print(f"{'=' * 60}")

                        # Train model
                        model, training_loss = self._train_single_config(
                            config, train_tensor, ar_order, ma_order
                        )

                        # Evaluate on validation
                        val_mse, val_mae, val_rmse = self._evaluate_on_validation(
                            model, train_tensor, val_tensor
                        )

                        if self.verbose:
                            print(f"\n Validation Forecasting Metrics:")
                            print(f"   MSE:  {val_mse:.6f}")
                            print(f"   MAE:  {val_mae:.6f}")
                            print(f"   RMSE: {val_rmse:.6f}")

                        # Store results
                        result = self._build_result_dict(
                            config,
                            (val_mse, val_mae, val_rmse),
                            training_loss,
                            ar_order,
                            ma_order
                        )
                        results.append(result)
        else:
            # Classic TCN: no AR/MA orders
            for config_values in product(*values):
                config = dict(zip(keys, config_values))

                if self.verbose:
                    print(f"\n{'=' * 60}")
                    print(f"Testing Classic TCN configuration: {config}")
                    print(f"{'=' * 60}")

                # Train model
                model, training_loss = self._train_single_config(config, train_tensor)

                # Evaluate on validation
                val_mse, val_mae, val_rmse = self._evaluate_on_validation(
                    model, train_tensor, val_tensor
                )

                if self.verbose:
                    print(f"\n Validation Forecasting Metrics:")
                    print(f"   MSE:  {val_mse:.6f}")
                    print(f"   MAE:  {val_mae:.6f}")
                    print(f"   RMSE: {val_rmse:.6f}")

                # Store results
                result = self._build_result_dict(
                    config,
                    (val_mse, val_mae, val_rmse),
                    training_loss
                )
                results.append(result)

        # Convert to DataFrame and sort by validation MSE
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('val_mse')

        # Get best configuration
        self.best_config = self.results_df.iloc[0].to_dict()

        if self.verbose:
            self._print_summary()

        return self

    def _print_summary(self):
        """Print cross-validation summary"""
        if self.model_type in ['additive', 'multiplicative']:
            ar_order = self.best_config.get('ar_order')
            ma_order = self.best_config.get('ma_order', 0)
            model_name = self._get_model_name(ar_order, ma_order)
        else:
            model_name = self._get_model_name()

        print(f"\n{'=' * 60}")
        print(f" {model_name.upper()} CROSS-VALIDATION RESULTS")
        print(f"{'=' * 60}")
        print(self.results_df.to_string(index=False))

        print(f"\n{'=' * 60}")
        print(" BEST CONFIGURATION:")
        print(f"{'=' * 60}")
        for key, value in self.best_config.items():
            if key not in ['val_mse', 'val_mae', 'val_rmse', 'val_mape', 'training_loss']:
                print(f"   {key}: {value}")
        print(f"\n   Validation MSE:  {self.best_config['val_mse']:.6f}")
        print(f"   Validation MAE:  {self.best_config['val_mae']:.6f}")
        print(f"   Validation RMSE: {self.best_config['val_rmse']:.6f}")

    def get_best_config(self):
        """Return best hyperparameter configuration"""
        return self.best_config

    def get_results(self):
        """Return full results DataFrame"""
        return self.results_df

    def get_top_n_configs(self, n=5):
        """Return top N configurations by validation MSE"""
        return self.results_df.head(n)

    def get_best_ar_ma_orders(self):
        """
        For hybrid models, return the best AR and MA orders

        Returns:
            tuple: (best_ar_order, best_ma_order) or None for classic TCN
        """
        if self.model_type in ['additive', 'multiplicative'] and self.best_config:
            return (
                self.best_config.get('ar_order'),
                self.best_config.get('ma_order', 0)
            )
        return None

    def compare_ar_ma_combinations(self):
        """
        For hybrid models, analyze performance across different AR/MA combinations

        Returns:
            DataFrame: Aggregated results grouped by AR and MA orders
        """
        if self.model_type not in ['additive', 'multiplicative']:
            print("This method is only available for hybrid models.")
            return None

        # Group by AR and MA orders and get mean metrics
        comparison = self.results_df.groupby(['ar_order', 'ma_order']).agg({
            'val_mse': ['mean', 'min', 'std'],
            'val_mae': ['mean', 'min', 'std'],
            'val_rmse': ['mean', 'min', 'std']
        }).round(6)

        return comparison

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