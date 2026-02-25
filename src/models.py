import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import warnings

warnings.filterwarnings('ignore')

class ChausalConv1d(nn.Module):
    """
    1D Convolution with causal padding.
    Padding is added only to the left side of the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        ))
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv.weight, 0, 0.01)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = self.conv(x)
        # Remove the extra padding on the right
        return x[:, :, :-self.padding].contiguous()


class TemporalBlock(nn.Module):
    """
    Residual block containing two causal convolutional layers.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = ChausalConv1d(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = ChausalConv1d(n_outputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
        self.downsample = weight_norm(nn.Conv1d(n_inputs, n_outputs, 1)) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Stacked Temporal Blocks forming a TCN.
    Can accept a single kernel_size/dilation (applied to all layers)
    or a list for per-layer configuration.
    """
    def __init__(self, num_inputs, num_channels, kernel_sizes=2, dilations=None, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_levels
        if dilations is None:
            dilations = [2**i for i in range(num_levels)]
        if isinstance(dilations, int):
            dilations = [dilations**i for i in range(num_levels)]

        for i in range(num_levels):
            dilation_size = dilations[i]
            k_size = kernel_sizes[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, k_size, stride=1, dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TimeSeriesModel(nn.Module):
    """
    Base class for time series models with common utilities.
    """
    def _prepare_input(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 2:
            x = x.unsqueeze(2)
        return x

    def predict(self, x, steps=1):
        """
        Generic multi-step recursive prediction.
        """
        self.eval()
        with torch.no_grad():
            x = self._prepare_input(x)
            batch_size = x.shape[0]
            sequence = x.clone()
            predictions = []

            for _ in range(steps):
                pred, _ = self.forward(sequence)
                next_val = pred[:, -1:, :]
                predictions.append(next_val)
                sequence = torch.cat([sequence, next_val], dim=1)

            predictions = torch.cat(predictions, dim=1)
            return predictions.squeeze(2) if batch_size > 1 else predictions.squeeze()


class Classic_TCN(TimeSeriesModel):
    """
    Standard TCN for direct time series forecasting.
    Follows the architecture by Bai et al. (2018).
    """
    def __init__(self, num_inputs=1, num_channels=[64, 64, 64], kernel_size=3, dilations=None, dropout=0.1):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_sizes=kernel_size, dilations=dilations, dropout=dropout)
        self.linear = nn.Conv1d(num_channels[-1], 1, kernel_size=1)

    def forward(self, x):
        x = self._prepare_input(x)
        batch_size, seq_len, _ = x.shape
        
        # x_transposed: (batch, 1, seq_len)
        x_transposed = x.transpose(1, 2)
        
        # TCN output: (batch, num_channels[-1], seq_len)
        output = self.tcn(x_transposed)
        
        # predictions: (batch, 1, seq_len) -> (batch, seq_len, 1)
        predictions = self.linear(output).transpose(1, 2)
        
        # Align targets for next-step prediction
        # Input: x[t-k:t], Target: x[t-k+1:t+1]
        preds_aligned = predictions[:, :-1, :]
        targets_aligned = x[:, 1:, :]
        
        return preds_aligned, targets_aligned


class AdditiveHybrid_ARMA_TCN(TimeSeriesModel):
    """
    Additive Hybrid Model: y_t = L_t + N_t (following Wang et al. 2013)
    
    1. AR(p) predicts linear component: L_hat_t
    2. Additive residuals: e_t = y_t - L_hat_t
    3. TCN models nonlinear pattern: N_hat_t = TCN(e_{t-k:t-1})
    4. Final prediction: y_hat_t = L_hat_t + N_hat_t
    """
    def __init__(self, ar_order=1, ma_order=0, num_channels=[64, 64, 64], kernel_size=3, dilations=None, dropout=0.1):
        super().__init__()
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.ar_weights = nn.Parameter(torch.zeros(ar_order))
        self.ar_weights.data[0] = 1
        self.ar_bias = nn.Parameter(torch.zeros(1))
        if ma_order > 0:
            self.ma_weights = nn.Parameter(torch.zeros(ma_order))
        
        self.tcn = TemporalConvNet(1, num_channels, kernel_sizes=kernel_size, dilations=dilations, dropout=dropout)
        self.tcn_head = nn.Conv1d(num_channels[-1], 1, kernel_size=1)

    def _compute_ar(self, x):
        batch_size, seq_len, _ = x.shape
        ar_preds = torch.zeros(batch_size, seq_len - self.ar_order, 1, device=x.device)
        for i in range(self.ar_order):
            ar_preds += self.ar_weights[i] * x[:, self.ar_order-1-i:seq_len-1-i, :]
        ar_preds += self.ar_bias
        return ar_preds

    def _compute_ma(self, residuals):
        """Add weighted past residuals to MA predictions"""
        batch_size, seq_len, _ = residuals.shape
        ma_correction = torch.zeros(batch_size, seq_len, 1, device=residuals.device)
        for i in range(min(self.ma_order, seq_len)):
            ma_correction[:, i+1:, :] += self.ma_weights[i] * residuals[:, :seq_len-i-1, :]
        return ma_correction

    def forward(self, x):
        x = self._prepare_input(x)
        if x.shape[1] <= self.ar_order:
            # For predict() starting with small sequences
            return torch.zeros(x.shape[0], x.shape[1], 1, device=x.device), x

        # 1. AR predictions
        ar_predictions = self._compute_ar(x)

        # 2. Residuals
        targets = x[:, self.ar_order:, :]
        residuals = targets - ar_predictions

        if self.ma_order > 0:
            ma_correction = self._compute_ma(residuals)
            arma_predictions = ar_predictions + ma_correction
        else:
            arma_predictions = ar_predictions

        
        # 3. TCN on residuals (causally shifted)
        res_input = residuals[:, :-1, :].transpose(1, 2)
        if res_input.shape[2] > 0:
            tcn_out = self.tcn(res_input)
            tcn_preds = self.tcn_head(tcn_out).transpose(1, 2)
            zero_pad = torch.zeros(x.shape[0], 1, 1, device=x.device)
            tcn_preds = torch.cat([zero_pad, tcn_preds], dim=1)
        else:
            tcn_preds = torch.zeros(x.shape[0], ar_predictions.shape[1], 1, device=x.device)

        #targets_aligned = targets
        final_preds = arma_predictions + tcn_preds

        return final_preds, targets


class MultiplicativeHybrid_ARMA_TCN(TimeSeriesModel):
    """
    Multiplicative Hybrid Model: y_t = L_t × N_t (following Wang et al. 2013)

    1. AR(p) predicts linear component: L_hat_t
    2. Multiplicative residuals: e_t = y_t / L_hat_t
    3. TCN models nonlinear pattern: N_hat_t = TCN(e_{t-k:t-1})
    4. Final prediction: y_hat_t = L_hat_t × N_hat_t
    """
    def __init__(self, ar_order=1, ma_order=0, num_channels=[64, 64, 64], kernel_size=3, dilations=None, dropout=0.1, epsilon=1e-6):
        super().__init__()
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.epsilon = epsilon
        self.ar_weights = nn.Parameter(torch.zeros(ar_order))
        self.ar_weights.data[0] = 1
        self.ar_bias = nn.Parameter(torch.zeros(1))
        if ma_order > 0:
            self.ma_weights = nn.Parameter(torch.zeros(ma_order))
        
        self.tcn = TemporalConvNet(1, num_channels, kernel_sizes=kernel_size, dilations=dilations, dropout=dropout)
        self.tcn_head = nn.Conv1d(num_channels[-1], 1, kernel_size=1)
        self.tcn_bias = nn.Parameter(torch.ones(1))

    def _compute_ar(self, x):
        batch_size, seq_len, _ = x.shape
        ar_preds = torch.zeros(batch_size, seq_len - self.ar_order, 1, device=x.device)
        for i in range(self.ar_order):
            ar_preds += self.ar_weights[i] * x[:, self.ar_order-1-i:seq_len-1-i, :]
        ar_preds += self.ar_bias
        return ar_preds

    def _compute_ma(self, residuals):
        """Add weighted past residuals to MA predictions"""
        batch_size, seq_len, _ = residuals.shape
        ma_correction = torch.zeros(batch_size, seq_len, 1, device=residuals.device)
        for i in range(min(self.ma_order, seq_len)):
            ma_correction[:, i+1:, :] += self.ma_weights[i] * residuals[:, :seq_len-i-1, :]  # ADD
        return ma_correction

    def forward(self, x):
        x = self._prepare_input(x)
        if x.shape[1] <= self.ar_order:
            return torch.ones(x.shape[0], x.shape[1], 1, device=x.device), x

        # 1. AR predictions
        ar_predictions = self._compute_ar(x)
        targets = x[:, self.ar_order:, :]

        # 2. Multiplicative residuals
        safe_ar = torch.where(
            torch.abs(ar_predictions) < self.epsilon,
            torch.sign(ar_predictions) * self.epsilon + self.epsilon,
            ar_predictions
        )
        multi_residuals = targets / safe_ar

            # Apply MA multiplier
        if self.ma_order > 0:
            ma_mult = self._compute_ma(multi_residuals)
            arma_predictions = ar_predictions + ma_mult
        else:
            arma_predictions = ar_predictions

        # 3. TCN on residuals
        res_input = multi_residuals[:, :-1, :].transpose(1, 2)
        if res_input.shape[2] > 0:
            tcn_out = self.tcn(res_input)
            tcn_preds = self.tcn_head(tcn_out).transpose(1, 2) + self.tcn_bias
            zero_pad = torch.ones(x.shape[0], 1, 1, device=x.device)  # multiplicative identity
            tcn_preds = torch.cat([zero_pad, tcn_preds], dim=1)
        else:
            tcn_preds = torch.ones(x.shape[0], ar_predictions.shape[1], 1, device=x.device)

        #targets_aligned = targets
        final_preds = arma_predictions * tcn_preds

        return final_preds, targets