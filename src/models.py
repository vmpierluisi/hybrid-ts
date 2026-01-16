import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pathlib
import os
import warnings
warnings.filterwarnings('ignore')

class Classic_TCN(nn.Module):
    """
    Classic TCN (Bai et al., 2028)

    """

    def __init__(self, kernel_size=3, num_filters=64, num_layers=3, dilation_base=2):
        super().__init__()

        # TCN for nonlinear dynamics
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation
            in_channels = 1 if i == 0 else num_filters

            conv = weight_norm(nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            ))
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        layers.append(nn.Conv1d(num_filters, 1, kernel_size=1))
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 2:
            x = x.unsqueeze(2)

        batch_size, seq_len, _ = x.shape

        # TCN processes raw time series directly
        x_transposed = x[:, :-1, :].transpose(1, 2)  # Input: all but last timestep
        tcn_output = self.tcn(x_transposed)
        predictions = tcn_output.transpose(1, 2)
        
        targets = x[:, 1:, :]  # Targets: all but first timestep

        # Align lengths (similar to hybrid model)
        target_len = targets.shape[1]
        pred_len = predictions.shape[1]
        if pred_len > target_len:
            predictions = predictions[:, :target_len, :]
        elif pred_len < target_len:
            padding = target_len - pred_len
            predictions = torch.cat([
                torch.zeros(batch_size, padding, 1, device=predictions.device),
                predictions
            ], dim=1)
        
        return predictions, targets

    def predict(self, x, steps=1):
        """
        Multi-step ahead prediction using the Classic TCN model.
        
        Args:
            x: Input sequence tensor of shape (seq_len,) or (batch_size, seq_len) or (batch_size, seq_len, 1)
            steps: Number of steps to predict ahead
        
        Returns:
            predictions: Tensor of shape (steps,) for single sequence or (batch_size, steps) for batch
        
        The model recursively predicts:
            $\hat{y}_{t+h} = \text{TCN}(y_{t-k:t+h-1})$
        """
        self.eval()
        with torch.no_grad():
            # Normalize input dimensions
            if x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(2)
                single_sequence = True
            elif x.dim() == 2:
                x = x.unsqueeze(2)
                single_sequence = False
            else:
                single_sequence = False
            
            batch_size = x.shape[0]
            
            # Initialize sequence with input
            sequence = x.clone()
            predictions = []
            
            for step in range(steps):
                # TCN prediction on entire sequence so far
                x_transposed = sequence.transpose(1, 2)
                tcn_output = self.tcn(x_transposed)
                
                # Take the last prediction
                next_pred = tcn_output[:, :, -1:].transpose(1, 2)  # Shape: (batch_size, 1, 1)
                
                predictions.append(next_pred)
                
                # Append prediction to sequence for next iteration
                sequence = torch.cat([sequence, next_pred], dim=1)
            
            # Stack predictions
            predictions = torch.cat(predictions, dim=1)
            
            if single_sequence:
                return predictions.squeeze()
            else:
                return predictions.squeeze(2)


#Flexible AR and TCN Additive Hybrid Model
class AdditiveHybrid_AR_TCN(nn.Module):
    """
    Additive Hybrid Model with flexible AR order: y_t = L_t + N_t (following Wang et al. 2013)

    1. AR(p) predicts linear component: L_hat_t = φ_1*y_{t-1} + ... + φ_p*y_{t-p} + c
    2. Additive residuals: e_t = y_t - L_hat_t
    3. TCN models nonlinear pattern: N_hat_t = TCN(e_{t-k:t-1})
    4. Final prediction: y_hat_t = L_hat_t + N_hat_t
    """

    def __init__(self, ar_order=1, kernel_size=3, num_filters=64, num_layers=3, dilation_base=2):
        super().__init__()

        self.ar_order = ar_order

        # AR(p) component - multiple weights for different lags
        self.ar_weights = nn.Parameter(torch.zeros(ar_order))
        self.ar_bias = nn.Parameter(torch.tensor([0.0]))

        # TCN for nonlinear residuals
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation
            in_channels = 1 if i == 0 else num_filters

            conv = weight_norm(nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            ))
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        layers.append(nn.Conv1d(num_filters, 1, kernel_size=1))
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 2:
            x = x.unsqueeze(2)

        batch_size, seq_len, _ = x.shape

        # Step 1: AR(p) predictions (linear component)
        # Need at least ar_order + 1 timesteps
        if seq_len <= self.ar_order:
            raise ValueError(f"Sequence length {seq_len} must be > AR order {self.ar_order}")

        # Compute AR predictions: sum of weighted past values
        ar_predictions = torch.zeros(batch_size, seq_len - self.ar_order, 1, device=x.device)
        for i in range(self.ar_order):
            # ar_weights[i] applies to lag (i+1)
            ar_predictions += self.ar_weights[i] * x[:, self.ar_order-1-i:seq_len-1-i, :]
        ar_predictions += self.ar_bias

        # Step 2: ADDITIVE residuals: e_t = y_t - L_hat_t
        targets = x[:, self.ar_order:, :]
        additive_residuals = targets - ar_predictions

        # Step 3: TCN models residuals
        residuals_transposed = additive_residuals.transpose(1, 2)
        tcn_output = self.tcn(residuals_transposed)
        tcn_predictions = tcn_output.transpose(1, 2)

        # Align lengths
        residual_len = ar_predictions.shape[1]
        tcn_len = tcn_predictions.shape[1]
        if tcn_len > residual_len:
            tcn_predictions = tcn_predictions[:, :residual_len, :]
        elif tcn_len < residual_len:
            padding = residual_len - tcn_len
            tcn_predictions = torch.cat([
                torch.zeros(batch_size, padding, 1, device=tcn_predictions.device),
                tcn_predictions
            ], dim=1)

        # Step 4: ADDITIVE combination: y_hat = L_hat + N_hat
        final_predictions = ar_predictions + tcn_predictions

        return final_predictions, targets

    def predict(self, x, steps=1):
        """
        Multi-step ahead prediction using the additive hybrid model.
        
        Args:
            x: Input sequence tensor of shape (seq_len,) or (batch_size, seq_len) or (batch_size, seq_len, 1)
            steps: Number of steps to predict ahead
        
        Returns:
            predictions: Tensor of shape (steps,) for single sequence or (batch_size, steps) for batch
        
        The model recursively predicts:
            $\hat{y}_{t+h} = \hat{L}_{t+h} + \hat{N}_{t+h}$
        
        where:
            $\hat{L}_{t+h} = \sum_{i=1}^{p} \phi_i y_{t+h-i} + c$
            $\hat{N}_{t+h} = \text{TCN}(e_{t+h-k:t+h-1})$
        """
        self.eval()
        with torch.no_grad():
            # Normalize input dimensions
            if x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(2)
                single_sequence = True
            elif x.dim() == 2:
                x = x.unsqueeze(2)
                single_sequence = False
            else:
                single_sequence = False
            
            batch_size = x.shape[0]
            
            # Initialize sequence with input
            sequence = x.clone()
            predictions = []
            
            for step in range(steps):
                seq_len = sequence.shape[1]
                
                # Step 1: Compute AR(p) prediction for next timestep
                ar_pred = self.ar_bias.expand(batch_size, 1)
                for i in range(self.ar_order):
                    lag_idx = seq_len - 1 - i
                    if lag_idx >= 0:
                        ar_pred += self.ar_weights[i] * sequence[:, lag_idx, :]
                
                # Step 2: Compute residuals for TCN
                # Get recent AR predictions and compute residuals
                if seq_len > self.ar_order:
                    recent_ar = torch.zeros(batch_size, seq_len - self.ar_order, 1, device=x.device)
                    for i in range(self.ar_order):
                        recent_ar += self.ar_weights[i] * sequence[:, self.ar_order-1-i:seq_len-1-i, :]
                    recent_ar += self.ar_bias
                    
                    recent_targets = sequence[:, self.ar_order:, :]
                    residuals = recent_targets - recent_ar
                else:
                    # Not enough history for residuals
                    residuals = torch.zeros(batch_size, 0, 1, device=x.device)
                
                # Step 3: TCN prediction on residuals
                if residuals.shape[1] > 0:
                    residuals_transposed = residuals.transpose(1, 2)
                    tcn_output = self.tcn(residuals_transposed)
                    tcn_pred = tcn_output[:, :, -1:].transpose(1, 2)
                else:
                    tcn_pred = torch.zeros(batch_size, 1, 1, device=x.device)
                
                # Step 4: Additive combination
                next_pred = ar_pred + tcn_pred
                predictions.append(next_pred)
                
                # Append prediction to sequence for next iteration
                # next_pred is already (batch_size, 1, 1), so just concatenate directly
                sequence = torch.cat([sequence, next_pred], dim=1)
            
            # Stack predictions
            predictions = torch.cat(predictions, dim=1)
            
            if single_sequence:
                return predictions.squeeze()
            else:
                return predictions.squeeze(2)

#Flexible AR and TCN Multiplicative Hybrid Model
class MultiplicativeHybrid_AR_TCN(nn.Module):
    """
    Multiplicative Hybrid Model with flexible AR order: y_t = L_t × N_t (following Wang et al. 2013)

    1. AR(p) predicts linear component: L_hat_t = φ_1*y_{t-1} + ... + φ_p*y_{t-p} + c
    2. MULTIPLICATIVE residuals: e_t = y_t / L_hat_t  ← KEY DIFFERENCE!
    3. TCN models nonlinear pattern: N_hat_t = TCN(e_{t-k:t-1})
    4. Final prediction: y_hat_t = L_hat_t × N_hat_t

    Note: For financial returns that cross zero, we use safe division.
    """

    def __init__(self, ar_order=1, kernel_size=3, num_filters=64, num_layers=3, dilation_base=2, epsilon=1e-6):
        super().__init__()

        self.ar_order = ar_order
        self.epsilon = epsilon

        # AR(p) component - multiple weights for different lags
        self.ar_weights = nn.Parameter(torch.zeros(ar_order))
        self.ar_bias = nn.Parameter(torch.tensor([0.0]))

        # TCN for multiplicative factor
        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation
            in_channels = 1 if i == 0 else num_filters

            conv = weight_norm(nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            ))
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        layers.append(nn.Conv1d(num_filters, 1, kernel_size=1))
        self.tcn = nn.Sequential(*layers)

        # Bias to center output around 1 (multiplicative identity)
        self.tcn_bias = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 2:
            x = x.unsqueeze(2)

        batch_size, seq_len, _ = x.shape

        # Step 1: AR(p) predictions (linear component)
        # Need at least ar_order + 1 timesteps
        if seq_len <= self.ar_order:
            raise ValueError(f"Sequence length {seq_len} must be > AR order {self.ar_order}")

        # Compute AR predictions: sum of weighted past values
        ar_predictions = torch.zeros(batch_size, seq_len - self.ar_order, 1, device=x.device)
        for i in range(self.ar_order):
            # ar_weights[i] applies to lag (i+1)
            ar_predictions += self.ar_weights[i] * x[:, self.ar_order-1-i:seq_len-1-i, :]
        ar_predictions += self.ar_bias

        targets = x[:, self.ar_order:, :]

        # Step 2: MULTIPLICATIVE residuals: e_t = y_t / L_hat_t
        # Safe division to handle near-zero values
        safe_ar = torch.where(
            torch.abs(ar_predictions) < self.epsilon,
            torch.sign(ar_predictions) * self.epsilon + self.epsilon,
            ar_predictions
        )
        multiplicative_residuals = targets / safe_ar

        # Step 3: TCN models the multiplicative factor
        residuals_transposed = multiplicative_residuals.transpose(1, 2)
        tcn_output = self.tcn(residuals_transposed)
        tcn_predictions = tcn_output.transpose(1, 2) + self.tcn_bias

        # Align lengths
        residual_len = ar_predictions.shape[1]
        tcn_len = tcn_predictions.shape[1]
        if tcn_len > residual_len:
            tcn_predictions = tcn_predictions[:, :residual_len, :]
        elif tcn_len < residual_len:
            padding = residual_len - tcn_len
            tcn_predictions = torch.cat([
                torch.ones(batch_size, padding, 1, device=tcn_predictions.device),
                tcn_predictions
            ], dim=1)

        # Step 4: MULTIPLICATIVE combination: y_hat = L_hat × N_hat
        final_predictions = ar_predictions * tcn_predictions

        return final_predictions, targets