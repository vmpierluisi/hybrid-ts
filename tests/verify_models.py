import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Classic_TCN

def test_flexible_tcn():
    print("Testing Flexible TCN (per-layer config)...")
    # 2 layers: 
    # Layer 1: kernel 2, dilation 1
    # Layer 2: kernel 3, dilation 2
    model = Classic_TCN(
        num_inputs=1, 
        num_channels=[64, 64], 
        kernel_size=[2, 3], 
        dilations=[1, 2]
    )
    x = torch.randn(10, 50, 1)
    preds, targets = model(x)
    print(f"Preds shape: {preds.shape}, Targets shape: {targets.shape}")
    assert preds.shape == targets.shape
    
    forecast = model.predict(x, steps=5)
    print(f"Forecast shape: {forecast.shape}")
    assert forecast.shape == (10, 5)
    print("Flexible TCN passed.\n")

if __name__ == "__main__":
    try:
        test_flexible_tcn()
        print("All tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
