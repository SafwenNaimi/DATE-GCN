import argparse
import os
import importlib.util
import torch
import numpy as np
from pyskl.models import build_model

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    USE_FVCORE = True
except ImportError:
    USE_FVCORE = False
    print("Warning: fvcore not installed, FLOPs counting unavailable")

def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 17, 32, 56, 56],  # Default shape for MSG3D
        help='input shape (N,M,T,V,C)')
    args = parser.parse_args()
    return args

def load_config(config_path):
    """Load Python config file without mmengine.Config"""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return {k: v for k, v in config.__dict__.items() if not k.startswith('__')}

def create_dummy_input(input_shape, model_type='gcn'):
    """Create appropriate dummy input based on model type"""
    if model_type == 'gcn':
        # For GCN-based models (N, M, T, V, C)
        return torch.randn(input_shape).cuda()
    else:
        # For CNN-based models (N, C, T, H, W)
        return torch.randn(input_shape).cuda()

def main():
    args = parse_args()
    input_shape = tuple(args.shape)
    
    # Load config
    cfg = load_config(args.config)
    
    # Build model
    model = build_model(cfg['model'])
    model.eval()
    model.cuda()

    # Create appropriate dummy input for MSG3D
    dummy_input = create_dummy_input(input_shape, model_type='gcn')
    
    # For models requiring labels, create dummy label
    num_classes = cfg['model'].get('num_classes', 60)  # Default to NTU60 classes
    dummy_label = torch.zeros(1, dtype=torch.long).cuda()

    if USE_FVCORE:
        try:
            # First try standard forward
            flops = FlopCountAnalysis(model, dummy_input)
            flops.total()  # Test if it works
        except (ValueError, RuntimeError) as e:
            if "Label" in str(e):
                # Wrap model to handle label requirement
                class WrappedModel(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, x):
                        return self.model(x, dummy_label)
                
                model = WrappedModel(model)
                flops = FlopCountAnalysis(model, dummy_input)
            else:
                raise e

        # Calculate metrics
        total_flops = flops.total() / 1e9  # GFLOPs
        params = parameter_count(model)[''] / 1e6  # MParams
        
        # Print results
        print('\n' + '=' * 60)
        print(f"Model: {cfg['model']['type']}")
        print(f"Input shape: {input_shape}")
        print(f"FLOPs: {total_flops:.2f} GFLOPs")
        print(f"Params: {params:.2f} M")
        print('=' * 60)
    else:
        # Fallback: only parameter count
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'\nParams: {params:.2f} M (install fvcore for FLOPs calculation)')

if __name__ == '__main__':
    main()