import requests
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from IPython.display import display
import torch
from PIL import Image
from contextlib import contextmanager
from typing import Callable, Union, Dict, Any
import os
import yaml
file_path = os.path.dirname(__file__)
config_file = os.path.join(file_path, 'config.yaml')
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

model_cache_dir = config['cache_dir']
if model_cache_dir is None:
    model_cache_dir = os.path.join(file_path, '..', 'models')


@contextmanager
def session_hook(model: torch.nn.Module, hook: Callable):
    handle = model.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


class HookedLVLM:
    """Hooked LVLM.
    """
    def __init__(self, 
                 model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 hook_loc: str = "text_model_in",
                 device: str = "cuda:0",
                 quantize: bool = False,
                 ):
        if quantize:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True,
                device_map=device,
                cache_dir=model_cache_dir,
                attn_implementation="eager"
                )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, 
                device_map=device,
                cache_dir=model_cache_dir,
                torch_dtype=torch.float16,
                )
        
        # processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.hook_loc = hook_loc 
        self.data = None
    
    @contextmanager
    def block_layer_mlp(self, layers_to_skip, device=None):
        """
        Context manager to skip the forward pass of specified layers' MLPs.
        The output of the previous non-skipped layer is forwarded through the skipped layers.
        
        Example: If layer 2 is normal, layer 3 is skipped, then layer 4 receives layer 2's output.

        :param layers_to_skip: list of layer indices to skip (e.g., [2] to skip layer 2)
        :param device: device for tensors
        """
        if not isinstance(layers_to_skip, (list, tuple)):
            layers_to_skip = [layers_to_skip]

        # We'll store outputs by layer index
        stored_outputs = {}

        hooks = []

        # First, register hooks on each layer to capture input/output or bypass computation
        for idx, layer in enumerate(self.model.language_model.layers):
            mlp = layer.mlp

            def create_hook(layer_idx):
                def hook_fn(module, args, output):
                    # Input to this layer
                    x = args[0]

                    # If this layer is to be skipped
                    if layer_idx in layers_to_skip:
                        # We need to find the last non-skipped layer before this one
                        prev_idx = layer_idx - 1
                        while prev_idx >= 0:
                            if prev_idx not in layers_to_skip:
                                break
                            prev_idx -= 1

                        if prev_idx >= 0 and prev_idx not in stored_outputs:
                            # In case earlier layers were also skipped, we may need to fallback
                            return x  # Fallback: just pass input through
                        elif prev_idx >= 0:
                            return stored_outputs[prev_idx]  # Use last valid output
                        else:
                            return x  # No valid previous layer

                    # If not skipped, store its output
                    stored_outputs[layer_idx] = output
                    return output

                return hook_fn

            # Register hook
            h = mlp.register_forward_hook(create_hook(idx))
            hooks.append(h)

        try:
            yield
        finally:
            for h in hooks:
                h.remove()
            # Clear stored outputs
            stored_outputs.clear()
    

    @contextmanager
    def hook_attention_weights(self, layers: list[int], hook_fn):
        """
        Advanced: inject a hook to modify attention weights after computation.
        hook_fn: Callable(attn_weights: Tensor) -> modified_attn_weights
        """
        handles = []

        def create_hook(layer_idx, hook_fn):
            def hook(module, inputs, outputs):
                # outputs[0]: context, outputs[1]: attn_weights
                # print(outputs)
                modified_weights = hook_fn(outputs[1])
                # 将注意力改为均匀注意力
                return (outputs[0], modified_weights) + outputs[2:]
            return hook

        for layer_idx in layers:
            layer = self.model.language_model.layers[layer_idx]
            h = layer.self_attn.register_forward_hook(create_hook(layer_idx, hook_fn))
            handles.append(h)

        try:
            yield
        finally:
            for h in handles:
                h.remove()
    
    @contextmanager
    def block_attention(self, layers_to_skip, device=None):
        """
        Context manager to skip the forward pass of specified layers' self-attention modules.
        The input to the attention module is directly passed through as output, bypassing attention computation.
        
        Example: If layer 2's attention is blocked, the residual input directly becomes the output.

        :param layers_to_skip: list of layer indices to skip attention (e.g., [2, 3] to skip layers 2 and 3)
        :param device: device for tensors (optional)
        """
        if not isinstance(layers_to_skip, (list, tuple)):
            layers_to_skip = [layers_to_skip]

        hooks = []

        for layer_idx in layers_to_skip:
            layer = self.model.language_model.layers[layer_idx]
            attn_module = layer.self_attn

            def create_hook(layer_idx):
                def hook_fn(module, args, output):
                    """
                    Bypass attention computation: return input as output.
                    
                    Args structure varies by model, but typically:
                    - args[0]: hidden_states (the input to attention)
                    
                    Output structure:
                    - output[0]: attention output (hidden_states)
                    - output[1]: attention_weights (optional)
                    - output[2:]: other optional outputs
                    """
                    # Get the input hidden states
                    hidden_states = args[0]
                    
                    # Return the input directly, bypassing attention
                    # Maintain the output structure: (hidden_states, None, ...)
                    if isinstance(output, tuple):
                        # Keep the same tuple structure but replace attention output with input
                        return (hidden_states,) + (None,) * (len(output) - 1)
                    else:
                        # If output is not a tuple, just return hidden_states
                        return hidden_states

                return hook_fn

            # Register hook on self_attn module
            h = attn_module.register_forward_hook(create_hook(layer_idx))
            hooks.append(h)

        try:
            yield
        finally:
            for h in hooks:
                h.remove()


    @contextmanager
    def block_layers(self, layers_to_skip):
        """
        Context manager to skip both self-attention and MLP of specified layers.
        This completely bypasses the layer's computation, passing input directly as output.
        
        :param layers_to_skip: list of layer indices to skip completely
        :param device: device for tensors (optional)
        """
        if not isinstance(layers_to_skip, (list, tuple)):
            layers_to_skip = [layers_to_skip]

        hooks = []

        for layer_idx in layers_to_skip:
            # print(self.model)
            layer = self.model.language_model.layers[layer_idx]

            def create_hook(layer_idx):
                def hook_fn(module, args, output):
                    """
                    Bypass the entire layer: return input as output.
                    """
                    # Get the input hidden states
                    hidden_states = args[0]
                    
                    # Return input directly
                    if isinstance(output, tuple):
                        return (hidden_states,) + (None,) * (len(output) - 1)
                    else:
                        return hidden_states

                return hook_fn

            # Register hook on the entire layer
            h = layer.register_forward_hook(create_hook(layer_idx))
            hooks.append(h)

        try:
            yield
        finally:
            for h in hooks:
                h.remove()
