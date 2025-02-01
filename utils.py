from modeling_gemma import PaliGemmaForConditionalGenerator, PaligemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os


def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGenerator, AutoTokenizer]:
    # load the HF tokenizer - we did not code for the tokenizer for this implementation
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side = "right")
    assert tokenizer.padding_side == "right"

    # safetensor files are dictionies that  basically contains some parts of the weights of the models
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # now load these safetensors one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaligemmaConfig(**model_config_file)
    
    # create the model using the configuration
    model = PaliGemmaForConditionalGenerator(config).to(device)

    # load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    model.tie_weights()

    return (model, tokenizer)
    




