import os
import json
from transformers import SamModel, SamConfig, SamProcessor

VALID_SAM_MODELS = ["sam-vit-b", "sam-vit-l", "sam-vit-h"]

def load_sam_config(model_type: str = "sam-vit-b"):
    """
    Reads and loads the bundled configuration for a specified Segment Anything Model (SAM) variant.

    This function facilitates strictly offline initialization by dynamically locating the 
    `config.json` file associated with the chosen SAM variant (Base, Large, or Huge) within 
    the local package directory. It reads the JSON file securely and converts it into a 
    Hugging Face `SamConfig` object, entirely bypassing external network requests.

    Args:
        model_type (str, optional): The identifier for the desired SAM variant. 
                                    Must be one of `["sam-vit-b", "sam-vit-l", "sam-vit-h"]`. 
                                    Default is `"sam-vit-b"`.

    Raises:
        ValueError: If the provided `model_type` is not within the supported valid variants list.
        FileNotFoundError: If the corresponding offline `config.json` file cannot be located.

    Returns:
        SamConfig: The loaded configuration object ready to be passed into a `SamModel`.
    """
    if model_type not in VALID_SAM_MODELS:
        raise ValueError(f"Invalid model_type '{model_type}'. Choose from: {VALID_SAM_MODELS}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Dynamically route to the correct variant folder
    config_path = os.path.join(current_dir, model_type, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Offline config missing at {config_path}.")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        
    return SamConfig.from_dict(config_dict)

def load_sam_processor(model_type: str = "sam-vit-b", use_fast: bool = False):
    """
    Loads the offline processor configuration for a specified Segment Anything Model (SAM) variant.

    This function instantiates a `SamProcessor` by reading bundled tokenizer and preprocessor 
    configuration files from the local variant directory. Loading from a local path ensures the 
    package remains completely air-gapped. Users can optionally toggle the PyTorch/Torchvision 
    C++ backend via the `use_fast` flag to balance speed with strict backward compatibility.

    Args:
        model_type (str, optional): The identifier for the desired SAM variant. 
                                    Must be one of `["sam-vit-b", "sam-vit-l", "sam-vit-h"]`. 
                                    Default is `"sam-vit-b"`.
        use_fast (bool, optional): Flag indicating whether to use the C++ optimized fast processor. 
                                   Default is False.

    Raises:
        ValueError: If the provided `model_type` is not within the supported valid variants list.
        FileNotFoundError: If the corresponding offline processor directory cannot be located.

    Returns:
        SamProcessor: The instantiated processor ready for image and prompt transformations.
    """
    if model_type not in VALID_SAM_MODELS:
        raise ValueError(f"Invalid model_type '{model_type}'. Choose from: {VALID_SAM_MODELS}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Dynamically route to the correct processor folder
    processor_path = os.path.join(current_dir, model_type, "preprocessor_config.json")
    
    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"Offline processor config missing at {processor_path}.")
    
    # Loading from a local directory disables Hugging Face network calls
    return SamProcessor.from_pretrained(processor_path, use_fast=use_fast)

__all__ = ["SamModel", "load_sam_config", "load_sam_processor"]
