import os
import json
from .BiRefNet_config import BiRefNetConfig

def load_birefnet_config():
    """
    Reads and loads the bundled configuration for the BiRefNet model.

    This function facilitates strictly offline initialization by dynamically locating the 
    `config.json` file associated with BiRefNet within the local package directory. It 
    securely reads the local JSON file and unpacks its contents directly into a Hugging Face 
    `BiRefNetConfig` object, ensuring the package remains completely air-gapped and independent 
    of external network requests.

    Raises:
        FileNotFoundError: If the corresponding offline `config.json` file cannot be located 
                           in the module directory.

    Returns:
        BiRefNetConfig: The loaded configuration object ready to be passed into the model loader.
    """
    # Locate the directory this specific python file lives in
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.json")
    
    # Read the local json securely
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        
    # Unpack the dictionary directly into the Config class
    return BiRefNetConfig(**config_dict)

__all__ = ["load_birefnet_config"]
