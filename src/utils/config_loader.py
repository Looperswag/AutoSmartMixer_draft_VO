# src/utils/config_loader.py

import yaml
import os
import logging

logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration, or None if an error occurs.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at: {config_path}")
        print(f"Error: Configuration file not found at: {config_path}")
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        # Basic validation (can be expanded)
        if not isinstance(config, dict):
            logger.error(f"Configuration file {config_path} is not a valid YAML dictionary.")
            print(f"Error: Configuration file {config_path} is not a valid YAML dictionary.")
            return None
        if "paths" not in config:
            logger.warning(f"'paths' section not found in {config_path}. This might cause issues.")
            # Depending on strictness, you might return None here.
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        print(f"Error: Could not parse YAML configuration file {config_path}. Details: {e}")
        return None
    except IOError as e:
        logger.error(f"Error reading configuration file {config_path}: {e}")
        print(f"Error: Could not read configuration file {config_path}. Details: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading configuration from {config_path}: {e}")
        print(f"Error: An unexpected error occurred while loading configuration from {config_path}. Details: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    # Create a dummy config.yaml for testing if it doesn't exist
    if not os.path.exists("../../config.yaml"):
        print("Creating a dummy config.yaml for testing load_config.")
        dummy_config_content = {
            "project_name": "AISmartMixer_Test",
            "paths": {
                "input_audio_dir": "data/input_audio",
                "input_video_clips_dir": "data/input_video_clips",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        with open("../../config.yaml", "w", encoding="utf-8") as cf:
            yaml.dump(dummy_config_content, cf, default_flow_style=False)
        config_data = load_config("../../config.yaml")
    else:
        # Assuming config.yaml exists at the project root for this test
        config_data = load_config("../../config.yaml")


    if config_data:
        print("\nConfiguration loaded successfully:")
        print(f"Project Name: {config_data.get('project_name', 'N/A')}")
        if 'paths' in config_data:
            print(f"Input Audio Directory: {config_data['paths'].get('input_audio_dir', 'N/A')}")
        if 'logging' in config_data:
            print(f"Logging Level: {config_data['logging'].get('level', 'N/A')}")
    else:
        print("\nFailed to load configuration.")

    # Test non-existent file
    print("\nTesting with a non-existent config file:")
    non_existent_config = load_config("non_existent_config.yaml")
    if non_existent_config is None:
        print("Correctly handled non-existent file.")

    # Test invalid YAML file
    print("\nTesting with an invalid YAML file:")
    with open("invalid_config.yaml", "w", encoding="utf-8") as f:
        f.write("this is not: valid yaml: [")
    invalid_config = load_config("invalid_config.yaml")
    if invalid_config is None:
        print("Correctly handled invalid YAML file.")
    os.remove("invalid_config.yaml")