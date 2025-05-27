# src/utils/file_handler.py

import os
import json
import logging

logger = logging.getLogger(__name__)

def ensure_dir_exists(dir_path):
    """
    Ensures that a directory exists. If it doesn't, it creates it.

    Args:
        dir_path (str): The path to the directory.

    Returns:
        bool: True if the directory exists or was created successfully, False otherwise.
    """
    if not dir_path:
        logger.warning("ensure_dir_exists called with an empty or None path.")
        return False
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Directory created: {dir_path}")
        else:
            logger.debug(f"Directory already exists: {dir_path}")
        return True
    except OSError as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred in ensure_dir_exists for {dir_path}: {e}")
        return False

def save_json(data, file_path, indent=4, ensure_ascii=False):
    """
    Saves a Python dictionary or list to a JSON file.

    Args:
        data (dict or list): The data to save.
        file_path (str): The path to the JSON file.
        indent (int, optional): Indentation level for pretty printing. Defaults to 4.
        ensure_ascii (bool, optional): If False, non-ASCII characters are written as is.
                                      Defaults to False for better readability with UTF-8.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if not file_path:
        logger.error("save_json called with an empty or None file_path.")
        return False
    try:
        # Ensure the directory for the file exists
        dir_name = os.path.dirname(file_path)
        if dir_name: # Only try to create if dirname is not empty (e.g. for relative paths in current dir)
            if not ensure_dir_exists(dir_name):
                logger.error(f"Could not ensure directory exists for {file_path}. JSON not saved.")
                return False

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        logger.info(f"Data successfully saved to JSON file: {file_path}")
        return True
    except TypeError as e:
        logger.error(f"TypeError saving data to JSON {file_path} (data might not be serializable): {e}")
        return False
    except IOError as e:
        logger.error(f"IOError saving data to JSON {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving JSON to {file_path}: {e}")
        return False

def load_json(file_path):
    """
    Loads data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: The loaded data, or None if an error occurs or file not found.
    """
    if not file_path:
        logger.error("load_json called with an empty or None file_path.")
        return None
    if not os.path.exists(file_path):
        logger.warning(f"JSON file not found: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Data successfully loaded from JSON file: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {file_path}: {e}")
        return None
    except IOError as e:
        logger.error(f"IOError reading JSON file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading JSON from {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Setup basic logging for testing this module directly
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Test ensure_dir_exists
    print("\n--- Testing ensure_dir_exists ---")
    test_dir = "temp_test_dir/subdir"
    if ensure_dir_exists(test_dir):
        print(f"Successfully ensured {test_dir} exists.")
        if os.path.exists(test_dir):
            print(f"Verified: {test_dir} was created.")
        else:
            print(f"Error: {test_dir} was not actually created.")
    else:
        print(f"Failed to ensure {test_dir} exists.")

    # Test save_json and load_json
    print("\n--- Testing save_json and load_json ---")
    test_data = {
        "name": "AISmartMixer Test",
        "version": 1.0,
        "features": ["audio processing", "video analysis", "timeline generation"],
        "unicode_test": "你好世界"
    }
    test_file_path = os.path.join(test_dir, "test_data.json")

    if save_json(test_data, test_file_path):
        print(f"Successfully saved data to {test_file_path}")
        loaded_data = load_json(test_file_path)
        if loaded_data:
            print("Successfully loaded data:")
            print(json.dumps(loaded_data, indent=2, ensure_ascii=False))
            assert loaded_data == test_data, "Loaded data does not match original data!"
            print("Data integrity check passed.")
        else:
            print(f"Failed to load data from {test_file_path}")
    else:
        print(f"Failed to save data to {test_file_path}")

    # Test loading a non-existent file
    print("\n--- Testing load_json with non-existent file ---")
    non_existent_data = load_json("non_existent.json")
    if non_existent_data is None:
        print("Correctly handled loading non-existent file (returned None).")
    else:
        print("Error: Should have returned None for non-existent file.")

    # Test saving non-serializable data
    print("\n--- Testing save_json with non-serializable data ---")
    non_serializable_data = {"name": "test", "value": object()} # object() is not JSON serializable
    if not save_json(non_serializable_data, os.path.join(test_dir, "bad_data.json")):
        print("Correctly handled non-serializable data during save (returned False).")
    else:
        print("Error: Should have failed to save non-serializable data.")


    # Clean up test directory and files
    print("\n--- Cleaning up ---")
    try:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"Removed {test_file_path}")
        if os.path.exists(os.path.join(test_dir, "bad_data.json")):
             os.remove(os.path.join(test_dir, "bad_data.json"))
             print(f"Removed {os.path.join(test_dir, 'bad_data.json')}")
        if os.path.exists(test_dir):
            os.rmdir(test_dir) # remove subdir
            os.rmdir(os.path.dirname(test_dir)) # remove temp_test_dir
            print(f"Removed directory {test_dir} and its parent.")
    except OSError as e:
        print(f"Error during cleanup: {e}")