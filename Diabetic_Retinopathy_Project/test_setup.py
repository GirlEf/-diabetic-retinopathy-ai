import sys
from pathlib import Path

def test_setup():
    print("Testing project setup...")
    
    # Check if directories exist
    required_dirs = ["src", "configs", "notebooks", "data"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
    
    # Check if config files exist
    config_files = ["configs/paths.yaml", "configs/hyperparams.yaml", "requirements.txt"]
    for file_name in config_files:
        if Path(file_name).exists():
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
    
    print("\nProject structure test complete!")

if __name__ == "__main__":
    test_setup()