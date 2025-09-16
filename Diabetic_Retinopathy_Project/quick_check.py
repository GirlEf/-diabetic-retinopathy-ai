import os
from pathlib import Path

def quick_check():
    print("🔍 DATASET QUICK CHECK")
    print("="*40)
    
    # Count files in each folder
    folders = ['fundus', 'oct', 'flio']
    
    for folder in folders:
        folder_path = Path(f"data/raw/{folder}")
        if folder_path.exists():
            file_count = len([f for f in folder_path.iterdir() if f.is_file()])
            print(f"📁 {folder}/: {file_count} files")
            
            # Show first few file names
            files = list(folder_path.iterdir())[:3]
            for f in files:
                if f.is_file():
                    print(f"   📄 {f.name}")
        else:
            print(f"❌ {folder}/ not found")
    
    # Check metadata file
    metadata_path = Path("data/raw/metadata.csv")
    if metadata_path.exists():
        size_mb = metadata_path.stat().st_size / (1024*1024)
        print(f"\n🏷️  metadata.csv: {size_mb:.1f} MB")
        
        # Try to read first few lines
        try:
            with open(metadata_path, 'r') as f:
                lines = f.readlines()[:5]
            print("   First 5 lines:")
            for i, line in enumerate(lines):
                print(f"   {i+1}: {line.strip()}")
        except Exception as e:
            print(f"   ❌ Error reading: {e}")
    else:
        print("❌ metadata.csv not found")

if __name__ == "__main__":
    quick_check()