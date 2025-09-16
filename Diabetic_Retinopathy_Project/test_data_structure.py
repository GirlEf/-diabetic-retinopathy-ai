from pathlib import Path
import pandas as pd

def analyze_dataset():
    print("🔍 DATASET ANALYSIS")
    print("="*50)
    
    # Check main structure
    raw_path = Path("data/raw")
    if not raw_path.exists():
        print("❌ data/raw folder not found!")
        return
    
    print(f"📁 Dataset location: {raw_path.absolute()}")
    
    # Count all files and folders
    all_files = list(raw_path.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    dirs_only = [f for f in all_files if f.is_dir()]
    
    print(f"📊 Total items: {len(all_files)}")
    print(f"📄 Files: {len(files_only)}")
    print(f"📁 Directories: {len(dirs_only)}")
    
    # Check for common medical imaging file types
    extensions = {}
    for file in files_only:
        ext = file.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    
    print(f"\n🎯 File types found:")
    for ext, count in sorted(extensions.items()):
        print(f"   {ext or 'no extension'}: {count} files")
    
    # Look for expected folders
    expected_folders = ['fundus', 'oct', 'flio']
    print(f"\n📋 Looking for expected modality folders:")
    
    for folder in expected_folders:
        folder_path = raw_path / folder
        if folder_path.exists():
            file_count = len(list(folder_path.rglob("*")))
            print(f"   ✅ {folder}/: {file_count} items")
        else:
            print(f"   ❓ {folder}/: not found")
    
    # Look for labels file
    possible_label_files = [
        "labels.csv", "Labels.csv", "metadata.csv", 
        "annotations.csv", "grades.csv"
    ]
    
    print(f"\n🏷️  Looking for labels file:")
    labels_found = False
    for label_file in possible_label_files:
        label_path = raw_path / label_file
        if label_path.exists():
            print(f"   ✅ Found: {label_file}")
            try:
                df = pd.read_csv(label_path)
                print(f"      📊 Rows: {len(df)}")
                print(f"      📋 Columns: {list(df.columns)}")
                labels_found = True
                break
            except Exception as e:
                print(f"      ❌ Error reading: {e}")
    
    if not labels_found:
        print("   ❓ No standard labels file found")
        print("   💡 Look for CSV files manually:")
        csv_files = list(raw_path.rglob("*.csv"))
        for csv_file in csv_files[:5]:  # Show first 5 CSV files
            print(f"      📄 {csv_file.relative_to(raw_path)}")
    
    # Show directory structure (first few levels)
    print(f"\n🌳 Directory structure (top level):")
    for item in sorted(raw_path.iterdir()):
        if item.is_dir():
            subitem_count = len(list(item.iterdir()))
            print(f"   📁 {item.name}/ ({subitem_count} items)")
        else:
            print(f"   📄 {item.name}")

if __name__ == "__main__":
    analyze_dataset()