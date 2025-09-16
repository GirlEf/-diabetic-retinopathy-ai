import csv
from pathlib import Path
from collections import Counter

def analyze_metadata():
    print("🔍 METADATA ANALYSIS")
    print("="*50)
    
    metadata_path = Path("data/raw/metadata.csv")
    
    if not metadata_path.exists():
        print("❌ metadata.csv not found")
        return
    
    # Read CSV
    data = []
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    print(f"📊 Total samples: {len(data)}")
    
    # Analyze splits
    splits = Counter(row['split'] for row in data)
    print(f"\n📋 Data splits:")
    for split, count in splits.items():
        print(f"   {split}: {count} samples")
    
    # Analyze labels
    labels = Counter(row['label'] for row in data)
    print(f"\n🏷️  Label distribution:")
    for label, count in sorted(labels.items()):
        print(f"   {label}: {count} samples")
    
    # Show some sample file paths
    print(f"\n📄 Sample file paths:")
    for i, row in enumerate(data[:3]):
        print(f"   Sample {i+1}:")
        print(f"     ID: {row['participant_id']}")
        print(f"     Split: {row['split']}")
        print(f"     Label: {row['label']}")
        print(f"     Fundus: {Path(row['fundus_path']).name}")
        print(f"     OCT: {Path(row['oct_path']).name}")
        print(f"     FLIO: {Path(row['flio_path']).name}")
        print()

if __name__ == "__main__":
    analyze_metadata()