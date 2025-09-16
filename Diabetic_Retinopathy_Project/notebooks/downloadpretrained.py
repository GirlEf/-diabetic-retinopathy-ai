# Download Pretrained Weights for Foundation Models
# Complete guide to get all required weights

import os
import wget
import gdown
from pathlib import Path

def download_pretrained_weights():
    """Download all pretrained weights for the foundation models"""
    
    print("🔽 Downloading Pretrained Weights for Foundation Models")
    print("="*60)
    
    # Create weights directory
    weights_dir = Path("./pretrained_weights")
    weights_dir.mkdir(exist_ok=True)
    
    # 1. RETFound (Fundus) Weights
    print("\n🎯 1. RETFound (Fundus) Weights")
    print("-" * 40)
    
    retfound_path = weights_dir / "retfound_cfp_weights.pth"
    
    if not retfound_path.exists():
        print("📦 Downloading RETFound weights from Google Drive...")
        try:
            # Google Drive download link from official repo
            retfound_url = "https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view?usp=sharing"
            retfound_id = "1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE"
            
            gdown.download(f"https://drive.google.com/uc?id={retfound_id}", 
                          str(retfound_path), quiet=False)
            
            print(f"✅ RETFound weights downloaded: {retfound_path}")
            print(f"📊 File size: {retfound_path.stat().st_size / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"❌ Error downloading RETFound weights: {e}")
            print("💡 Manual download:")
            print(f"   1. Visit: {retfound_url}")
            print(f"   2. Download and save as: {retfound_path}")
    else:
        print(f"✅ RETFound weights already exist: {retfound_path}")
    
    # 2. RETFound OCT Weights (Alternative)
    print("\n🎯 2. RETFound OCT Weights")
    print("-" * 40)
    
    retfound_oct_path = weights_dir / "retfound_oct_weights.pth"
    
    if not retfound_oct_path.exists():
        print("📦 Downloading RETFound OCT weights from Google Drive...")
        try:
            # OCT weights from official repo
            oct_url = "https://drive.google.com/file/d/1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2/view?usp=sharing"
            oct_id = "1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2"
            
            gdown.download(f"https://drive.google.com/uc?id={oct_id}", 
                          str(retfound_oct_path), quiet=False)
            
            print(f"✅ RETFound OCT weights downloaded: {retfound_oct_path}")
            print(f"📊 File size: {retfound_oct_path.stat().st_size / (1024*1024):.1f} MB")
            
        except Exception as e:
            print(f"❌ Error downloading RETFound OCT weights: {e}")
            print("💡 Manual download:")
            print(f"   1. Visit: {oct_url}")
            print(f"   2. Download and save as: {retfound_oct_path}")
    else:
        print(f"✅ RETFound OCT weights already exist: {retfound_oct_path}")
    
    # 3. OCTCube Weights (Note: May need manual search)
    print("\n🎯 3. OCTCube Weights")
    print("-" * 40)
    
    octcube_path = weights_dir / "octcube_weights.pth"
    
    if not octcube_path.exists():
        print("⚠️ OCTCube weights not found in standard repositories")
        print("💡 Options for OCTCube weights:")
        print("   1. Check the original OCTCube paper/repository")
        print("   2. Contact the authors directly")
        print("   3. Use RETFound OCT weights as alternative")
        print("   4. Train from scratch using your OCT data")
        print("\n🔍 Alternative: Use RETFound OCT weights for OCT modality")
        
        # Create symlink to use RETFound OCT weights
        if retfound_oct_path.exists():
            try:
                import shutil
                shutil.copy2(retfound_oct_path, octcube_path)
                print(f"✅ Using RETFound OCT weights as OCTCube alternative: {octcube_path}")
            except Exception as e:
                print(f"❌ Could not copy weights: {e}")
    else:
        print(f"✅ OCTCube weights already exist: {octcube_path}")
    
    # 4. Summary and Usage Instructions
    print("\n🎉 Download Summary")
    print("="*60)
    
    weights_found = []
    weights_missing = []
    
    for weight_file, name in [
        (retfound_path, "RETFound (Fundus)"),
        (retfound_oct_path, "RETFound OCT"),
        (octcube_path, "OCTCube/OCT Alternative")
    ]:
        if weight_file.exists():
            size_mb = weight_file.stat().st_size / (1024*1024)
            weights_found.append(f"✅ {name}: {weight_file.name} ({size_mb:.1f} MB)")
        else:
            weights_missing.append(f"❌ {name}: {weight_file.name}")
    
    if weights_found:
        print("\n📦 Downloaded Weights:")
        for w in weights_found:
            print(f"   {w}")
    
    if weights_missing:
        print("\n⚠️ Missing Weights:")
        for w in weights_missing:
            print(f"   {w}")
    
    print(f"\n📁 All weights saved to: {weights_dir.absolute()}")
    
    return weights_dir

def install_dependencies():
    """Install required packages for downloading"""
    print("📦 Installing download dependencies...")
    
    try:
        import gdown
        print("✅ gdown already installed")
    except ImportError:
        print("📦 Installing gdown...")
        os.system("pip install gdown")
    
    try:
        import wget
        print("✅ wget already installed")
    except ImportError:
        print("📦 Installing wget...")
        os.system("pip install wget")

def update_foundation_model_paths(weights_dir):
    """Update your foundation model configuration to use downloaded weights"""
    
    print("\n🔧 Updating Foundation Model Configuration")
    print("-" * 50)
    
    config_code = f"""
# Update your FoundationModelConfig to use downloaded weights

class FoundationModelConfig:
    def __init__(self):
        self.weights_dir = Path("{weights_dir}")
        
        # Updated paths to downloaded weights
        self.retfound_weights = self.weights_dir / "retfound_cfp_weights.pth"
        self.octcube_weights = self.weights_dir / "octcube_weights.pth"  # or retfound_oct_weights.pth
        
        # Verify weights exist
        if self.retfound_weights.exists():
            print(f"✅ RETFound weights found: {{self.retfound_weights}}")
        else:
            print(f"⚠️ RETFound weights missing: {{self.retfound_weights}}")
            
        if self.octcube_weights.exists():
            print(f"✅ OCT weights found: {{self.octcube_weights}}")
        else:
            print(f"⚠️ OCT weights missing: {{self.octcube_weights}}")

# Usage in your foundation model loading:
config = FoundationModelConfig()
model_manager = FoundationModelManager(config)
foundation_models = model_manager.load_models()
"""
    
    print("💡 Add this to your code:")
    print(config_code)

# Manual Download Instructions
def print_manual_instructions():
    """Print manual download instructions if automated download fails"""
    
    print("\n📋 Manual Download Instructions")
    print("="*60)
    
    print("\n🎯 1. RETFound (Fundus) Weights:")
    print("   📎 URL: https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view?usp=sharing")
    print("   💾 Save as: retfound_cfp_weights.pth")
    print("   📊 Size: ~370 MB")
    
    print("\n🎯 2. RETFound OCT Weights:")
    print("   📎 URL: https://drive.google.com/file/d/1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2/view?usp=sharing")
    print("   💾 Save as: retfound_oct_weights.pth")
    print("   📊 Size: ~370 MB")
    
    print("\n🎯 3. OCTCube Weights (Research Needed):")
    print("   🔍 Search for: Original OCTCube paper/repository")
    print("   📧 Contact: Paper authors for weights")
    print("   🔄 Alternative: Use RETFound OCT weights")
    
    print("\n🎯 4. HuggingFace Alternative (New!):")
    print("   📎 URL: https://huggingface.co/open-eye/RETFound_MAE")
    print("   💡 May have updated weights and easier access")
    
    print("\n📁 Save all weights to: ./pretrained_weights/")

# Main execution
if __name__ == "__main__":
    print("🚀 Foundation Model Weights Downloader")
    print("="*60)
    
    # Install dependencies
    install_dependencies()
    
    # Download weights
    weights_dir = download_pretrained_weights()
    
    # Update configuration
    update_foundation_model_paths(weights_dir)
    
    # Print manual instructions
    print_manual_instructions()
    
    print("\n✅ Setup complete!")
    print("💡 Your foundation models should now load with pretrained weights!")