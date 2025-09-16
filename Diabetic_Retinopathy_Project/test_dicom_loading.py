import pydicom
import numpy as np
from pathlib import Path

def test_dicom_loading():
    print("🔍 TESTING DICOM FILE LOADING")
    print("="*40)
    
    # Test files from each modality
    modalities = ['fundus', 'oct', 'flio']
    
    for modality in modalities:
        print(f"\n📁 Testing {modality} files:")
        
        folder_path = Path(f"data/raw/{modality}")
        dcm_files = list(folder_path.glob("*.dcm"))
        
        if dcm_files:
            test_file = dcm_files[0]  # Test first file
            print(f"   Testing: {test_file.name}")
            
            try:
                # Load DICOM
                ds = pydicom.dcmread(test_file)
                
                # Get image data
                if hasattr(ds, 'pixel_array'):
                    image = ds.pixel_array
                    print(f"   ✅ Shape: {image.shape}")
                    print(f"   ✅ Data type: {image.dtype}")
                    print(f"   ✅ Value range: [{image.min()}, {image.max()}]")
                    
                    # Check DICOM metadata
                    if hasattr(ds, 'PatientID'):
                        print(f"   📋 Patient ID: {ds.PatientID}")
                    if hasattr(ds, 'Modality'):
                        print(f"   📋 Modality: {ds.Modality}")
                        
                else:
                    print(f"   ❌ No pixel data found")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ❌ No .dcm files found in {folder_path}")

if __name__ == "__main__":
    test_dicom_loading()