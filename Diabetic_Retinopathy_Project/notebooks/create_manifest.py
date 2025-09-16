import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_fixed_manifest():
    """Create a manifest with the EXACT correct paths based on debugging results."""
    
    print("🎯 CREATING MANIFEST WITH EXACT CORRECT PATHS")
    print("=" * 60)
    
    # Define paths based on debugging results
    base_path = Path(r"C:\Users\tw0271\Documents\Diabetic_Retinopathy_Project\Diabetic_Retinopathy_Project\data")
    
    manifest_paths = {
        'fundus': Path(r"C:\Users\tw0271\Documents\Diabetic_Retinopathy_Project\Diabetic_Retinopathy_Project\data\raw\fundus\retinal_photography\retinal_photography\manifest.tsv"),
        'oct': Path(r"C:\Users\tw0271\Documents\Diabetic_Retinopathy_Project\Diabetic_Retinopathy_Project\data\raw\oct\retina_oct\manifest.tsv"),
        'flio': Path(r"C:\Users\tw0271\Documents\Diabetic_Retinopathy_Project\Diabetic_Retinopathy_Project\data\raw\flio\retinal_flio\retinal_flio\manifest.tsv")
    }
    
    # Define EXACT correct base directories based on debugging output
    correct_bases = {
        'fundus': base_path / 'raw' / 'fundus' / 'retinal_photography' / 'retinal_photography',
        'oct': base_path / 'raw' / 'oct',  # Use raw/oct directly, not raw/oct/retina_oct
        'flio': base_path / 'raw' / 'flio' / 'retinal_flio' / 'retinal_flio'
    }
    
    all_dfs = []
    stats = {'total_records': 0, 'unique_participants': 0, 'valid_files': 0}
    
    for modality, manifest_path in manifest_paths.items():
        print(f"\n📥 Processing {modality.upper()} manifest...")
        
        if not manifest_path.exists():
            print(f"   ❌ Manifest not found: {manifest_path}")
            continue
            
        # Load manifest
        df = pd.read_csv(manifest_path, sep="\t")
        initial_count = len(df)
        stats['total_records'] += initial_count
        
        print(f"   📊 Loaded {initial_count:,} records")
        
        # Clean participant IDs
        df['participant_id'] = df['participant_id'].astype(str).str.strip()
        df = df[df['participant_id'].notna() & (df['participant_id'] != '') & (df['participant_id'] != 'nan')]
        
        # Remove duplicates within this modality (keep first occurrence per participant)
        df = df.drop_duplicates(subset=['participant_id'], keep='first').copy()
        unique_count = len(df)
        duplicates_removed = initial_count - unique_count
        
        print(f"   🗑️ Removed {duplicates_removed:,} duplicates")
        print(f"   👥 Unique participants: {unique_count:,}")
        
        stats['unique_participants'] += unique_count
        
        # Create corrected full paths using EXACT logic from debugging
        base_dir = correct_bases[modality]
        
        def create_exact_path(relative_path):
            """Create exact correct path based on debugging findings."""
            if pd.isna(relative_path):
                return None
                
            rel_path_str = str(relative_path).strip().lstrip('/')
            
            if modality == 'fundus':
                # Fundus: Remove the duplicate 'retinal_photography/' from the manifest path
                # Manifest has: /retinal_photography/cfp/icare_eidon/...
                # We want: base/cfp/icare_eidon/... (base already includes retinal_photography)
                if rel_path_str.startswith('retinal_photography/'):
                    rel_path_str = rel_path_str[len('retinal_photography/'):]
                full_path = base_dir / rel_path_str
                
            elif modality == 'oct':
                # OCT: Change retinal_oct -> retina_oct and use raw/oct as base
                # Manifest has: /retinal_oct/structural_oct/heidelberg_spectralis/...
                # We want: raw/oct/retina_oct/structural_oct/heidelberg_spectralis/...
                rel_path_corrected = rel_path_str.replace('retinal_oct/', 'retina_oct/')
                full_path = base_dir / rel_path_corrected
                
            elif modality == 'flio':
                # FLIO: Remove the duplicate 'retinal_flio/' from the manifest path  
                # Manifest has: /retinal_flio/flio/heidelberg_flio/...
                # We want: base/flio/heidelberg_flio/... (base already includes retinal_flio)
                if rel_path_str.startswith('retinal_flio/'):
                    rel_path_str = rel_path_str[len('retinal_flio/'):]
                full_path = base_dir / rel_path_str
                
            else:
                full_path = base_dir / rel_path_str
            
            return str(full_path) if full_path.exists() else None
        
        # Apply path correction
        df[f'{modality}_path'] = df['filepath'].apply(create_exact_path)
        
        # Only keep rows where files actually exist
        df_valid = df[df[f'{modality}_path'].notna()].copy()
        valid_count = len(df_valid)
        
        print(f"   ✅ Valid files found: {valid_count:,}")
        stats['valid_files'] += valid_count
        
        if valid_count > 0:
            # Keep only needed columns
            df_final = df_valid[['participant_id', f'{modality}_path']].copy()
            all_dfs.append(df_final)
            
            # Show sample paths
            print(f"   📝 Sample valid path: {df_final[f'{modality}_path'].iloc[0]}")
        else:
            print(f"   ❌ No valid files found for {modality}")
    
    if not all_dfs:
        print("\n❌ ERROR: No valid data found for any modality!")
        return None
    
    # Merge all modalities on participant_id
    print(f"\n🔗 Merging {len(all_dfs)} modalities...")
    
    merged_df = all_dfs[0]
    for df in all_dfs[1:]:
        merged_df = merged_df.merge(df, on='participant_id', how='outer')
    
    # Check how many participants have each modality
    path_cols = [col for col in merged_df.columns if col.endswith('_path')]
    
    print(f"\n📊 MODALITY COVERAGE:")
    for col in path_cols:
        count = merged_df[col].notna().sum()
        percentage = count / len(merged_df) * 100
        modality = col.replace('_path', '').upper()
        print(f"   {modality}: {count:,} participants ({percentage:.1f}%)")
    
    # Filter to participants with at least one modality
    has_any_modality = merged_df[path_cols].notna().any(axis=1)
    final_df = merged_df[has_any_modality].copy()
    
    print(f"\n✅ Final dataset: {len(final_df):,} participants with at least one modality")
    
    # Show modality combinations
    print(f"\n🔗 MODALITY COMBINATIONS:")
    
    # Count combinations
    combo_counts = {}
    for _, row in final_df.iterrows():
        available = []
        for col in path_cols:
            if pd.notna(row[col]):
                modality = col.replace('_path', '')
                available.append(modality)
        
        combo = tuple(sorted(available))
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
    
    # Show top combinations
    for combo, count in sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        if combo:  # Only show non-empty combinations
            combo_str = " + ".join(combo)
            percentage = count / len(final_df) * 100
            print(f"   {combo_str:>20}: {count:>6,} participants ({percentage:>5.1f}%)")
    
    # Save the corrected manifest
    output_path = base_path / 'manifest_fixed.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"\n💾 FIXED MANIFEST SAVED")
    print(f"📁 Location: {output_path}")
    print(f"📊 Participants: {len(final_df):,}")
    
    # Show summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   📥 Total records loaded: {stats['total_records']:,}")
    print(f"   👥 Total unique participants: {stats['unique_participants']:,}")
    print(f"   ✅ Total valid files: {stats['valid_files']:,}")
    print(f"   🎯 Final participants: {len(final_df):,}")
    
    # Show sample of final data
    print(f"\n📋 SAMPLE OF FIXED DATA:")
    print(final_df.head().to_string(index=False))
    
    return final_df

def verify_exact_paths():
    """Quick verification that our exact path logic works."""
    print("🔍 VERIFYING EXACT PATH LOGIC")
    print("=" * 40)
    
    base_path = Path(r"C:\Users\tw0271\Documents\Diabetic_Retinopathy_Project\Diabetic_Retinopathy_Project\data")
    
    # Test the exact corrections
    test_cases = [
        {
            'modality': 'fundus',
            'manifest_path': '/retinal_photography/cfp/icare_eidon/1001/1001_eidon_mosaic_cfp_l_1.2.826.0.1.3680043.8.641.1.20230809.2044.20521.dcm',
            'base': base_path / 'raw' / 'fundus' / 'retinal_photography' / 'retinal_photography',
            'expected': 'cfp/icare_eidon/1001/1001_eidon_mosaic_cfp_l_1.2.826.0.1.3680043.8.641.1.20230809.2044.20521.dcm'
        },
        {
            'modality': 'oct',
            'manifest_path': '/retinal_oct/structural_oct/heidelberg_spectralis/1001/1001_spectralis_onh_rc_hr_oct_l_1.3.6.1.4.1.33437.11.4.7587979.98316546453556.22400.4.1.dcm',
            'base': base_path / 'raw' / 'oct',
            'expected': 'retina_oct/structural_oct/heidelberg_spectralis/1001/1001_spectralis_onh_rc_hr_oct_l_1.3.6.1.4.1.33437.11.4.7587979.98316546453556.22400.4.1.dcm'
        },
        {
            'modality': 'flio',
            'manifest_path': '/retinal_flio/flio/heidelberg_flio/1001/1001_flio_long_wavelength_l_1.2.826.0.1.3680043.8.498.72262700290222155880211485085233319011.dcm',
            'base': base_path / 'raw' / 'flio' / 'retinal_flio' / 'retinal_flio',
            'expected': 'flio/heidelberg_flio/1001/1001_flio_long_wavelength_l_1.2.826.0.1.3680043.8.498.72262700290222155880211485085233319011.dcm'
        }
    ]
    
    for test in test_cases:
        modality = test['modality']
        manifest_path = test['manifest_path']
        base = test['base']
        expected_rel = test['expected']
        
        print(f"\n🔍 Testing {modality.upper()}:")
        
        # Apply our exact correction logic
        rel_path_str = manifest_path.strip().lstrip('/')
        
        if modality == 'fundus':
            if rel_path_str.startswith('retinal_photography/'):
                rel_path_str = rel_path_str[len('retinal_photography/'):]
        elif modality == 'oct':
            rel_path_str = rel_path_str.replace('retinal_oct/', 'retina_oct/')
        elif modality == 'flio':
            if rel_path_str.startswith('retinal_flio/'):
                rel_path_str = rel_path_str[len('retinal_flio/'):]
        
        full_path = base / rel_path_str
        exists = full_path.exists()
        
        print(f"   📁 Manifest path: {manifest_path}")
        print(f"   🔧 Processed to: {rel_path_str}")
        print(f"   📂 Full path: {full_path}")
        print(f"   📋 Expected: {base / expected_rel}")
        print(f"   {'✅ EXISTS!' if exists else '❌ NOT FOUND'}")
        
        if rel_path_str == expected_rel:
            print(f"   ✅ Path processing CORRECT")
        else:
            print(f"   ❌ Path processing INCORRECT")
            print(f"      Got: {rel_path_str}")
            print(f"      Expected: {expected_rel}")

if __name__ == "__main__":
    # First verify our logic
    verify_exact_paths()
    
    print(f"\n" + "="*80)
    
    # Then create the fixed manifest
    try:
        result = create_fixed_manifest()
        if result is not None:
            print(f"\n🎉 SUCCESS! Fixed manifest created successfully!")
        else:
            print(f"\n❌ FAILED: Could not create fixed manifest")
    except Exception as e:
        logger.error(f"Error creating fixed manifest: {e}")
        import traceback
        traceback.print_exc()