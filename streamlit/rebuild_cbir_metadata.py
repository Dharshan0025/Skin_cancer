"""
rebuild_cbir_metadata.py
========================
Fixes the critical CBIR bug where all rows in cbir_metadata.csv have
'HAM10000' as the diagnosis column instead of the actual biopsy label.

USAGE
-----
1. Set HAM10000_METADATA_PATH below to point to your local copy of the
   official HAM10000 metadata CSV from Kaggle:
   https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
   (file is called HAM10000_metadata.csv, ~600 KB)

2. Run:  python rebuild_cbir_metadata.py

3. Output files written to streamlit/models/:
   - cbir_metadata_backup.csv   (original, untouched)
   - cbir_metadata.csv          (fixed — real diagnosis labels)
   - cbir_metadata.json         (JSON equivalent, for reference)
"""

import os
import shutil
import pandas as pd

# ── USER CONFIGURATION ────────────────────────────────────────────────────────
# Set this to the path of the official HAM10000 metadata CSV downloaded from
# Kaggle. The file has columns: image_id, dx, dx_type, age, sex, localization
HAM10000_METADATA_PATH = r"C:\Users\Jeevakumar\.cache\kagglehub\datasets\dharshan0025\skin-cancer-dataset\versions\1\HAM10000_metadata"
# ─────────────────────────────────────────────────────────────────────────────

CBIR_METADATA_PATH = os.path.join(os.path.dirname(__file__), "models", "cbir_metadata.csv")
CBIR_BACKUP_PATH   = os.path.join(os.path.dirname(__file__), "models", "cbir_metadata_backup.csv")
CBIR_JSON_PATH     = os.path.join(os.path.dirname(__file__), "models", "cbir_metadata.json")

# HAM10000 dx abbreviation → full class name used in DermAI
LABEL_MAP = {
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'bcc':   'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'vasc':  'Vascular Lesion',
}


def main():
    # ── Validate paths ────────────────────────────────────────────────────────
    if HAM10000_METADATA_PATH == "path/to/HAM10000_metadata.csv":
        raise ValueError(
            "Please set HAM10000_METADATA_PATH at the top of this script "
            "to point to your local HAM10000_metadata.csv file."
        )
    if not os.path.exists(HAM10000_METADATA_PATH):
        raise FileNotFoundError(f"HAM10000 metadata not found at: {HAM10000_METADATA_PATH}")
    if not os.path.exists(CBIR_METADATA_PATH):
        raise FileNotFoundError(f"CBIR metadata not found at: {CBIR_METADATA_PATH}")

    # ── Load files ────────────────────────────────────────────────────────────
    print(f"Loading CBIR metadata from: {CBIR_METADATA_PATH}")
    cbir_df = pd.read_csv(CBIR_METADATA_PATH)
    print(f"  Rows: {len(cbir_df)}   Columns: {list(cbir_df.columns)}")

    print(f"\nLoading HAM10000 metadata from: {HAM10000_METADATA_PATH}")
    ham_df = pd.read_csv(HAM10000_METADATA_PATH, sep=None, engine='python')
    print(f"  Rows: {len(ham_df)}   Columns: {list(ham_df.columns)}")

    # ── Strip .jpg extension from CBIR image_id for joining ───────────────────
    # cbir_metadata: image_id = "ISIC_0024306.jpg"
    # ham_df:        image_id = "ISIC_0024306"
    cbir_df['image_id_bare'] = cbir_df['image_id'].str.replace('.jpg', '', regex=False)

    # ── Left join on image_id_bare == ham_df.image_id ─────────────────────────
    merged = cbir_df.merge(
        ham_df[['image_id', 'dx']],
        left_on='image_id_bare',
        right_on='image_id',
        how='left',
        suffixes=('', '_ham'),
    )

    # ── Map dx abbreviation → full class name ─────────────────────────────────
    merged['diagnosis'] = merged['dx'].map(LABEL_MAP)

    # ── Fill any failed joins with 'Unknown' and report ───────────────────────
    unknown_mask  = merged['diagnosis'].isna()
    unknown_count = unknown_mask.sum()
    if unknown_count > 0:
        print(f"\n⚠️  WARNING: {unknown_count} rows could not be matched — "
              f"setting diagnosis to 'Unknown'")
        unmatched_ids = merged.loc[unknown_mask, 'image_id_bare'].tolist()
        print(f"   First 10 unmatched IDs: {unmatched_ids[:10]}")
    merged['diagnosis'] = merged['diagnosis'].fillna('Unknown')

    # ── Drop temporary columns ────────────────────────────────────────────────
    cols_to_drop = ['image_id_bare', 'dx']
    # Also drop the duplicate image_id_ham column if present from the merge
    if 'image_id_ham' in merged.columns:
        cols_to_drop.append('image_id_ham')
    merged.drop(columns=cols_to_drop, inplace=True)

    # ── Backup original ───────────────────────────────────────────────────────
    print(f"\nBacking up original to: {CBIR_BACKUP_PATH}")
    shutil.copy2(CBIR_METADATA_PATH, CBIR_BACKUP_PATH)
    print("  ✓ Backup created")

    # ── Save updated CSV ──────────────────────────────────────────────────────
    print(f"\nSaving updated CSV to: {CBIR_METADATA_PATH}")
    merged.to_csv(CBIR_METADATA_PATH, index=False)
    print("  ✓ CSV saved")

    # ── Rebuild JSON ──────────────────────────────────────────────────────────
    print(f"\nSaving updated JSON to: {CBIR_JSON_PATH}")
    merged.to_json(CBIR_JSON_PATH, orient='records', indent=2)
    print("  ✓ JSON saved")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("REBUILD SUMMARY")
    print("=" * 60)
    total             = len(merged)
    successfully_labeled = (merged['diagnosis'] != 'Unknown').sum()
    print(f"  Total rows        : {total}")
    print(f"  Successfully labeled: {successfully_labeled}")
    print(f"  Unknown (join failed): {unknown_count}")
    print("\n  Class distribution:")
    for label, count in merged['diagnosis'].value_counts().items():
        pct = count / total * 100
        print(f"    {label:<25} {count:>5}  ({pct:.1f}%)")
    print("=" * 60)
    print("\n✅ cbir_metadata rebuild complete. Restart your Streamlit app.")


if __name__ == "__main__":
    main()
