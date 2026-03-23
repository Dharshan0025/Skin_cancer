import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os

class CBIREngine:
    def __init__(self):
        self.features_path = "models/cbir_features.npy"
        self.metadata_path = "models/cbir_metadata.csv"
        self._base_model = None
        self.load_index()

    def load_index(self):
        """Load precomputed features + metadata"""
        if os.path.exists(self.features_path) and os.path.exists(self.metadata_path):
            self.features = np.load(self.features_path)
            self.df = pd.read_csv(self.metadata_path)
            st.success(f"✅ CBIR loaded: {len(self.df)} cases")
            return True
        return False

    def _get_base_model(self):
        """Lazy-load DenseNet201 backbone (matches cbir_features.npy at 1920-dim)"""
        if self._base_model is None:
            from tensorflow.keras.applications import DenseNet201
            self._base_model = DenseNet201(
                weights='imagenet', include_top=False, pooling='avg'
            )
            self._base_model.trainable = False
        return self._base_model

    def extract_features(self, image):
        """Extract DenseNet121 features — bypasses Lambda layer entirely"""
        base_model = self._get_base_model()
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        features = base_model.predict(img_array, verbose=0).flatten()
        return features

    def find_similar(self, image, top_k=5):
        """Find top-k most similar cases using real matched image paths"""
        if not hasattr(self, 'features'):
            return []

        query_features = self.extract_features(image)
        similarities = cosine_similarity([query_features], self.features)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                'similarity': float(similarities[idx]),
                'diagnosis': row.get('diagnosis', 'HAM10000'),
                'image_id': row.get('image_id', f'HAM{idx}'),
                'path': row.get('image_path', None)
            })
        return results


# Global engine (singleton)
cbir = None

def init_cbir():
    global cbir
    if cbir is None:
        cbir = CBIREngine()
    return cbir
