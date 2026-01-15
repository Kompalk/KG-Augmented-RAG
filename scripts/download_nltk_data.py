#!/usr/bin/env python3
"""Download required NLTK data with SSL certificate fix"""
import ssl
import nltk

# Fix SSL certificate issue (common on macOS)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
print("Downloading NLTK data...")
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Warning: NLTK data download failed: {e}")
    print("   The system may still work, but some features may be limited.")
