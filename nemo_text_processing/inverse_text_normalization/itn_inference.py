# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lightweight ITN inference using pre-built .far files.

This module provides a minimal-dependency interface for Inverse Text Normalization
using pre-compiled FST grammars (.far files). It does NOT require joblib, sacremoses,
or other heavy dependencies - only pynini (which must be installed via conda).

Usage:
    from nemo_text_processing.inverse_text_normalization.itn_inference import ITNInference
    
    itn = ITNInference(lang='hi')  # Loads hi_itn.far from package
    result = itn.normalize("एक सौ तेईस")  # Returns "123"

Supported languages with pre-built .far files:
    - en (English)
    - hi (Hindi)
    - mr (Marathi)
    - pa (Punjabi)
    - ta (Tamil)
    - bn (Bengali)
    - ml (Malayalam)
"""

import os
from typing import List, Optional

import pynini
from pynini.lib.rewrite import top_rewrite

# Path to pre-built .far files (shipped with package)
_FAR_DIR = os.path.join(os.path.dirname(__file__), "far_files")

# Mapping of language codes to .far file names
_FAR_FILES = {
    "en": "en_itn_lower_cased.far",
    "hi": "hi_itn.far",
    "mr": "mr_itn.far",
    "pa": "pa_itn.far",
    "ta": "ta_itn.far",
    "bn": "bn_itn.far",
    "ml": "ml_itn.far",
}


class ITNInference:
    """
    Lightweight Inverse Text Normalization using pre-built .far files.
    
    This class provides minimal-dependency ITN inference. It loads pre-compiled
    FST grammars and does not require build-time dependencies like joblib or sacremoses.
    
    Args:
        lang: Language code (e.g., 'hi', 'en', 'mr', 'pa', 'ta', 'bn', 'ml')
        far_dir: Optional path to directory containing .far files.
                 Defaults to the package's far_files/ directory.
    
    Raises:
        ValueError: If language is not supported or .far file not found
    """
    
    def __init__(self, lang: str, far_dir: Optional[str] = None):
        self.lang = lang
        self.far_dir = far_dir or _FAR_DIR
        
        if lang not in _FAR_FILES:
            available = ", ".join(sorted(_FAR_FILES.keys()))
            raise ValueError(
                f"Language '{lang}' not supported. Available: {available}"
            )
        
        far_path = os.path.join(self.far_dir, _FAR_FILES[lang])
        
        if not os.path.exists(far_path):
            raise ValueError(
                f"Pre-built grammar not found: {far_path}\n"
                f"Either build the grammar using InverseNormalizer with cache_dir, "
                f"or ensure the .far file is included in the package."
            )
        
        # Load the FST from .far file
        far = pynini.Far(far_path, mode="r")
        self.tagger = far["tokenize_and_classify"]
        self.verbalizer = far["verbalize"]
        
    def normalize(self, text: str) -> str:
        """
        Normalize a single text string (inverse text normalization).
        
        Converts spoken-form text to written form, e.g.:
            "one hundred twenty three" -> "123"
            "एक सौ तेईस" -> "123"
        
        Args:
            text: Input text in spoken form
            
        Returns:
            Normalized text in written form
        """
        if not text or not text.strip():
            return text
            
        text = text.strip().lower()
        text = pynini.escape(text)
        
        try:
            # Tag
            tagged = top_rewrite(text, self.tagger)
            # Verbalize
            result = top_rewrite(tagged, self.verbalizer)
            return result
        except pynini.lib.rewrite.Error:
            # If FST can't process, return original
            return text
    
    def normalize_list(self, texts: List[str]) -> List[str]:
        """
        Normalize a list of text strings.
        
        Args:
            texts: List of input texts in spoken form
            
        Returns:
            List of normalized texts in written form
        """
        return [self.normalize(text) for text in texts]
    
    @staticmethod
    def available_languages() -> List[str]:
        """Return list of supported language codes."""
        return sorted(_FAR_FILES.keys())
    
    @classmethod
    def is_available(cls, lang: str, far_dir: Optional[str] = None) -> bool:
        """
        Check if a language has pre-built .far files available.
        
        Args:
            lang: Language code
            far_dir: Optional path to .far files directory
            
        Returns:
            True if .far file exists, False otherwise
        """
        if lang not in _FAR_FILES:
            return False
        dir_path = far_dir or _FAR_DIR
        far_path = os.path.join(dir_path, _FAR_FILES[lang])
        return os.path.exists(far_path)


# Convenience function
def inverse_normalize(text: str, lang: str = "en") -> str:
    """
    Quick inverse text normalization using pre-built grammars.
    
    Args:
        text: Input text in spoken form
        lang: Language code (default: 'en')
        
    Returns:
        Normalized text in written form
        
    Example:
        >>> inverse_normalize("one two three", lang="en")
        '123'
        >>> inverse_normalize("एक दो तीन", lang="hi")
        '123'
    """
    itn = ITNInference(lang=lang)
    return itn.normalize(text)
