#!/usr/bin/env python
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
Build script to export combined .far files (tagger + verbalizer) for Indic languages.

This creates self-contained .far files that can be used for inference without
needing to rebuild grammars at runtime.

Usage:
    python build_grammars.py                    # Build all supported languages
    python build_grammars.py --langs hi mr pa   # Build specific languages
    python build_grammars.py --output-dir /path/to/output  # Custom output directory

The output .far files contain both 'tokenize_and_classify' and 'verbalize' FSTs,
enabling full inference with just the .far file.
"""

import argparse
import os
import sys
from typing import List

import pynini

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generator_main(far_file: str, graphs: dict):
    """Export FST graphs to a .far file."""
    exporter = pynini.export.Exporter(far_file)
    for name, graph in graphs.items():
        exporter[name] = graph.optimize()
    exporter.close()
    print(f"  Exported: {far_file}")


def build_language(lang: str, output_dir: str) -> bool:
    """
    Build combined .far file for a single language.
    
    Args:
        lang: Language code
        output_dir: Directory to save .far files
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\nBuilding {lang}...")
    
    try:
        if lang == 'en':
            from nemo_text_processing.inverse_text_normalization.en.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.en.verbalizers.verbalize_final import VerbalizeFinalFst
            far_name = "en_itn.far"
        elif lang == 'hi':
            from nemo_text_processing.inverse_text_normalization.hi.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.hi.verbalizers.verbalize_final import VerbalizeFinalFst
            far_name = "hi_itn.far"
        elif lang == 'mr':
            from nemo_text_processing.inverse_text_normalization.mr.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.mr.verbalizers.verbalize_final import VerbalizeFinalFst
            far_name = "mr_itn.far"
        elif lang == 'pa':
            from nemo_text_processing.inverse_text_normalization.pa.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.pa.verbalizers.verbalize_final import VerbalizeFinalFst
            far_name = "pa_itn.far"
        elif lang == 'ta':
            from nemo_text_processing.inverse_text_normalization.ta.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.ta.verbalizers.verbalize_final import VerbalizeFinalFst
            far_name = "ta_itn.far"
        elif lang == 'bn':
            from nemo_text_processing.inverse_text_normalization.bn.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.bn.verbalizers.verbalize_final import VerbalizeFinalFst
            far_name = "bn_itn.far"
        elif lang == 'ml':
            from nemo_text_processing.inverse_text_normalization.ml.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.ml.verbalizers.verbalize_final import VerbalizeFinalFst
            far_name = "ml_itn.far"
        else:
            print(f"  ERROR: Language '{lang}' not supported")
            return False
        
        # Build tagger (without cache to get fresh FST)
        print(f"  Building tagger...")
        tagger = ClassifyFst(cache_dir=None)
        
        # Build verbalizer
        print(f"  Building verbalizer...")
        verbalizer = VerbalizeFinalFst()
        
        # Export combined .far file
        far_path = os.path.join(output_dir, far_name)
        generator_main(far_path, {
            "tokenize_and_classify": tagger.fst,
            "verbalize": verbalizer.fst,
        })
        
        # Print file size
        size_mb = os.path.getsize(far_path) / (1024 * 1024)
        print(f"  ✓ {lang} complete ({size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"  ERROR building {lang}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build combined .far grammar files for ITN inference"
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en", "hi", "mr", "pa", "ta", "bn", "ml"],
        help="Languages to build (default: all Indic + English)"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "nemo_text_processing/inverse_text_normalization/far_files"
        ),
        help="Output directory for .far files"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Build each language
    success = []
    failed = []
    
    for lang in args.langs:
        if build_language(lang, args.output_dir):
            success.append(lang)
        else:
            failed.append(lang)
    
    # Summary
    print("\n" + "=" * 50)
    print("BUILD SUMMARY")
    print("=" * 50)
    print(f"Successful: {', '.join(success) if success else 'None'}")
    print(f"Failed: {', '.join(failed) if failed else 'None'}")
    print(f"\nOutput directory: {args.output_dir}")
    
    if success:
        print("\nFiles created:")
        for f in sorted(os.listdir(args.output_dir)):
            if f.endswith('.far'):
                path = os.path.join(args.output_dir, f)
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"  {f}: {size:.2f} MB")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
