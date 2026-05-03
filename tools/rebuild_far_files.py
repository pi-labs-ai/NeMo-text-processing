"""
Rebuilds all ITN .far files to include BOTH 'tokenize_and_classify' and 'verbalize' FSTs.

The existing .far files (built via InverseNormalizer with cache_dir) only contain
'tokenize_and_classify'. ITNInference (itn_inference.py) requires both keys.
This script regenerates all supported language .far files with both keys.

Usage:
    conda run -n nemo_tn python3 tools/rebuild_far_files.py
"""

import os
import sys

FAR_DIR = os.path.join(
    os.path.dirname(__file__),
    "../nemo_text_processing/inverse_text_normalization/far_files"
)

LANGUAGES = {
    "hi": "hi_itn.far",
    "mr": "mr_itn.far",
    "pa": "pa_itn.far",
    "ta": "ta_itn.far",
    "bn": "bn_itn.far",
    "ml": "ml_itn.far",
    "en": "en_itn_lower_cased.far",
    "zh": "zh_itn.far",
}

# Map lang -> (ClassifyFst module, VerbalizeFinalFst module)
LANG_MODULES = {
    "hi": (
        "nemo_text_processing.inverse_text_normalization.hi.taggers.tokenize_and_classify",
        "nemo_text_processing.inverse_text_normalization.hi.verbalizers.verbalize_final",
    ),
    "mr": (
        "nemo_text_processing.inverse_text_normalization.mr.taggers.tokenize_and_classify",
        "nemo_text_processing.inverse_text_normalization.mr.verbalizers.verbalize_final",
    ),
    "pa": (
        "nemo_text_processing.inverse_text_normalization.pa.taggers.tokenize_and_classify",
        "nemo_text_processing.inverse_text_normalization.pa.verbalizers.verbalize_final",
    ),
    "ta": (
        "nemo_text_processing.inverse_text_normalization.ta.taggers.tokenize_and_classify",
        "nemo_text_processing.inverse_text_normalization.ta.verbalizers.verbalize_final",
    ),
    "bn": (
        "nemo_text_processing.inverse_text_normalization.bn.taggers.tokenize_and_classify",
        "nemo_text_processing.inverse_text_normalization.bn.verbalizers.verbalize_final",
    ),
    "ml": (
        "nemo_text_processing.inverse_text_normalization.ml.taggers.tokenize_and_classify",
        "nemo_text_processing.inverse_text_normalization.ml.verbalizers.verbalize_final",
    ),
    "en": (
        "nemo_text_processing.inverse_text_normalization.en.taggers.tokenize_and_classify",
        "nemo_text_processing.inverse_text_normalization.en.verbalizers.verbalize_final",
    ),
    "zh": (
        "nemo_text_processing.inverse_text_normalization.zh.taggers.tokenize_and_classify",
        "nemo_text_processing.inverse_text_normalization.zh.verbalizers.verbalize_final",
    ),
}


def rebuild(lang: str, far_filename: str):
    import importlib
    from nemo_text_processing.text_normalization.en.graph_utils import generator_main

    far_path = os.path.join(FAR_DIR, far_filename)
    tagger_mod, verbalizer_mod = LANG_MODULES[lang]

    print(f"\n[{lang}] Building tagger...")
    ClassifyFst = importlib.import_module(tagger_mod).ClassifyFst
    # Build tagger without cache so we get a fresh FST
    tagger = ClassifyFst(cache_dir=None, overwrite_cache=True, input_case="lower_cased")

    print(f"[{lang}] Building verbalizer...")
    VerbalizeFinalFst = importlib.import_module(verbalizer_mod).VerbalizeFinalFst
    verbalizer = VerbalizeFinalFst()

    print(f"[{lang}] Saving to {far_path} ...")
    generator_main(far_path, {
        "tokenize_and_classify": tagger.fst,
        "verbalize": verbalizer.fst,
    })
    print(f"[{lang}] Done -> {far_path}")


if __name__ == "__main__":
    os.makedirs(FAR_DIR, exist_ok=True)

    langs = sys.argv[1:] if len(sys.argv) > 1 else list(LANGUAGES.keys())
    print(f"Rebuilding .far files for: {langs}")
    print(f"Output dir: {os.path.abspath(FAR_DIR)}\n")

    for lang in langs:
        if lang not in LANGUAGES:
            print(f"[SKIP] Unknown language: {lang}")
            continue
        try:
            rebuild(lang, LANGUAGES[lang])
        except Exception as e:
            print(f"[ERROR] {lang}: {e}")
            import traceback
            traceback.print_exc()
