# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.bn.graph_utils import (
    NEMO_CHAR,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.bn.utils import get_abs_path

shunya = (
    pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
    | pynini.string_file(get_abs_path("data/telephone/eng_zero.tsv")).invert()
)
digit_without_shunya = (
    pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
    | pynini.string_file(get_abs_path("data/telephone/eng_digit.tsv")).invert()
)
digit = digit_without_shunya | shunya

# Two-digit Bengali numbers (10-99) for Aadhaar patterns
two_digit = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()
# Compound hundreds (100-999) for Aadhaar patterns like "দুই হাজার একশো সাতাত্তর"
compound_hundreds = pynini.string_file(get_abs_path("data/numbers/compound_hundreds.tsv")).invert()


def get_context(keywords: list):
    keywords = pynini.union(*keywords)

    # Load Bengali digits from TSV files
    bengali_digits = (
        pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
    ).project("output")

    # Load English digits from TSV files
    english_digits = (
        pynini.string_file(get_abs_path("data/telephone/eng_digit.tsv"))
        | pynini.string_file(get_abs_path("data/telephone/eng_zero.tsv"))
    ).project("output")

    all_digits = bengali_digits | english_digits

    non_digit_char = pynini.difference(NEMO_CHAR, pynini.union(all_digits, NEMO_WHITE_SPACE))
    word = pynini.closure(non_digit_char, 1) + NEMO_WHITE_SPACE
    window = pynini.closure(word, 0, 5)
    before = (keywords + window).optimize()
    after = (window + keywords).optimize()

    return before, after


def generate_context_graph(context_keywords, length):
    context_before, context_after = get_context(context_keywords)
    digits = pynini.closure(digit + delete_space, length - 1, length - 1) + digit

    graph_after_context = digits + NEMO_WHITE_SPACE + context_after
    graph_before_context = context_before + NEMO_WHITE_SPACE + digits
    graph_without_context = digits

    return (
        pynutil.insert("number_part: \"")
        + (graph_before_context | graph_after_context | graph_without_context)
        + pynutil.insert("\" ")
    ).optimize()


def generate_pincode(context_keywords):
    return generate_context_graph(context_keywords, 6)


def generate_credit(context_keywords):
    return generate_context_graph(context_keywords, 4)


def generate_mobile(context_keywords):
    context_before, context_after = get_context(context_keywords)

    # Bengali: প্লাস (plus) for "+"
    country_code = pynini.cross("প্লাস", "+") + pynini.closure(delete_space + digit, 2, 2) + NEMO_WHITE_SPACE
    graph_country_code = (
        pynutil.insert("country_code: \"")
        + (context_before + NEMO_WHITE_SPACE) ** (0, 1)
        + country_code
        + pynutil.insert("\" ")
    )

    number_part = digit_without_shunya + delete_space + pynini.closure(digit + delete_space, 8, 8) + digit
    graph_number = (
        pynutil.insert("number_part: \"")
        + number_part
        + pynini.closure(NEMO_WHITE_SPACE + context_after, 0, 1)
        + pynutil.insert("\" ")
    )

    graph = (graph_country_code + graph_number) | graph_number
    return graph.optimize()


def generate_telephone(context_keywords):
    context_before, context_after = get_context(context_keywords)

    landline = shunya + delete_space + pynini.closure(digit + delete_space, 9, 9) + digit
    landline_with_context_before = context_before + NEMO_WHITE_SPACE + landline
    landline_with_context_after = landline + NEMO_WHITE_SPACE + context_after

    return (
        pynutil.insert("number_part: \"")
        + (landline | landline_with_context_before | landline_with_context_after)
        + pynutil.insert("\" ")
    )


def generate_aadhaar(context_keywords, cardinal):
    """
    Generate Aadhaar number pattern - 12 digits spoken as pairs of 2-digit numbers
    e.g. একষট্টি চুয়াল্লিশ চুয়াত্তর সাতাশ দুই হাজার একশো সাতাত্তর -> ৬১৪৪৭৪২৭২১৭৭
    """
    context_before, context_after = get_context(context_keywords)
    
    # Two-digit numbers (10-99) from teens_and_ties or single digits with zero padding
    two_digit_num = two_digit | (pynutil.insert("০") + digit_without_shunya)
    
    # 4-digit patterns for Aadhaar:
    # 1. Colloquial: "দুই একশো সাতাত্তর" -> ২১৭৭
    # 2. With হাজার: "দুই হাজার একশো সাতাত্তর" -> ২১৭৭
    # 3. Simple year-like: "দুই হাজার ছাব্বিশ" -> ২০২৬
    four_digit_colloquial = cardinal.graph_four_digit_colloquial
    
    # Pattern with হাজার: digit + হাজার + compound_hundred + two_digit
    delete_hajar = pynutil.delete("হাজার")
    four_digit_with_hajar = (
        (digit_without_shunya | pynutil.insert("০")) + delete_space + 
        delete_hajar + delete_space +
        cardinal.graph_compound_hundred_with_two_digit
    )
    
    # Year-like pattern: "দুই হাজার ছাব্বিশ" -> ২০২৬ (digit + হাজার + two_digit)
    four_digit_year_like = (
        (digit_without_shunya | pynutil.insert("০")) + delete_space +
        delete_hajar + delete_space +
        pynutil.insert("০") + two_digit
    )
    
    four_digit_all = four_digit_colloquial | four_digit_with_hajar | four_digit_year_like
    
    # Pattern: 4 two-digit numbers + 1 four-digit number (for 12 digits total)
    # e.g., 61 44 74 27 + 2177 = ৬১৪৪৭৪২৭২১৭৭
    aadhaar_4_plus_4 = (
        two_digit_num + delete_space +
        two_digit_num + delete_space +
        two_digit_num + delete_space +
        two_digit_num + delete_space +
        four_digit_all
    )
    
    # Pattern: All 6 two-digit pairs (for 12 digits total)
    aadhaar_all_pairs = pynini.closure(two_digit_num + delete_space, 5, 5) + two_digit_num
    
    aadhaar_number = aadhaar_4_plus_4 | aadhaar_all_pairs
    
    # With context (like "আধার কার্ড নম্বর")
    aadhaar_with_context_before = context_before + NEMO_WHITE_SPACE + aadhaar_number
    aadhaar_with_context_after = aadhaar_number + NEMO_WHITE_SPACE + context_after
    
    return (
        pynutil.insert("number_part: \"")
        + (aadhaar_number | aadhaar_with_context_before | aadhaar_with_context_after)
        + pynutil.insert("\" ")
    ).optimize()


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone
        e.g. প্লাস নয় এক ৯৮৭৬৫৪৩২১০ -> telephone { country_code: "+৯১" number_part: "৯৮৭৬৫৪৩২১০" }
        e.g. একষট্টি চুয়াল্লিশ চুয়াত্তর সাতাশ দুই হাজার একশো সাতাত্তর -> telephone { number_part: "৬১৪৪৭৪২৭২১৭৭" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")

        context_cue = pynini.string_file(get_abs_path("data/telephone/context_cues.tsv"))
        context_keywords = list(context_cue.paths().ostrings())

        graph_mobile = generate_mobile(context_keywords)
        graph_telephone = generate_telephone(context_keywords)
        graph_aadhaar = generate_aadhaar(context_keywords, cardinal)

        final_graph = self.add_tokens(graph_mobile | graph_telephone | graph_aadhaar)
        self.fst = final_graph.optimize()
