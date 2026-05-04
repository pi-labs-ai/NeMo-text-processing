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

from nemo_text_processing.inverse_text_normalization.pa.graph_utils import (
    NEMO_CHAR,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.pa.utils import get_abs_path

shunya = (
    pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
    | pynini.string_file(get_abs_path("data/telephone/eng_zero.tsv")).invert()
)
digit_without_shunya = (
    pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
    | pynini.string_file(get_abs_path("data/telephone/eng_digit.tsv")).invert()
)
digit = digit_without_shunya | shunya


def get_context(keywords: list):
    keywords = pynini.union(*keywords)

    # Load Punjabi digits from TSV files
    punjabi_digits = (
        pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
    ).project("output")

    # Load English digits from TSV files
    english_digits = (
        pynini.string_file(get_abs_path("data/telephone/eng_digit.tsv"))
        | pynini.string_file(get_abs_path("data/telephone/eng_zero.tsv"))
    ).project("output")

    all_digits = punjabi_digits | english_digits

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

    # Punjabi word for "plus" - ਪਲੱਸ
    country_code = pynini.cross("ਪਲੱਸ", "+") + pynini.closure(delete_space + digit, 2, 2) + NEMO_WHITE_SPACE
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
        + pynutil.insert("\" ")
    )

    graph_mobile = (graph_country_code + graph_number) | graph_number

    return graph_mobile


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone (Punjabi)
        e.g. ਮੋਬਾਈਲ ਨੰਬਰ ਇੱਕ ਦੋ ਤਿੰਨ ਚਾਰ ਪੰਜ ਛੇ ਸੱਤ ਅੱਠ ਨੌਂ ਸਿਫਰ -> 
             telephone { number_part: "੧੨੩੪੫੬੭੮੯੦" }
    
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")

        # Load context keywords
        context_keywords_mobile = ["ਮੋਬਾਈਲ", "ਫੋਨ", "ਨੰਬਰ", "ਕਾਲ"]
        context_keywords_landline = ["ਲੈਂਡਲਾਈਨ", "ਫੋਨ", "ਨੰਬਰ"]
        context_keywords_pincode = ["ਪਿੰਨ", "ਕੋਡ", "ਪਿੰਨਕੋਡ"]
        context_keywords_credit = ["ਕਾਰਡ", "ਕ੍ਰੈਡਿਟ", "ਡੈਬਿਟ"]

        graph_mobile = generate_mobile(context_keywords_mobile)
        graph_pincode = generate_pincode(context_keywords_pincode)
        graph_credit = generate_credit(context_keywords_credit)

        graph = pynutil.add_weight(graph_mobile, 0.1) | pynutil.add_weight(graph_pincode, 0.1) | pynutil.add_weight(graph_credit, 0.1)

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
