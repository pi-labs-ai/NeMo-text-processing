# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2024 and onwards Google, Inc.
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
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_PA_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.pa.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals (Punjabi)
        e.g. ਰਿਣ ਤੇਈ -> cardinal { integer: "੨੩" negative: "-" }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")
        self.input_case = input_case
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()
        graph_paune = pynini.string_file(get_abs_path("data/numbers/paune.tsv")).invert()
        self.graph_zero = graph_zero
        self.graph_digit = graph_digit
        self.graph_single_digit_with_zero = pynutil.insert("੦") + graph_digit
        self.graph_teens_and_ties = graph_teens_and_ties
        self.graph_two_digit = graph_teens_and_ties | (pynutil.insert("੦") + graph_digit)
        
        # Punjabi words for hundred and thousand
        graph_hundred = pynini.cross("ਸੌ", "") | pynini.cross("ਹੰਡਰੈੱਡ", "") | pynini.cross("ਸੋ", "")
        delete_hundred = pynutil.delete("ਸੌ") | pynutil.delete("ਹੰਡਰੈੱਡ") | pynutil.delete("ਸੋ")
        delete_thousand = pynutil.delete("ਹਜ਼ਾਰ") | pynutil.delete("ਹਜਾਰ") | pynutil.delete("ਥਾਊਜ਼ੈਂਡ") | pynutil.delete("ਥਾਉਜ਼ੈਂਡ")
        
        graph_hundred_component = pynini.union(graph_digit + delete_space + graph_hundred, pynutil.insert("੦"))
        graph_hundred_component += delete_space
        graph_hundred_component += self.graph_two_digit | pynutil.insert("੦੦")

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_PA_DIGIT) + (NEMO_PA_DIGIT - "੦") + pynini.closure(NEMO_PA_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        # Transducer for eleven hundred -> 1100 or twenty one hundred eleven -> 2111
        graph_hundred_as_thousand = pynini.union(
            graph_teens_and_ties + delete_space + graph_hundred, pynutil.insert("੦")
        )
        graph_hundred_as_thousand += delete_space
        graph_hundred_as_thousand += self.graph_two_digit | pynutil.insert("੦੦")
        
        # ਸਾਢੇ (saadhey) - one and a half
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("ਸਾਢੇ")
            + delete_space
            + graph_digit
            + pynutil.insert("੫੦੦", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )
        # ਸਵਾ (sava) - one and a quarter
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("ਸਵਾ")
            + delete_space
            + graph_digit
            + pynutil.insert("੨੫੦", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )
        # ਪੌਣੇ (paune) - three-quarters
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("ਪੌਣੇ")
            + delete_space
            + graph_paune
            + pynutil.insert("੭੫੦", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )
        # ਡੇਢ (dedh) - 1.5
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynini.union(pynutil.delete("ਡੇਢ") | pynutil.delete("ਡੇੜ"))
            + delete_space
            + pynutil.insert("੧੫੦੦", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )
        # ਢਾਈ (dhaai) - 2.5
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("ਢਾਈ")
            + delete_space
            + pynutil.insert("੨੫੦੦", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )

        graph_in_hundreds = pynutil.add_weight(
            pynutil.delete("ਸਾਢੇ")
            + delete_space
            + (graph_digit | self.graph_two_digit)
            + pynutil.insert("੫੦", weight=-0.1)
            + delete_space
            + delete_hundred,
            -0.1,
        )
        graph_in_hundreds |= pynutil.add_weight(
            pynutil.delete("ਸਵਾ")
            + delete_space
            + (graph_digit | self.graph_two_digit)
            + pynutil.insert("੨੫", weight=-0.1)
            + delete_space
            + delete_hundred,
            -0.1,
        )
        graph_in_hundreds |= pynutil.add_weight(
            pynutil.delete("ਪੌਣੇ")
            + delete_space
            + graph_paune
            + pynutil.insert("੭੫", weight=-0.1)
            + delete_space
            + delete_hundred,
            -0.1,
        )
        graph_in_hundreds |= pynutil.add_weight(
            pynini.union(pynutil.delete("ਡੇਢ") | pynutil.delete("ਡੇੜ"))
            + delete_space
            + pynutil.insert("੧੫੦", weight=-0.1)
            + delete_space
            + delete_hundred,
            -0.1,
        )
        graph_in_hundreds |= pynutil.add_weight(
            pynutil.delete("ਢਾਈ") + delete_space + pynutil.insert("੨੫੦", weight=-0.1) + delete_space + delete_hundred,
            -0.1,
        )
        self.graph_hundreds = graph_hundred_component | graph_hundred_as_thousand | graph_in_hundreds

        graph_teens_and_ties_component = pynini.union(
            graph_teens_and_ties | pynutil.insert("੦੦") + delete_space + (graph_digit | pynutil.insert("੦")),
        )
        graph_ties_component_at_least_one_none_zero_digit = self.graph_two_digit @ (
            pynini.closure(NEMO_PA_DIGIT) + pynini.closure(NEMO_PA_DIGIT)
        )
        self.graph_ties_component_at_least_one_none_zero_digit = graph_ties_component_at_least_one_none_zero_digit

        # Indian numeric format - https://en.wikipedia.org/wiki/Indian_numbering_system
        graph_in_thousands = pynini.union(
            self.graph_two_digit + delete_space + delete_thousand,
            pynutil.insert("੦੦", weight=0.1),
        )
        self.graph_thousands = graph_in_thousands

        # ਲੱਖ (lakh)
        graph_in_lakhs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("ਲੱਖ"),
            pynutil.insert("੦੦", weight=0.1),
        )

        # ਕਰੋੜ (crore)
        graph_in_crores = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("ਕਰੋੜ") | pynutil.delete("crores"),
            pynutil.insert("੦੦", weight=0.1),
        )

        # ਅਰਬ (arab)
        graph_in_arabs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("ਅਰਬ"),
            pynutil.insert("੦੦", weight=0.1),
        )

        # ਖਰਬ (kharab)
        graph_in_kharabs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("ਖਰਬ"),
            pynutil.insert("੦੦", weight=0.1),
        )

        # ਨੀਲ (neel)
        graph_in_nils = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("ਨੀਲ"),
            pynutil.insert("੦੦", weight=0.1),
        )

        # ਪਦਮ (padam)
        graph_in_padmas = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("ਪਦਮ"),
            pynutil.insert("੦੦", weight=0.1),
        )

        # ਸ਼ੰਖ (shankh)
        graph_in_shankhs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("ਸ਼ੰਖ"),
            pynutil.insert("੦੦", weight=0.1),
        )

        graph_ind = (
            graph_in_shankhs
            + delete_space
            + graph_in_padmas
            + delete_space
            + graph_in_nils
            + delete_space
            + graph_in_kharabs
            + delete_space
            + graph_in_arabs
            + delete_space
            + graph_in_crores
            + delete_space
            + graph_in_lakhs
            + delete_space
            + graph_in_thousands
        )
        graph_no_prefix = pynutil.add_weight(
            pynini.cross("ਸੌ", "੧੦੦")
            | pynini.cross("ਹਜ਼ਾਰ", "੧੦੦੦")
            | pynini.cross("ਹਜਾਰ", "੧੦੦੦")
            | pynini.cross("ਲੱਖ", "੧੦੦੦੦੦")
            | pynini.cross("ਕਰੋੜ", "੧੦੦੦੦੦੦੦"),
            2,
        )

        graph = pynini.union(
            graph_ind + delete_space + self.graph_hundreds, graph_zero, graph_no_prefix
        )

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("੦"))
            + pynini.difference(NEMO_PA_DIGIT, "੦")
            + pynini.closure(NEMO_PA_DIGIT),
            "੦",
        )

        labels_exception = [pynini.string_file(get_abs_path("data/numbers/labels_exception.tsv"))]

        graph_exception = pynini.union(*labels_exception).optimize()

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
