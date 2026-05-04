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

from nemo_text_processing.inverse_text_normalization.ta.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_TA_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.ta.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals (Tamil)
        e.g. மைனஸ் இருபத்திமூன்று -> cardinal { integer: "௨௩" negative: "-" }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")
        self.input_case = input_case
        
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()
        graph_hundred = pynini.string_file(get_abs_path("data/numbers/hundred.tsv")).invert()
        graph_thousands = pynini.string_file(get_abs_path("data/numbers/thousands.tsv")).invert()
        
        self.graph_zero = graph_zero
        self.graph_digit = graph_digit
        self.graph_single_digit_with_zero = pynutil.insert("௦") + graph_digit
        self.graph_teens_and_ties = graph_teens_and_ties
        self.graph_two_digit = graph_teens_and_ties | (pynutil.insert("௦") + graph_digit)
        
        # Tamil words for hundred (including English and Hindi transliterations)
        delete_hundred = pynutil.delete("நூறு") | pynutil.delete("நூற்று") | pynutil.delete("ஹண்ட்ரட்") | pynutil.delete("சௌ") | pynutil.delete("சோ")
        delete_thousand = pynutil.delete("ஆயிரம்") | pynutil.delete("ஆயிரத்து") | pynutil.delete("தௌசண்ட்") | pynutil.delete("ஹஜார்")
        
        # Hundred component (100-999): e.g., "இருநூறு முப்பத்தைந்து" -> ௨௩௫
        graph_hundred_component = pynini.union(
            graph_hundred + delete_space + (self.graph_two_digit | pynutil.insert("௦௦")),
            graph_hundred,
            pynutil.insert("௦") + (self.graph_two_digit | pynutil.insert("௦௦"))
        )

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_TA_DIGIT) + (NEMO_TA_DIGIT - "௦") + pynini.closure(NEMO_TA_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )
        self.graph_hundreds = graph_hundred_component

        # Thousands component
        graph_in_thousands = pynini.union(
            graph_thousands + delete_space,
            self.graph_two_digit + delete_space + delete_thousand + delete_space,
            pynutil.insert("௦௦", weight=0.1),
        )
        self.graph_thousands_component = graph_in_thousands

        # லட்சம் (lakh - 100,000)
        delete_lakh = pynutil.delete("லட்சம்") | pynutil.delete("லட்சத்து")
        graph_in_lakhs = pynini.union(
            self.graph_two_digit + delete_space + delete_lakh + delete_space,
            pynutil.insert("௦௦", weight=0.1),
        )

        # கோடி (crore - 10,000,000)
        delete_crore = pynutil.delete("கோடி") | pynutil.delete("கோடியே")
        graph_in_crores = pynini.union(
            self.graph_two_digit + delete_space + delete_crore + delete_space,
            pynutil.insert("௦௦", weight=0.1),
        )

        # Indian numbering: crores + lakhs + thousands + hundreds
        graph_ind = (
            graph_in_crores
            + graph_in_lakhs
            + graph_in_thousands
            + graph_hundred_component
        )

        # Handle single words for powers of 10 - higher priority (negative weight)
        graph_no_prefix = pynutil.add_weight(
            pynini.cross("நூறு", "௧௦௦")
            | pynini.cross("ஆயிரம்", "௧௦௦௦")
            | pynini.cross("லட்சம்", "௧௦௦௦௦௦")
            | pynini.cross("கோடி", "௧௦௦௦௦௦௦௦"),
            -0.1,
        )

        graph = pynini.union(
            pynutil.add_weight(graph_ind, 0.1),
            graph_zero,
            graph_digit,
            self.graph_two_digit,
            graph_hundred,
            graph_no_prefix,
        )

        # Remove leading zeros except for single zero
        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("௦"))
            + pynini.difference(NEMO_TA_DIGIT, "௦")
            + pynini.closure(NEMO_TA_DIGIT),
            "௦",
        )

        self.graph_no_exception = graph
        self.graph = graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def add_tokens(self, fst):
        return pynutil.insert("cardinal { ") + fst + pynutil.insert(" }")
