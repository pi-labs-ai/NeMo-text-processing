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

from nemo_text_processing.inverse_text_normalization.ml.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_ML_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.ml.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals (Malayalam)
        e.g. മൈനസ് ഇരുപത്തിമൂന്ന് -> cardinal { integer: "൨൩" negative: "-" }

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
        self.graph_single_digit_with_zero = pynutil.insert("൦") + graph_digit
        self.graph_teens_and_ties = graph_teens_and_ties
        self.graph_two_digit = graph_teens_and_ties | (pynutil.insert("൦") + graph_digit)
        
        # Malayalam words for hundred
        delete_hundred = pynutil.delete("നൂറ്") | pynutil.delete("നൂറു")
        delete_thousand = pynutil.delete("ആയിരം") | pynutil.delete("ആയിരത്തി")
        
        # Hundred component (100-999): e.g., "ഇരുനൂറ് മുപ്പത്തഞ്ച്" -> ൨൩൫
        graph_hundred_component = pynini.union(
            graph_hundred + delete_space + (self.graph_two_digit | pynutil.insert("൦൦")),
            graph_hundred,
            pynutil.insert("൦") + (self.graph_two_digit | pynutil.insert("൦൦"))
        )

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_ML_DIGIT) + (NEMO_ML_DIGIT - "൦") + pynini.closure(NEMO_ML_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )
        self.graph_hundreds = graph_hundred_component

        # Thousands component
        graph_in_thousands = pynini.union(
            graph_thousands + delete_space,
            self.graph_two_digit + delete_space + delete_thousand + delete_space,
            pynutil.insert("൦൦", weight=0.1),
        )
        self.graph_thousands_component = graph_in_thousands

        # ലക്ഷം (lakh - 100,000)
        delete_lakh = pynutil.delete("ലക്ഷം") | pynutil.delete("ലക്ഷത്തി")
        graph_in_lakhs = pynini.union(
            self.graph_two_digit + delete_space + delete_lakh + delete_space,
            pynutil.insert("൦൦", weight=0.1),
        )

        # കോടി (crore - 10,000,000)
        delete_crore = pynutil.delete("കോടി") | pynutil.delete("കോടിയും")
        graph_in_crores = pynini.union(
            self.graph_two_digit + delete_space + delete_crore + delete_space,
            pynutil.insert("൦൦", weight=0.1),
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
            pynini.cross("നൂറ്", "൧൦൦")
            | pynini.cross("നൂറു", "൧൦൦")
            | pynini.cross("ആയിരം", "൧൦൦൦")
            | pynini.cross("ലക്ഷം", "൧൦൦൦൦൦")
            | pynini.cross("കോടി", "൧൦൦൦൦൦൦൦"),
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
            pynutil.delete(pynini.closure("൦"))
            + pynini.difference(NEMO_ML_DIGIT, "൦")
            + pynini.closure(NEMO_ML_DIGIT),
            "൦",
        )

        self.graph_no_exception = graph
        self.graph = graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
