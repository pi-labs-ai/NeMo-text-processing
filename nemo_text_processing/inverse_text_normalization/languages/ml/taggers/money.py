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
    NEMO_ML_DIGIT,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.ml.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money (Malayalam)
        e.g. "അഞ്ഞൂറ് രൂപ" -> money { integer_part: "൫൦൦" currency: "₹" }
        e.g. "അഞ്ഞൂറ് രൂപ അമ്പത് പൈസ" -> money { integer_part: "൫൦൦" fractional_part: "൫൦" currency: "₹" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst = None):
        super().__init__(name="money", kind="classify")

        cardinal_graph = cardinal.graph

        # Currency graph - invert to get word -> symbol mapping for ITN
        currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv")).invert()
        # Fractional units graph
        cents_graph = pynini.string_file(get_abs_path("data/money/cents.tsv")).invert()

        # Integer part with currency
        integer_part = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        
        # Fractional part (paise, cents, etc.)
        fractional_part = (
            pynutil.insert(" fractional_part: \"") + cardinal_graph + pynutil.insert("\"")
        )

        # Currency symbol
        currency_symbol = (
            pynutil.insert(" currency: \"") + currency_graph + pynutil.insert("\"")
        )

        # Optional fractional part
        optional_fractional = pynini.closure(
            delete_space + fractional_part + delete_space + pynutil.delete(cents_graph),
            0,
            1,
        )

        # Main graph: amount + currency_word -> integer_part + currency
        # e.g., "അഞ്ഞൂറ് രൂപ" -> integer_part: "൫൦൦" currency: "₹"
        graph = integer_part + delete_space + currency_symbol + optional_fractional

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
