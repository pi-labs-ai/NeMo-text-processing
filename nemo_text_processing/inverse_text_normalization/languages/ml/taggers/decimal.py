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


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimals (Malayalam)
        e.g. "രണ്ട് പോയിന്റ് അഞ്ച്" -> decimal { integer_part: "൨" fractional_part: "൫" }
        e.g. "മൈനസ് മൂന്ന് പോയിന്റ് ഒന്ന് നാല്" -> decimal { negative: "-" integer_part: "൩" fractional_part: "൧൪" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph
        digit_graph = cardinal.graph_digit
        graph_zero = cardinal.graph_zero

        # Point - "പോയിന്റ്" or "ദശാംശം" (decimal)
        delete_point = pynutil.delete("പോയിന്റ്") | pynutil.delete("ദശാംശം") | pynutil.delete("പോയന്റ്")

        # Fractional part - sequence of digits after decimal point
        graph_fractional_part = pynini.closure(
            delete_space + (digit_graph | graph_zero), 1
        )

        # Integer part
        graph_integer_part = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        # Fractional part
        graph_fractional = (
            pynutil.insert(" fractional_part: \"") + graph_fractional_part + pynutil.insert("\"")
        )

        # Optional negative
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: \"") + pynini.cross("മൈനസ്", "-") + pynutil.insert("\"") + delete_space,
            0,
            1,
        )

        graph = optional_graph_negative + graph_integer_part + delete_space + delete_point + graph_fractional

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
