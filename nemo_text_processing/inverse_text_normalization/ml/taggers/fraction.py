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


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions (Malayalam)
        e.g. "രണ്ട് അംശം മൂന്ന്" -> fraction { numerator: "൨" denominator: "൩" }
        e.g. "ഒന്ന് മൂന്നിൽ" -> fraction { numerator: "൧" denominator: "൩" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")

        cardinal_graph = cardinal.graph

        # Words for "by", "over" or "divided by" in fractions
        # "അംശം" means "part of" / "fraction of"
        # "ഇൽ" suffix means "in" or "out of"
        delete_fraction_word = (
            pynutil.delete("അംശം")
            | pynutil.delete("ൽ")
            | pynutil.delete("ഇൽ")
            | pynutil.delete("വിഭജിച്ച്")
            | pynutil.delete("മേൽ")
        )

        # Numerator
        graph_numerator = pynutil.insert("numerator: \"") + cardinal_graph + pynutil.insert("\"")

        # Denominator
        graph_denominator = pynutil.insert(" denominator: \"") + cardinal_graph + pynutil.insert("\"")

        # Standard format: num + "അംശം" + denom
        graph = graph_numerator + delete_space + delete_fraction_word + delete_space + graph_denominator

        # Common fractions - half, quarter, etc.
        graph_half = pynini.cross("അര", "numerator: \"൧\" denominator: \"൨\"")
        graph_quarter = pynini.cross("കാൽ", "numerator: \"൧\" denominator: \"൪\"")
        graph_three_quarter = pynini.cross("മുക്കാൽ", "numerator: \"൩\" denominator: \"൪\"")

        graph = graph | graph_half | graph_quarter | graph_three_quarter

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
