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
    GraphFst,
    delete_space,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction (Tamil),
        e.g. மூன்று நான்கில் -> fraction { numerator: "௩" denominator: "௪" }
        e.g. ஒரு மூன்றில் -> fraction { integer_part: "௧" denominator: "௩" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")

        cardinal_graph = cardinal.graph
        digit_graph = cardinal.graph_digit
        
        # Tamil fractions use -இல் (il) suffix for denominator
        # E.g., நான்கில் = out of four
        
        graph_numerator = pynutil.insert("numerator: \"") + cardinal_graph + pynutil.insert("\"")
        graph_denominator = pynutil.insert(" denominator: \"") + cardinal_graph + pynutil.insert("\"")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        # Delete Tamil fraction marker -இல் (il) or related words
        delete_fraction_marker = pynutil.delete("இல்") | pynutil.delete("இல்ஒன்று")

        # numerator denominator form: "மூன்று நான்கில்"
        graph_frac = (
            graph_numerator
            + delete_space
            + graph_denominator
            + delete_fraction_marker
        )

        # Common fractions
        # அரை (arai) = half = 1/2
        # கால் (kaal) = quarter = 1/4
        # முக்கால் (mukkaal) = three-quarter = 3/4
        graph_half = pynutil.delete("அரை") + pynutil.insert("numerator: \"௧\" denominator: \"௨\"")
        graph_quarter = pynutil.delete("கால்") + pynutil.insert("numerator: \"௧\" denominator: \"௪\"")
        graph_three_quarter = pynutil.delete("முக்கால்") + pynutil.insert("numerator: \"௩\" denominator: \"௪\"")
        
        graph = graph_frac | graph_half | graph_quarter | graph_three_quarter

        self.final_graph = self.add_tokens(graph)
        self.fst = self.final_graph.optimize()

    def add_tokens(self, fst):
        return pynutil.insert("fraction { ") + fst + pynutil.insert(" }")
