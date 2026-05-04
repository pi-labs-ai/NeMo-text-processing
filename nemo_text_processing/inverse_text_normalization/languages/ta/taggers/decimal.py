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
    insert_space,
)


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal (Tamil),
        e.g. மைனஸ் பன்னிரண்டு புள்ளி ஐந்து ஒன்று இரண்டு -> decimal { negative: "true" integer_part: "௧௨" fractional_part: "௫௧௨" }
        e.g. பன்னிரண்டு புள்ளி ஐந்து ஒன்று இரண்டு -> decimal { integer_part: "௧௨" fractional_part: "௫௧௨" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        graph_cardinal = cardinal.graph
        graph_digit = cardinal.graph_digit | cardinal.graph_zero
        
        graph_integer = pynutil.insert("integer_part: \"") + graph_cardinal + pynutil.insert("\"")

        # Tamil word for decimal point - புள்ளி (pulli)
        delete_point = pynutil.delete("புள்ளி")
        
        # Fractional part is a series of digits read one by one
        graph_fractional_part = pynini.closure(graph_digit + delete_space, 1) + graph_digit
        graph_fractional = pynutil.insert(" fractional_part: \"") + graph_fractional_part + pynutil.insert("\"")

        graph_decimal = graph_integer + delete_space + delete_point + delete_space + graph_fractional

        # Tamil word for minus - கழித்தல் (kazhithal) or எதிர்மறை (ethirmarai) or மைனஸ் (minus)
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: \"true\" ")
            + (pynutil.delete("கழித்தல்") | pynutil.delete("எதிர்மறை") | pynutil.delete("மைனஸ்"))
            + delete_space,
            0,
            1,
        )

        graph = optional_graph_negative + graph_decimal

        self.final_graph = self.add_tokens(graph)
        self.fst = self.final_graph.optimize()

    def add_tokens(self, fst):
        return pynutil.insert("decimal { ") + fst + pynutil.insert(" }")
