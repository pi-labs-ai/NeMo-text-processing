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


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone (Tamil),
        e.g. ஒன்று இரண்டு மூன்று நான்கு ஐந்து ஆறு ஏழு எட்டு ஒன்பது சுழியம் -> telephone { number_part: "௧௨௩௪௫௬௭௮௯௦" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")

        digit_graph = cardinal.graph_digit | cardinal.graph_zero
        
        # Phone number as sequence of digits
        graph_phone = pynini.closure(digit_graph + delete_space, 5) + digit_graph

        graph = pynutil.insert("number_part: \"") + graph_phone + pynutil.insert("\"")

        self.final_graph = self.add_tokens(graph)
        self.fst = self.final_graph.optimize()

    def add_tokens(self, fst):
        return pynutil.insert("telephone { ") + fst + pynutil.insert(" }")
