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
from nemo_text_processing.inverse_text_normalization.ta.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure (Tamil),
        e.g. ஐந்து கிலோமீட்டர் -> measure { cardinal { integer: "௫" } units: "கி.மீ." }
    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph
        
        unit_graph = pynini.string_file(get_abs_path("data/measure/measurements.tsv")).invert()

        graph_unit = pynutil.insert(" units: \"") + unit_graph + pynutil.insert("\"")
        graph_cardinal = pynutil.insert("cardinal { integer: \"") + cardinal_graph + pynutil.insert("\" }")

        graph = graph_cardinal + delete_space + graph_unit

        self.final_graph = self.add_tokens(graph)
        self.fst = self.final_graph.optimize()

    def add_tokens(self, fst):
        return pynutil.insert("measure { ") + fst + pynutil.insert(" }")
