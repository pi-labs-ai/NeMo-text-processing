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
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.ml.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure (Malayalam)
        e.g. "അഞ്ച് കിലോമീറ്റർ" -> measure { cardinal { integer: "൫" } units: "കി.മീ." }
        e.g. "രണ്ട് കിലോഗ്രാം" -> measure { cardinal { integer: "൨" } units: "കി.ഗ്രാം" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst = None):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph

        # Measurement units
        unit_graph = pynini.string_file(get_abs_path("data/measure/measurements.tsv"))

        # Cardinal value
        graph_cardinal = (
            pynutil.insert("cardinal { integer: \"")
            + cardinal_graph
            + pynutil.insert("\" }")
        )

        # Units
        graph_units = pynutil.insert(" units: \"") + unit_graph + pynutil.insert("\"")

        graph = graph_cardinal + delete_space + graph_units

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
