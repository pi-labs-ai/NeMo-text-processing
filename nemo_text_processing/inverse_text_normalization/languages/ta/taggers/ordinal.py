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


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal (Tamil),
        e.g. முதல் -> ordinal { integer: "௧" }
        e.g. ஐந்தாவது -> ordinal { integer: "௫" }
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="classify")

        ordinal_graph = pynini.string_file(get_abs_path("data/ordinals/ordinal.tsv")).invert()
        
        graph = pynutil.insert("integer: \"") + ordinal_graph + pynutil.insert("\"")

        self.final_graph = self.add_tokens(graph)
        self.fst = self.final_graph.optimize()

    def add_tokens(self, fst):
        return pynutil.insert("ordinal { ") + fst + pynutil.insert(" }")
