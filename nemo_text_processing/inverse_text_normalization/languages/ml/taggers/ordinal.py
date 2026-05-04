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


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinals (Malayalam)
        e.g. "ഒന്നാം" -> ordinal { integer: "൧" }
        e.g. "രണ്ടാം" -> ordinal { integer: "൨" }
        e.g. "മൂന്നാം" -> ordinal { integer: "൩" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        # Load ordinal mappings from file
        # The TSV has digit -> word format, so we invert for ITN (word -> digit)
        ordinal_graph = pynini.string_file(get_abs_path("data/ordinals/ordinal.tsv")).invert()

        # Malayalam ordinals are typically formed by adding -ാം suffix
        # e.g., ഒന്ന് -> ഒന്നാം, രണ്ട് -> രണ്ടാം
        
        graph = pynutil.insert("integer: \"") + ordinal_graph + pynutil.insert("\"")

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
