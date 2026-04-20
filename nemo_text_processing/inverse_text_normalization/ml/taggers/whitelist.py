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
    GraphFst,
)
from nemo_text_processing.inverse_text_normalization.ml.utils import get_abs_path


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelisted tokens (Malayalam)
        e.g. "ശ്രീ" -> tokens { name: "ശ്രീ." }

    Args:
        input_case: accepting either "lower_cased" or "cased" input
    """

    def __init__(self, input_case: str = "cased"):
        super().__init__(name="whitelist", kind="classify")

        whitelist = pynini.string_file(get_abs_path("data/whitelist/whitelist.tsv"))
        
        graph = pynutil.insert("name: \"") + whitelist + pynutil.insert("\"")

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
