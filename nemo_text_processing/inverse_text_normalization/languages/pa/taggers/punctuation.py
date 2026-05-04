# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.pa.graph_utils import (
    NEMO_NOT_SPACE,
    NEMO_PUNCT,
    GraphFst,
)


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation (Punjabi)
        e.g. . -> tokens { name: "." }
    """

    def __init__(self):
        super().__init__(name="punct", kind="classify")
        
        # Standard punctuation
        graph = pynutil.insert("name: \"") + NEMO_PUNCT + pynutil.insert("\"")
        
        # Punjabi specific punctuation marks - use escape for special characters
        # ॥ - Danda (sentence end marker)
        # । - Single Danda  
        punjabi_punct = pynini.union(
            "।", "॥", "?", "!", ",", ";", ":", "-", "–", "—", "'", 
            pynini.escape("\""),
            pynini.escape("("),
            pynini.escape(")"),
            pynini.escape("["),
            pynini.escape("]"),
            pynini.escape("{"),
            pynini.escape("}")
        )
        graph_punjabi = pynutil.insert("name: \"") + punjabi_punct + pynutil.insert("\"")
        
        final_graph = graph | graph_punjabi
        self.fst = final_graph.optimize()
