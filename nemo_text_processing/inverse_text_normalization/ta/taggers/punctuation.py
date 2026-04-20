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
)


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation (Tamil),
        e.g. . -> punctuation { name: "." }
    """

    def __init__(self):
        super().__init__(name="punctuation", kind="classify")

        # Define punctuation marks
        punct = pynini.union(
            ".",
            ",",
            "!",
            "?",
            ":",
            ";",
            "-",
            "'",
            '"',
            pynini.escape("("),
            pynini.escape(")"),
            pynini.escape("["),
            pynini.escape("]"),
            pynini.escape("{"),
            pynini.escape("}"),
            "/",
            "\\",
            "@",
            "#",
            "$",
            "%",
            "^",
            "&",
            "*",
            "_",
            "+",
            "=",
            "<",
            ">",
            "|",
            "~",
            "`",
        )

        graph = pynutil.insert("name: \"") + punct + pynutil.insert("\"")

        self.final_graph = self.add_tokens(graph)
        self.fst = self.final_graph.optimize()

    def add_tokens(self, fst):
        return pynutil.insert("punctuation { ") + fst + pynutil.insert(" }")
