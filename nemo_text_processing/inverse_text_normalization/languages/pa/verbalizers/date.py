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
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date (Punjabi)
        e.g. date { month: "ਜਨਵਰੀ" day: "੫" year: "੨੦੧੨" } -> ੫ ਜਨਵਰੀ ੨੦੧੨
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        text = (
            pynutil.delete("text:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        era = (
            pynutil.delete("era:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        morpho = pynini.closure(
            delete_space
            + pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\""),
            0,
            1,
        )

        graph_dmy = day + delete_space + insert_space + month + pynini.closure(delete_space + insert_space + year, 0, 1)
        graph_mdy = month + delete_space + insert_space + day + pynini.closure(delete_space + insert_space + year, 0, 1)
        graph_my = month + delete_space + insert_space + year
        graph_y = year
        graph_y_text = year + delete_space + insert_space + text
        graph_era = era + morpho + delete_space + insert_space + text

        graph = graph_dmy | graph_mdy | graph_my | graph_y | graph_y_text | graph_era
        graph += delete_preserve_order

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
