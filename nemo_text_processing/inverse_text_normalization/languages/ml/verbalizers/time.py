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
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
)


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time
        e.g. time { hours: "൧൨" minutes: "൩൦" } -> ൧൨:൩൦
        e.g. time { hours: "൩" } -> ൩:൦൦

    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        second = (
            pynutil.delete("seconds:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        suffix = (
            pynutil.delete("suffix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        optional_suffix = pynini.closure(delete_space + pynutil.insert(" ") + suffix, 0, 1)

        # Hour only -> H:00
        graph_h_only = hour + pynutil.insert(":൦൦")

        # Hour and minute -> H:MM
        graph_hm = hour + delete_space + pynutil.insert(":") + minute

        # Hour, minute and second -> H:MM:SS
        graph_hms = (
            hour
            + delete_space
            + pynutil.insert(":")
            + minute
            + delete_space
            + pynutil.insert(":")
            + second
        )

        graph = (graph_hms | graph_hm | graph_h_only) + optional_suffix

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
