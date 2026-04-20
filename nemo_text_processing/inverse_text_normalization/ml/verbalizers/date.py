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
    GraphFst,
    delete_space,
)


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date
        e.g. date { day: "൧൧" month: "ഡിസംബർ" year: "൨൦൨൩" } -> ൧൧/ഡിസംബർ/൨൦൨൩

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

        # Day/Month/Year format
        graph_dmy = day + delete_space + pynutil.insert("/") + month + delete_space + pynutil.insert("/") + year

        # Day/Month format
        graph_dm = day + delete_space + pynutil.insert("/") + month

        # Month/Year format
        graph_my = month + delete_space + pynutil.insert("/") + year

        graph = graph_dmy | graph_dm | graph_my

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
