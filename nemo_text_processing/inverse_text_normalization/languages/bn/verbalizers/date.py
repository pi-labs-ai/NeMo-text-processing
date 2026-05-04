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

from nemo_text_processing.inverse_text_normalization.bn.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { month: "জানুয়ারি" day: "৫" year: "২০১২" preserve_order: true } -> জানুয়ারি ৫ ২০১২
        date { day: "৫" month: "জানুয়ারি" year: "২০১২" preserve_order: true } -> ৫ জানুয়ারি ২০১২
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst):
        super().__init__(name="date", kind="verbalize")
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        day = (
            pynutil.delete("day:")
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
            + delete_space
            + pynutil.delete("\"")
        )
        period = (
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
        morpho = (
            pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        preserve_order = pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true")

        graph_year = year
        graph_month = pynutil.insert(" ") + month
        graph_day = pynutil.insert(" ") + day
        graph_period = pynutil.insert(" ") + period

        # dmy
        graph_dmy = year + delete_space + graph_month + delete_space + graph_day
        # mdy
        graph_mdy = month + delete_space + graph_day + delete_space + pynutil.insert(" ") + year
        graph_mdy += delete_space + pynini.closure(preserve_order, 0, 1)
        # dm
        graph_dm = day + delete_space + graph_month
        # md
        graph_md = month + delete_space + graph_day
        graph_md += delete_space + pynini.closure(preserve_order, 0, 1)
        # my
        graph_my = month + delete_space + pynutil.insert(" ") + year
        # y
        graph_saal = year
        # y + period
        graph_y_period = year + delete_space + graph_period
        # d m y + period
        graph_dmy_period = graph_dmy + delete_space + graph_period
        # m y + period
        graph_my_period = graph_my + delete_space + graph_period
        # year range
        graph_y_range = year
        graph_y_range_period = graph_y_range + delete_space + graph_period
        # ordinal century
        graph_ordinal_century = era + morpho + delete_space + graph_period

        graph = (
            graph_dmy
            | graph_mdy
            | graph_dm
            | graph_md
            | graph_my
            | graph_saal
            | graph_y_period
            | graph_dmy_period
            | graph_my_period
            | graph_y_range
            | graph_y_range_period
            | graph_ordinal_century
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
