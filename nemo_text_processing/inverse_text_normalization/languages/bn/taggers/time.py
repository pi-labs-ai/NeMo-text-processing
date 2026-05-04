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

from nemo_text_processing.inverse_text_normalization.bn.graph_utils import (
    BENGALI_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
    integer_to_bengali,
)
from nemo_text_processing.inverse_text_normalization.bn.utils import get_abs_path


class TimeFst(GraphFst):
    """
        Finite state transducer for classifying time,
        e.g. একটা বেজে সাত মিনিট -> time { hours: "১" minutes: "৭" }
        e.g. চারটা বেজে চুয়াল্লিশ মিনিট -> time { hours: "৪" minutes: "৪৪" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        hour_graph = cardinal.graph_digit | cardinal.graph_teens_and_ties
        time_hours = pynini.union(*[integer_to_bengali(i) for i in range(1, 25)]).optimize()
        hour_graph = hour_graph @ time_hours

        cardinal_graph = cardinal.graph_single_digit_with_zero | cardinal.graph_teens_and_ties
        paune_hour_graph = pynini.string_file(get_abs_path("data/time/hour_for_paune.tsv")).invert()

        # Bengali time markers: টা (ta), বাজে (baje), বেজে (beje)
        delete_baje = pynini.union(
            pynutil.delete("বেজে") | pynutil.delete("বাজে") | pynutil.delete("টায়") | pynutil.delete("টা") | pynutil.delete("ঘণ্টা") | pynutil.delete("টার")
        )

        # Bengali: মিনিট (minit) for minute
        delete_minute = pynutil.delete("মিনিট")
        # Bengali: সেকেন্ড (second) for second
        delete_second = pynutil.delete("সেকেন্ড")
        
        # Bengali minute suffix "ে" (locative case) - e.g., পঁয়ত্রিশে = at 35
        delete_minute_locative = pynutil.delete("ে")

        self.hour = pynutil.insert("hours: \"") + hour_graph + pynutil.insert("\" ")
        self.paune_hour = pynutil.insert("hours: \"") + paune_hour_graph + pynutil.insert("\" ")
        self.minute = pynutil.insert("minutes: \"") + cardinal_graph + pynutil.insert("\" ")
        # Minute with locative suffix "ে"
        self.minute_with_locative = pynutil.insert("minutes: \"") + cardinal_graph + delete_minute_locative + pynutil.insert("\" ")
        self.second = pynutil.insert("seconds: \"") + cardinal_graph + pynutil.insert("\" ")

        # hour minute second
        graph_hms = (
            self.hour
            + delete_space
            + delete_baje
            + delete_space
            + self.minute
            + delete_space
            + delete_minute
            + delete_space
            + self.second
            + delete_space
            + delete_second
        )

        # hour minute and hour minute without "baje and minat"
        graph_hm = pynutil.add_weight(
            self.hour
            + delete_space
            + pynini.closure(delete_baje, 0, 1)
            + delete_space
            + self.minute
            + pynini.closure(delete_space + delete_minute, 0, 1),
            0.01,
        )
        
        # hour minute with locative suffix "ে" - e.g., চারটা পঁয়ত্রিশে = 4:35
        graph_hm_locative = pynutil.add_weight(
            self.hour
            + delete_space
            + pynini.closure(delete_baje, 0, 1)
            + delete_space
            + self.minute_with_locative,
            0.01,
        )

        # hour second
        graph_hs = pynutil.add_weight(
            self.hour + delete_space + delete_baje + delete_space + self.second + delete_space + delete_second, 0.01
        )

        # minute second
        graph_ms = (
            self.minute + delete_space + delete_minute + delete_space + self.second + delete_space + delete_second
        )

        # hour only
        graph_hour = self.hour + delete_space + delete_baje

        # Bengali: সাড়ে (sare) for half-past, e.g., সাড়ে তিনটা = 3:30
        graph_saade = pynutil.add_weight(
            pynutil.delete("সাড়ে")
            + delete_space
            + self.hour
            + delete_space
            + pynutil.insert(" minutes: \"৩০\"")
            + delete_space
            + pynini.closure(delete_baje),
            0.01,
        )
        # Bengali: সোয়া (soya) for quarter-past, e.g., সোয়া তিনটা = 3:15
        graph_sava = pynutil.add_weight(
            pynutil.delete("সোয়া")
            + delete_space
            + self.hour
            + delete_space
            + pynutil.insert(" minutes: \"১৫\"")
            + delete_space
            + pynini.closure(delete_baje),
            0.01,
        )
        # Bengali: পৌনে (poune) for quarter-to, e.g., পৌনে চারটা = 3:45
        # paune_hour_graph now handles suffixes like টা/টার directly
        graph_paune = pynutil.add_weight(
            pynutil.delete("পৌনে")
            + delete_space
            + self.paune_hour
            + pynutil.insert(" minutes: \"৪৫\""),
            0.01,
        )
        # Bengali: দেড়টা (derta) for 1:30
        graph_dedh = pynutil.add_weight(
            pynutil.delete("দেড়টা")
            + delete_space
            + pynini.closure(delete_baje)
            + pynutil.insert("hours: \"১\"")
            + delete_space
            + pynutil.insert(" minutes: \"৩০\""),
            0.01,
        )
        # Bengali: আড়াইটা (araita) for 2:30
        graph_dhaai = pynutil.add_weight(
            pynutil.delete("আড়াইটা")
            + delete_space
            + pynini.closure(delete_baje)
            + pynutil.insert("hours: \"২\"")
            + delete_space
            + pynutil.insert(" minutes: \"৩০\""),
            0.01,
        )
        graph_quarterly_measures = (
            graph_dedh
            | graph_dhaai
            | graph_hour
            | graph_saade
            | graph_sava
            | graph_paune
        )

        final_graph = graph_hms | graph_hm | graph_hm_locative | graph_hs | graph_ms | graph_quarterly_measures

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
