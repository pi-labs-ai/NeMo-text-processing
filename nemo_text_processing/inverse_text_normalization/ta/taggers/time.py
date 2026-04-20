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
    TAMIL_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
    integer_to_tamil,
)
from nemo_text_processing.inverse_text_normalization.ta.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time (Tamil),
        e.g. ஒரு மணி ஏழு நிமிடம் -> time { hours: "௧" minutes: "௭" }
        e.g. நான்கு மணி நாற்பத்தினான்கு நிமிடம் -> time { hours: "௪" minutes: "௪௪" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        hour_graph = cardinal.graph_digit | cardinal.graph_teens_and_ties
        time_hours = pynini.union(*[integer_to_tamil(i) for i in range(1, 25)]).optimize()
        hour_graph = hour_graph @ time_hours

        cardinal_graph = cardinal.graph_single_digit_with_zero | cardinal.graph_teens_and_ties
        paune_hour_graph = pynini.string_file(get_abs_path("data/time/hour_for_paune.tsv")).invert()

        # Tamil words for time expressions
        delete_mani = pynini.union(
            pynutil.delete("மணி") | pynutil.delete("மணிக்கு")
        )

        delete_minute = pynutil.delete("நிமிடம்") | pynutil.delete("நிமிடங்கள்")
        delete_second = pynutil.delete("வினாடி") | pynutil.delete("வினாடிகள்") | pynutil.delete("செக்கண்ட்")

        self.hour = pynutil.insert("hours: \"") + hour_graph + pynutil.insert("\" ")
        self.paune_hour = pynutil.insert("hours: \"") + paune_hour_graph + pynutil.insert("\" ")
        self.minute = pynutil.insert("minutes: \"") + cardinal_graph + pynutil.insert("\" ")
        self.second = pynutil.insert("seconds: \"") + cardinal_graph + pynutil.insert("\" ")

        # hour minute second
        graph_hms = (
            self.hour
            + delete_space
            + delete_mani
            + delete_space
            + self.minute
            + delete_space
            + delete_minute
            + delete_space
            + self.second
            + delete_space
            + delete_second
        )

        # hour minute and hour minute without "mani and nimidam"
        graph_hm = pynutil.add_weight(
            self.hour
            + delete_space
            + pynini.closure(delete_mani, 0, 1)
            + delete_space
            + self.minute
            + pynini.closure(delete_space + delete_minute, 0, 1),
            0.01,
        )

        # hour second
        graph_hs = pynutil.add_weight(
            self.hour + delete_space + delete_mani + delete_space + self.second + delete_space + delete_second, 0.01
        )

        # minute second
        graph_ms = (
            self.minute + delete_space + delete_minute + delete_space + self.second + delete_space + delete_second
        )

        # hour only
        graph_hour = self.hour + delete_space + delete_mani

        # அரை (arai) - half past (30 minutes)
        graph_arai = pynutil.add_weight(
            pynutil.delete("அரை")
            + delete_space
            + self.hour
            + delete_space
            + pynutil.insert(" minutes: \"௩௦\"")
            + delete_space
            + pynini.closure(delete_mani),
            0.01,
        )
        # கால் (kaal) - quarter past (15 minutes)
        graph_kaal = pynutil.add_weight(
            pynutil.delete("கால்")
            + delete_space
            + self.hour
            + delete_space
            + pynutil.insert(" minutes: \"௧௫\"")
            + delete_space
            + pynini.closure(delete_mani),
            0.01,
        )
        # முக்கால் (mukkaal) - three-quarters past (45 minutes)
        graph_mukkaal = pynutil.add_weight(
            pynutil.delete("முக்கால்")
            + delete_space
            + self.paune_hour
            + delete_space
            + pynutil.insert(" minutes: \"௪௫\"")
            + delete_space
            + pynini.closure(delete_mani),
            0.01,
        )
        # ஒன்றரை (onrarai) - 1:30
        graph_onrarai = pynutil.add_weight(
            pynutil.delete("ஒன்றரை")
            + delete_space
            + delete_mani
            + pynutil.insert("hours: \"௧\"")
            + delete_space
            + pynutil.insert(" minutes: \"௩௦\""),
            0.01,
        )
        # இரண்டரை (irandrai) - 2:30
        graph_irandrai = pynutil.add_weight(
            pynutil.delete("இரண்டரை")
            + delete_space
            + delete_mani
            + pynutil.insert("hours: \"௨\"")
            + delete_space
            + pynutil.insert(" minutes: \"௩௦\""),
            0.01,
        )
        graph_quarterly_measures = (
            graph_onrarai
            | graph_irandrai
            | graph_arai
            | graph_kaal
            | graph_mukkaal
        )

        graph = graph_hms | graph_hm | graph_hs | graph_ms | graph_hour | graph_quarterly_measures

        self.final_graph = self.add_tokens(graph)
        self.fst = self.final_graph.optimize()

    def add_tokens(self, fst):
        return pynutil.insert("time { ") + fst + pynutil.insert(" }")
