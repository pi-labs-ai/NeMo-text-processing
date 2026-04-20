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

from nemo_text_processing.inverse_text_normalization.pa.graph_utils import (
    GURMUKHI_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
    integer_to_gurmukhi,
)
from nemo_text_processing.inverse_text_normalization.pa.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time (Punjabi),
        e.g. ਇੱਕ ਵੱਜ ਕੇ ਸੱਤ ਮਿੰਟ -> time { hours: "੧" minutes: "੭" }
        e.g. ਚਾਰ ਵਜੇ ਚੁਆਲੀ ਮਿੰਟ -> time { hours: "੪" minutes: "੪੪" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        hour_graph = cardinal.graph_digit | cardinal.graph_teens_and_ties
        time_hours = pynini.union(*[integer_to_gurmukhi(i) for i in range(1, 25)]).optimize()
        hour_graph = hour_graph @ time_hours

        cardinal_graph = cardinal.graph_single_digit_with_zero | cardinal.graph_teens_and_ties
        paune_hour_graph = pynini.string_file(get_abs_path("data/time/hour_for_paune.tsv")).invert()

        # Punjabi words for time expressions
        delete_vaje = pynini.union(
            pynutil.delete("ਵੱਜ ਕੇ") | pynutil.delete("ਵੱਜਕੇ") | pynutil.delete("ਵਜੇ") | pynutil.delete("ਘੰਟਾ") | pynutil.delete("ਘੰਟੇ")
        )

        delete_minute = pynutil.delete("ਮਿੰਟ")
        delete_second = pynutil.delete("ਸਕਿੰਟ") | pynutil.delete("ਸੈਕੰਡ")

        self.hour = pynutil.insert("hours: \"") + hour_graph + pynutil.insert("\" ")
        self.paune_hour = pynutil.insert("hours: \"") + paune_hour_graph + pynutil.insert("\" ")
        self.minute = pynutil.insert("minutes: \"") + cardinal_graph + pynutil.insert("\" ")
        self.second = pynutil.insert("seconds: \"") + cardinal_graph + pynutil.insert("\" ")

        # hour minute second
        graph_hms = (
            self.hour
            + delete_space
            + delete_vaje
            + delete_space
            + self.minute
            + delete_space
            + delete_minute
            + delete_space
            + self.second
            + delete_space
            + delete_second
        )

        # hour minute and hour minute without "vaje and minat"
        graph_hm = pynutil.add_weight(
            self.hour
            + delete_space
            + pynini.closure(delete_vaje, 0, 1)
            + delete_space
            + self.minute
            + pynini.closure(delete_space + delete_minute, 0, 1),
            0.01,
        )

        # hour second
        graph_hs = pynutil.add_weight(
            self.hour + delete_space + delete_vaje + delete_space + self.second + delete_space + delete_second, 0.01
        )

        # minute second
        graph_ms = (
            self.minute + delete_space + delete_minute + delete_space + self.second + delete_space + delete_second
        )

        # hour only
        graph_hour = self.hour + delete_space + delete_vaje

        # ਸਾਢੇ (saadhey) - half past
        graph_saade = pynutil.add_weight(
            pynutil.delete("ਸਾਢੇ")
            + delete_space
            + self.hour
            + delete_space
            + pynutil.insert(" minutes: \"੩੦\"")
            + delete_space
            + pynini.closure(delete_vaje),
            0.01,
        )
        # ਸਵਾ (sava) - quarter past
        graph_sava = pynutil.add_weight(
            pynutil.delete("ਸਵਾ")
            + delete_space
            + self.hour
            + delete_space
            + pynutil.insert(" minutes: \"੧੫\"")
            + delete_space
            + pynini.closure(delete_vaje),
            0.01,
        )
        # ਪੌਣੇ (paune) - quarter to
        graph_paune = pynutil.add_weight(
            pynutil.delete("ਪੌਣੇ")
            + delete_space
            + self.paune_hour
            + delete_space
            + pynutil.insert(" minutes: \"੪੫\"")
            + delete_space
            + pynini.closure(delete_vaje),
            0.01,
        )
        # ਡੇਢ (dedh) - 1:30
        graph_dedh = pynutil.add_weight(
            pynini.union(pynutil.delete("ਡੇਢ") | pynutil.delete("ਡੇੜ"))
            + delete_space
            + delete_vaje
            + pynutil.insert("hours: \"੧\"")
            + delete_space
            + pynutil.insert(" minutes: \"੩੦\""),
            0.01,
        )
        # ਢਾਈ (dhaai) - 2:30
        graph_dhaai = pynutil.add_weight(
            pynutil.delete("ਢਾਈ")
            + delete_space
            + delete_vaje
            + pynutil.insert("hours: \"੨\"")
            + delete_space
            + pynutil.insert(" minutes: \"੩੦\""),
            0.01,
        )
        graph_quarterly_measures = (
            graph_dedh
            | graph_dhaai
            | graph_saade
            | graph_sava
            | graph_paune
        )

        graph = graph_hms | graph_hm | graph_hs | graph_ms | graph_hour | graph_quarterly_measures

        self.final_graph = self.add_tokens(graph)
        self.fst = self.final_graph.optimize()
