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

from nemo_text_processing.inverse_text_normalization.pa.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time (Punjabi)
        e.g. time { hours: "੧੨" minutes: "੩੦" } -> ੧੨:੩੦
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
        graph = (
            hour
            + delete_space
            + pynutil.insert(":")
            + minute
            + pynini.closure(delete_space + pynutil.insert(":") + second, 0, 1)
        )
        graph |= hour + delete_space + pynutil.insert(":") + second
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
