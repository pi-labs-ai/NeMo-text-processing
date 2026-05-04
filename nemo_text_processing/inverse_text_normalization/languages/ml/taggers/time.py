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
    NEMO_ML_DIGIT,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.ml.utils import get_abs_path, load_labels


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time (Malayalam)
        e.g. "മൂന്ന് മണി" -> time { hours: "൩" }
        e.g. "മൂന്ന് മണി പത്ത് മിനിറ്റ്" -> time { hours: "൩" minutes: "൧൦" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")

        cardinal_graph = cardinal.graph

        # Words for hour and minute
        delete_hour = pynutil.delete("മണി") | pynutil.delete("മണിക്ക്") | pynutil.delete("മണിയ്ക്ക്")
        delete_minute = pynutil.delete("മിനിറ്റ്") | pynutil.delete("മിനുട്ട്") | pynutil.delete("മിനിറ്റു")
        delete_second = pynutil.delete("സെക്കന്റ്") | pynutil.delete("സെക്കൻഡ്")
        
        # Time suffixes (AM/PM equivalent)
        time_suffix = pynini.string_file(get_abs_path("data/time/time_suffix.tsv"))
        
        graph_hours = (
            pynutil.insert("hours: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + delete_space
            + delete_hour
        )
        
        graph_minutes = (
            pynutil.insert(" minutes: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + delete_space
            + delete_minute
        )
        
        graph_seconds = (
            pynutil.insert(" seconds: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + delete_space
            + delete_second
        )
        
        optional_minutes = pynini.closure(delete_space + graph_minutes, 0, 1)
        optional_seconds = pynini.closure(delete_space + graph_seconds, 0, 1)
        optional_suffix = pynini.closure(
            delete_space
            + pynutil.insert(" suffix: \"")
            + time_suffix
            + pynutil.insert("\""),
            0,
            1,
        )

        graph = graph_hours + optional_minutes + optional_seconds + optional_suffix

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
