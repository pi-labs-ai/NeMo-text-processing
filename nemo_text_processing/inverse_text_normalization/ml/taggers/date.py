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
from nemo_text_processing.inverse_text_normalization.ml.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date (Malayalam)
        e.g. "പതിനൊന്ന് ഡിസംബർ രണ്ടായിരത്തി ഇരുപത്തിമൂന്ന്" -> date { day: "൧൧" month: "ഡിസംബർ" year: "൨൦൨൩" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst = None):
        super().__init__(name="date", kind="classify")

        cardinal_graph = cardinal.graph
        month_graph = pynini.string_file(get_abs_path("data/date/months.tsv")).invert()

        # Day (1-31)
        day_graph = pynutil.insert("day: \"") + cardinal_graph + pynutil.insert("\"")

        # Month
        month_graph_final = pynutil.insert(" month: \"") + month_graph + pynutil.insert("\"")

        # Year
        year_graph = pynutil.insert(" year: \"") + cardinal_graph + pynutil.insert("\"")

        # Format: day month year - e.g., "പതിനൊന്ന് ഡിസംബർ രണ്ടായിരത്തി ഇരുപത്തിമൂന്ന്"
        graph_dmy = day_graph + delete_space + month_graph_final + delete_space + year_graph

        # Format: day month - e.g., "പതിനൊന്ന് ഡിസംബർ"
        graph_dm = day_graph + delete_space + month_graph_final

        # Format: month year - e.g., "ഡിസംബർ രണ്ടായിരത്തി ഇരുപത്തിമൂന്ന്"
        graph_my = (
            pynutil.insert("month: \"")
            + month_graph
            + pynutil.insert("\"")
            + delete_space
            + pynutil.insert(" year: \"")
            + cardinal_graph
            + pynutil.insert("\"")
        )

        graph = graph_dmy | graph_dm | graph_my

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
