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


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers (Malayalam)
        e.g. "ഒന്ന് രണ്ട് മൂന്ന്" -> telephone { number_part: "൧൨൩" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")

        digit_graph = cardinal.graph_digit
        graph_zero = cardinal.graph_zero
        
        # Each digit spoken separately
        single_digit = digit_graph | graph_zero

        # Sequence of digits (at least 3 for a phone number)
        graph_number_part = pynini.closure(single_digit + delete_space, 2) + single_digit

        # Country code (optional)
        country_code_graph = pynini.string_file(get_abs_path("data/telephone/country_codes.tsv"))
        optional_country_code = pynini.closure(
            pynutil.insert("country_code: \"")
            + country_code_graph
            + pynutil.insert("\" ")
            + delete_space,
            0,
            1,
        )

        graph_number = pynutil.insert("number_part: \"") + graph_number_part + pynutil.insert("\"")

        graph = optional_country_code + graph_number

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
