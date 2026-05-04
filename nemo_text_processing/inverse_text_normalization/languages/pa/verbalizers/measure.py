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

from nemo_text_processing.inverse_text_normalization.pa.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space
from nemo_text_processing.inverse_text_normalization.pa.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.pa.verbalizers.cardinal import CardinalFst


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure (Punjabi)
        e.g. measure { cardinal { integer: "੧੨" } units: "kg" } -> ੧੨ kg
    """

    def __init__(self):
        super().__init__(name="measure", kind="verbalize")

        cardinal = CardinalFst()
        decimal = DecimalFst()

        units = (
            pynutil.delete("units:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + cardinal.numbers
            + delete_space
            + pynutil.delete("}")
        )

        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + decimal.fst
            + delete_space
            + pynutil.delete("}")
        )

        graph = (graph_cardinal | graph_decimal) + delete_space + insert_space + units

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
