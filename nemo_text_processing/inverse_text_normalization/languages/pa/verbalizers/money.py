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


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money (Punjabi)
        e.g. money { integer_part: "੭੨" currency: "₹" } -> ₹੭੨
    """

    def __init__(self):
        super().__init__(name="money", kind="verbalize")

        integer_part = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        fractional_part = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        currency = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        graph_integer = integer_part + delete_space + currency
        graph_decimal = integer_part + pynutil.insert(".") + delete_space + fractional_part + delete_space + currency
        graph_fractional = fractional_part + delete_space + currency

        # Reorder to put currency symbol first
        graph = pynini.union(
            graph_integer,
            graph_decimal,
            graph_fractional,
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
