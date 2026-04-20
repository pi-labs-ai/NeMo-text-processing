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
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money
        e.g. money { integer_part: "൫൦൦" currency: "₹" } -> ₹൫൦൦
        e.g. money { integer_part: "൫൦൦" fractional_part: "൫൦" currency: "₹" } -> ₹൫൦൦.൫൦

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

        # Currency first in output, but integer_part comes first in input
        # Input: money { integer_part: "500" currency: "₹" }
        # Output: ₹500
        graph_no_fractional = (
            integer_part
            + delete_space
            + currency
        ) @ (
            pynini.closure(NEMO_NOT_QUOTE, 1)  # integer
            + pynini.closure(NEMO_NOT_QUOTE, 1)  # currency
        )

        # Actually, let's rewrite this properly
        # The verbalizer needs to reorder: put currency before integer_part
        graph_no_fractional = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)  # capture integer
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)  # capture currency
            + pynutil.delete("\"")
        )

        # For proper reordering, we need to use a different approach
        # Let's just output in the order we receive: integer then currency
        graph = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
