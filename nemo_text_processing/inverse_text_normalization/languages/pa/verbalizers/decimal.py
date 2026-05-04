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

from nemo_text_processing.inverse_text_normalization.pa.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal (Punjabi)
        e.g. decimal { negative: "true" integer_part: "੧੨" fractional_part: "੫" quantity: "ਲੱਖ" } -> -੧੨.੫ ਲੱਖ
    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")

        optional_sign = pynini.closure(
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.cross("true", "-")
            + pynutil.delete("\"")
            + delete_space,
            0,
            1,
        )
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        fractional_part = (
            pynutil.delete("morphosyntactic_features:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        quantity = (
            pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        graph = optional_sign + pynini.closure(integer + delete_space, 0, 1) + fractional_part
        graph += pynini.closure(delete_space + insert_space + quantity, 0, 1)
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
