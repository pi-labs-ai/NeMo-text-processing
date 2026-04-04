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

from nemo_text_processing.inverse_text_normalization.bn.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MIN_NEG_WEIGHT,
    NEMO_BN_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.bn.utils import get_abs_path


def get_quantity(
    decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike', input_case: str = INPUT_LOWER_CASED
) -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. দশ লক্ষ -> integer_part: "১০" quantity: "লক্ষ"
    e.g. এক দশমিক পাঁচ লক্ষ -> integer_part: "১" fractional_part: "৫" quantity: "লক্ষ"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
        input_case: accepting either "lower_cased" or "cased" input.
    """
    numbers = cardinal_up_to_hundred @ (
        pynutil.delete(pynini.closure("০")) + pynini.difference(NEMO_BN_DIGIT, "০") + pynini.closure(NEMO_BN_DIGIT)
    )

    suffix = pynini.string_file(get_abs_path("data/numbers/thousands.tsv"))
    # Add weight penalty to prefer full cardinal conversion over quantity format
    res = pynutil.add_weight(
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + delete_extra_space
        + pynutil.insert("quantity: \"")
        + suffix
        + pynutil.insert("\""),
        1.0  # Penalty to prefer cardinal graph
    )
    res |= pynutil.add_weight(
        decimal + delete_extra_space + pynutil.insert("quantity: \"") + suffix + pynutil.insert("\""),
        1.0
    )
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        Decimal point "." is determined by "দশমিক"
            e.g. ঋণ এক দশমিক দুই ছয় -> decimal { negative: "true" integer_part: "১" morphosyntactic_features: "." fractional_part: "২৬" }

        This decimal rule assumes that decimals can be pronounced as:
        (a cardinal) + ('দশমিক') plus (any sequence of cardinals <১০০০, including 'শূন্য')

        Also writes large numbers in shortened form, e.g.
            e.g. এক দশমিক দুই ছয় লক্ষ -> decimal { negative: "false" integer_part: "১" morphosyntactic_features: "." fractional_part: "২৬" quantity: "লক্ষ" }
            e.g. দুই লক্ষ -> decimal { negative: "false" integer_part: "২" quantity: "লক্ষ" }
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_decimal = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_decimal |= pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()

        graph_decimal = pynini.closure(graph_decimal + delete_space) + graph_decimal
        self.graph = graph_decimal

        # Bengali: দশমিক (doshomik) for decimal point
        point = pynutil.delete("দশমিক")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("ঋণ", "\"true\"") + delete_extra_space,
            0,
            1,
        )

        graph_fractional = pynutil.insert("fractional_part: \"") + graph_decimal + pynutil.insert("\"")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        final_graph_wo_sign = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1) + point + delete_extra_space + graph_fractional
        )
        final_graph = optional_graph_negative + final_graph_wo_sign

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal_graph, input_case=input_case
        )

        quantity_graph = get_quantity(final_graph_wo_sign, cardinal_graph, input_case=input_case)
        final_graph |= optional_graph_negative + quantity_graph

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
