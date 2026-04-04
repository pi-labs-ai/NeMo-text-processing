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
    NEMO_CHAR,
    NEMO_WHITE_SPACE,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.bn.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. ঋণ বারো কিলোগ্রাম -> measure { decimal { negative: "true"  integer_part: "১২"  fractional_part: "৫০"} units: "kg" }
        e.g. ঋণ বারো কিলোগ্রাম -> measure { cardinal { negative: "true"  integer_part: "১২"} units: "kg" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        decimal_graph = decimal.final_graph_wo_negative

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("ঋণ", "\"true\"") + delete_extra_space,
            0,
            1,
        )

        measurements_graph = pynini.string_file(get_abs_path("data/measure/measurements.tsv")).invert()
        paune_graph = pynini.string_file(get_abs_path("data/numbers/paune.tsv")).invert()

        self.measurements = pynutil.insert("units: \"") + measurements_graph + pynutil.insert("\" ")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        graph_integer_paune = pynutil.insert("integer_part: \"") + paune_graph + pynutil.insert("\"")

        # Bengali: সাড়ে (sare) for half-past
        graph_saade_single_digit = pynutil.add_weight(
            pynutil.delete("সাড়ে")
            + delete_space
            + graph_integer
            + delete_space
            + pynutil.insert(" fractional_part: \"৫\""),
            0.1,
        )
        # Bengali: সোয়া (soya) for quarter-past
        graph_sava_single_digit = pynutil.add_weight(
            pynutil.delete("সোয়া")
            + delete_space
            + graph_integer
            + delete_space
            + pynutil.insert(" fractional_part: \"২৫\""),
            0.1,
        )
        # Bengali: পৌনে (poune) for three-quarters before
        graph_paune_single_digit = pynutil.add_weight(
            pynutil.delete("পৌনে")
            + delete_space
            + graph_integer_paune
            + delete_space
            + pynutil.insert(" fractional_part: \"৭৫\""),
            1,
        )
        # Bengali: দেড় (der) for one and a half
        graph_dedh_single_digit = pynutil.add_weight(
            pynutil.delete("দেড়")
            + delete_space
            + pynutil.insert("integer_part: \"১\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"৫\""),
            0.1,
        )
        # Bengali: আড়াই (arai) for two and a half
        graph_dhaai_single_digit = pynutil.add_weight(
            pynutil.delete("আড়াই")
            + delete_space
            + pynutil.insert("integer_part: \"২\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"৫\""),
            1,
        )

        graph_quarterly_measures = pynini.union(
            graph_saade_single_digit,
            graph_sava_single_digit,
            graph_paune_single_digit,
            graph_dedh_single_digit,
            graph_dhaai_single_digit,
        )

        graph_decimal = (
            optional_graph_negative
            + pynutil.insert("decimal { ")
            + (decimal_graph | graph_quarterly_measures)
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )

        graph_cardinal = (
            optional_graph_negative
            + pynutil.insert("cardinal { ")
            + pynutil.insert("integer: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + self.measurements
        )

        final_graph = graph_decimal | graph_cardinal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
