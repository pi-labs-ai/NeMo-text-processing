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

from nemo_text_processing.inverse_text_normalization.pa.graph_utils import (
    NEMO_CHAR,
    NEMO_WHITE_SPACE,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.pa.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure (Punjabi)
        e.g. ਰਿਣ ਬਾਰਾਂ ਕਿਲੋਗ੍ਰਾਮ -> measure { decimal { negative: "true"  integer_part: "੧੨"  fractional_part: "੫੦"} units: "kg" }
        e.g. ਰਿਣ ਬਾਰਾਂ ਕਿਲੋਗ੍ਰਾਮ -> measure { cardinal { negative: "true"  integer_part: "੧੨"} units: "kg" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        decimal_graph = decimal.final_graph_wo_negative

        # Punjabi word for negative
        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("ਰਿਣ", "\"true\"") + delete_extra_space,
            0,
            1,
        )

        measurements_graph = pynini.string_file(get_abs_path("data/measure/measurements.tsv")).invert()
        paune_graph = pynini.string_file(get_abs_path("data/numbers/paune.tsv")).invert()

        self.measurements = pynutil.insert("units: \"") + measurements_graph + pynutil.insert("\" ")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        graph_integer_paune = pynutil.insert("integer_part: \"") + paune_graph + pynutil.insert("\"")

        # ਸਾਢੇ (saadhey) - one and a half
        graph_saade_single_digit = pynutil.add_weight(
            pynutil.delete("ਸਾਢੇ")
            + delete_space
            + graph_integer
            + delete_space
            + pynutil.insert(" fractional_part: \"੫\""),
            0.1,
        )
        # ਸਵਾ (sava) - one and a quarter
        graph_sava_single_digit = pynutil.add_weight(
            pynutil.delete("ਸਵਾ")
            + delete_space
            + graph_integer
            + delete_space
            + pynutil.insert(" fractional_part: \"੨੫\""),
            0.1,
        )
        # ਪੌਣੇ (paune) - three-quarters
        graph_paune_single_digit = pynutil.add_weight(
            pynutil.delete("ਪੌਣੇ")
            + delete_space
            + graph_integer_paune
            + delete_space
            + pynutil.insert(" fractional_part: \"੭੫\""),
            0.1,
        )
        # ਡੇਢ (dedh) - 1.5
        graph_dedh = pynutil.add_weight(
            pynini.union(pynutil.delete("ਡੇਢ") | pynutil.delete("ਡੇੜ"))
            + delete_space
            + pynutil.insert("integer_part: \"੧\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"੫\""),
            0.1,
        )
        # ਢਾਈ (dhaai) - 2.5
        graph_dhaai = pynutil.add_weight(
            pynutil.delete("ਢਾਈ")
            + delete_space
            + pynutil.insert("integer_part: \"੨\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"੫\""),
            0.1,
        )
        graph_quarterly_measures = (
            graph_saade_single_digit
            | graph_sava_single_digit
            | graph_paune_single_digit
            | graph_dedh
            | graph_dhaai
        )

        graph_decimal = pynutil.insert("decimal { ") + optional_graph_negative + decimal_graph + pynutil.insert(" }")
        graph_cardinal = pynutil.insert("cardinal { ") + optional_graph_negative + graph_integer + pynutil.insert(" }")
        graph_quarterly = (
            pynutil.insert("decimal { ") + optional_graph_negative + graph_quarterly_measures + pynutil.insert(" }")
        )

        graph = (graph_decimal | graph_cardinal | graph_quarterly) + delete_extra_space + self.measurements

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
