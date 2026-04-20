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
    INPUT_CASED,
    INPUT_LOWER_CASED,
    NEMO_PA_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.pa.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction (Punjabi)
        Fraction "/" is determined by "ਭਾਗ" (divided by)
            e.g. ਰਿਣ ਇੱਕ ਭਾਗ ਛੱਬੀ -> fraction { negative: "true" numerator: "੧" denominator: "੨੬" }
            e.g. ਛੇ ਸੌ ਸੱਠ ਭਾਗ ਪੰਜ ਸੌ ਤੇਤੀ -> fraction { negative: "false" numerator: "੬੬੦" denominator: "੫੪੩" }

        The fractional rule assumes that fractions can be pronounced as:
        (a cardinal) + ('ਭਾਗ') plus (a cardinal, excluding 'ਸਿਫਰ')
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator
        graph_cardinal = cardinal.graph_no_exception

        integer = pynutil.insert("integer_part: \"") + graph_cardinal + pynutil.insert("\" ")
        integer += delete_space
        
        # Punjabi word for "divided by" - ਭਾਗ
        delete_bhaag = pynini.union(pynutil.delete(" ਭਾਗ ") | pynutil.delete(" ਵਿੱਚ "))

        numerator = pynutil.insert("numerator: \"") + graph_cardinal + pynutil.insert("\"")
        denominator = pynutil.insert(" denominator: \"") + graph_cardinal + pynutil.insert("\"")

        graph_fraction = numerator + delete_bhaag + denominator
        
        # "ਸਹੀ" for mixed fractions (like "proper")
        graph_mixed_fraction = integer + delete_extra_space + pynutil.delete("ਸਹੀ") + delete_space + graph_fraction

        # ਸਾਢੇ - one and a half
        graph_saade = pynutil.add_weight(
            pynutil.delete("ਸਾਢੇ")
            + delete_space
            + integer
            + pynutil.insert("numerator: \"੧\"")
            + delete_space
            + pynutil.insert(" denominator: \"੨\""),
            -0.01,
        )
        # ਸਵਾ - one and a quarter
        graph_sava = pynutil.add_weight(
            pynutil.delete("ਸਵਾ")
            + delete_space
            + integer
            + pynutil.insert("numerator: \"੧\"")
            + delete_space
            + pynutil.insert(" denominator: \"੪\""),
            -0.01,
        )
        # ਪੌਣੇ - three-quarters
        graph_paune = pynutil.add_weight(
            pynutil.delete("ਪੌਣੇ")
            + delete_space
            + integer
            + pynutil.insert("numerator: \"੩\"")
            + delete_space
            + pynutil.insert(" denominator: \"੪\""),
            -0.01,
        )

        graph_quarterly_measures = graph_saade | graph_sava | graph_paune

        graph = graph_fraction | graph_mixed_fraction | graph_quarterly_measures

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
