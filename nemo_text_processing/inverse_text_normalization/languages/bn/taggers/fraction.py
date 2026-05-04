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


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        Fraction "/" is determined by "ভাগ" (bhag) or "বাই" (by)
            e.g. ঋণ এক ভাগ ছাব্বিশ -> fraction { negative: "true" numerator: "১" denominator: "২৬" }
            e.g. ছয় শত ষাট ভাগ পাঁচ শত তেতাল্লিশ -> fraction { negative: "false" numerator: "৬৬০" denominator: "৫৪৩" }

        The fractional rule assumes that fractions can be pronounced as:
        (a cardinal) + ('ভাগ'/'বাই') plus (a cardinal, excluding 'শূন্য')
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator
        graph_cardinal = cardinal.graph_no_exception

        integer = pynutil.insert("integer_part: \"") + graph_cardinal + pynutil.insert("\" ")
        integer += delete_space
        # Bengali: ভাগ (bhag) for "divided by", বাই (by) alternative
        delete_bata = pynini.union(pynutil.delete(" ভাগ ") | pynutil.delete(" বাই "))

        numerator = pynutil.insert("numerator: \"") + graph_cardinal + pynutil.insert("\"")
        denominator = pynutil.insert(" denominator: \"") + graph_cardinal + pynutil.insert("\"")

        graph_fraction = numerator + delete_bata + denominator
        # Bengali: সহী (sohi) or এবং (ebong) for "and" in mixed fractions
        graph_mixed_fraction = integer + delete_extra_space + pynutil.delete("সহী") + delete_space + graph_fraction

        # Bengali: সাড়ে (sare) for half-past
        graph_saade = pynutil.add_weight(
            pynutil.delete("সাড়ে")
            + delete_space
            + integer
            + pynutil.insert("numerator: \"১\"")
            + delete_space
            + pynutil.insert(" denominator: \"২\""),
            -0.01,
        )
        # Bengali: সোয়া (soya) for quarter-past
        graph_sava = pynutil.add_weight(
            pynutil.delete("সোয়া")
            + delete_space
            + integer
            + pynutil.insert("numerator: \"১\"")
            + delete_space
            + pynutil.insert(" denominator: \"৪\""),
            -0.001,
        )
        # Bengali: পৌনে (poune) for three-quarters before
        graph_paune = pynutil.add_weight(
            pynutil.delete("পৌনে")
            + delete_space
            + integer
            + pynutil.insert("numerator: \"৩\"")
            + delete_space
            + pynutil.insert(" denominator: \"৪\""),
            -0.01,
        )
        # Bengali: দেড় (der) for one and a half
        graph_dedh = pynutil.add_weight(
            pynutil.delete("দেড়")
            + delete_space
            + pynutil.insert("integer_part: \"১\"")
            + pynutil.insert(" numerator: \"১\"")
            + delete_space
            + pynutil.insert(" denominator: \"২\""),
            -0.01,
        )
        # Bengali: আড়াই (arai) for two and a half
        graph_dhaai = pynutil.add_weight(
            pynutil.delete("আড়াই")
            + delete_space
            + pynutil.insert("integer_part: \"২\"")
            + pynutil.insert(" numerator: \"১\"")
            + delete_space
            + pynutil.insert(" denominator: \"২\""),
            -0.1,
        )

        # Bengali: আধা (adha) for half
        graph_aadha_and_saade_only = (
            pynini.union(pynutil.delete("আধা") | pynutil.delete("সাড়ে"))
            + delete_space
            + pynutil.insert(" numerator: \"১\"")
            + delete_space
            + pynutil.insert(" denominator: \"২\"")
        )
        graph_sava_only = (
            pynutil.delete("সোয়া")
            + delete_space
            + pynutil.insert(" numerator: \"১\"")
            + delete_space
            + pynutil.insert(" denominator: \"৪\"")
        )
        # Bengali: পৌন/পোয়া (poun/poya) for three-quarters
        graph_paune_only = (
            pynini.union(pynutil.delete("পৌন") | pynutil.delete("পোয়া"))
            + delete_space
            + pynutil.insert("numerator: \"৩\"")
            + delete_space
            + pynutil.insert(" denominator: \"৪\"")
        )

        graph_quarterly_exceptions = (
            graph_saade
            | graph_sava
            | graph_paune
            | graph_dedh
            | graph_dhaai
            | graph_aadha_and_saade_only
            | graph_sava_only
            | graph_paune_only
        )

        final_graph = graph_quarterly_exceptions | graph_fraction | graph_mixed_fraction

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
