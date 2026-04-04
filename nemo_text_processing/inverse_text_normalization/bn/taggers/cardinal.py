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
    MINUS,
    NEMO_BN_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.bn.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. ঋণ তেইশ -> cardinal { integer: "২৩" negative: "-" }
        e.g. এক শত তেইশ -> cardinal { integer: "১২৩" }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="cardinal", kind="classify")
        self.input_case = input_case
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_teens_and_ties = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv")).invert()
        graph_paune = pynini.string_file(get_abs_path("data/numbers/paune.tsv")).invert()
        self.graph_zero = graph_zero
        self.graph_digit = graph_digit
        self.graph_single_digit_with_zero = pynutil.insert("০") + graph_digit
        self.graph_teens_and_ties = graph_teens_and_ties
        self.graph_two_digit = graph_teens_and_ties | (pynutil.insert("০") + graph_digit)
        
        # Bengali: শত (shoto) for hundred
        graph_hundred = (
            pynini.cross("শত", "") | pynini.cross("শো", "") | pynini.cross("শ", "")
        )
        delete_hundred = pynutil.delete("শত") | pynutil.delete("শো") | pynutil.delete("শ")
        
        # Bengali compound hundreds: একশো, নয়শো, etc.
        graph_compound_hundred = pynini.string_file(get_abs_path("data/numbers/compound_hundreds.tsv")).invert()
        
        # Bengali: হাজার (hajar) for thousand
        delete_thousand = pynutil.delete("হাজার")
        
        # Three-digit component (hundreds place)
        graph_hundred_component = pynini.union(
            graph_digit + delete_space + graph_hundred,
            graph_compound_hundred,
            pynutil.insert("০")
        )
        graph_hundred_component += delete_space
        graph_hundred_component += self.graph_two_digit | pynutil.insert("০০")
        
        # Compound hundred with two-digit: নয়শো সাতান্ন -> ৯৫৭
        graph_compound_hundred_with_two_digit = (
            graph_compound_hundred + delete_space + (self.graph_two_digit | pynutil.insert("০০"))
        )
        
        # Four-digit component for colloquial patterns like "নয় নয়শো সাতান্ন" = 9957
        # This is digit (thousands) + compound_hundred + two_digit (hundreds place)
        # e.g., "নয় নয়শো সাতান্ন" = 9 (thousands) + 9 (hundreds) + 57 (tens+units) = 9957
        graph_four_digit_colloquial = (
            graph_digit + delete_space + graph_compound_hundred_with_two_digit
        )
        self.graph_four_digit_colloquial = graph_four_digit_colloquial
        
        # Also expose graph_compound_hundred for telephone use
        self.graph_compound_hundred = graph_compound_hundred
        self.graph_compound_hundred_with_two_digit = graph_compound_hundred_with_two_digit
        
        # Four-digit component (thousands place before multipliers like crore, lakh)
        # For numbers like "নয় নয়শো সাতান্ন" (9957) = নয় (9) + নয়শো (900) + সাতান্ন (57)
        graph_four_digit = pynini.union(
            graph_digit + delete_space + graph_hundred_component,
            graph_digit + delete_space + delete_thousand + delete_space + graph_hundred_component,
            pynutil.insert("০") + graph_hundred_component
        )
        
        # Flexible number before multipliers (1-4 digits)
        graph_before_multiplier = pynini.union(
            graph_four_digit,
            pynutil.insert("০") + graph_hundred_component,
            pynutil.insert("০০") + self.graph_two_digit,
            pynutil.insert("০০০") + graph_digit,
        )

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_BN_DIGIT) + (NEMO_BN_DIGIT - "০") + pynini.closure(NEMO_BN_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        # Transducer for eleven hundred -> 1100 or twenty one hundred eleven -> 2111
        graph_hundred_as_thousand = pynini.union(
            graph_teens_and_ties + delete_space + graph_hundred, pynutil.insert("০")
        )
        graph_hundred_as_thousand += delete_space
        graph_hundred_as_thousand += self.graph_two_digit | pynutil.insert("০০")
        
        # Bengali: সাড়ে (sare) for half-past quantities (e.g., সাড়ে তিন = 3.5)
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("সাড়ে")
            + delete_space
            + graph_digit
            + pynutil.insert("৫০০", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )
        # Bengali: সোয়া (soya) for quarter-past quantities (e.g., সোয়া তিন = 3.25)
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("সোয়া")
            + delete_space
            + graph_digit
            + pynutil.insert("২৫০", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )
        # Bengali: পৌনে (poune) for three-quarter quantities (e.g., পৌনে তিন = 2.75 -> input 3)
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("পৌনে")
            + delete_space
            + graph_paune
            + pynutil.insert("৭৫০", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )
        # Bengali: দেড় (der) for one and a half
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("দেড়")
            + delete_space
            + pynutil.insert("১৫০০", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )
        # Bengali: আড়াই (arai) for two and a half
        graph_hundred_as_thousand |= pynutil.add_weight(
            pynutil.delete("আড়াই")
            + delete_space
            + pynutil.insert("২৫০০", weight=-0.1)
            + delete_space
            + delete_thousand,
            -0.1,
        )

        graph_in_hundreds = pynutil.add_weight(
            pynutil.delete("সাড়ে")
            + delete_space
            + (graph_digit | self.graph_two_digit)
            + pynutil.insert("৫০", weight=-0.1)
            + delete_space
            + delete_hundred,
            -0.1,
        )
        graph_in_hundreds |= pynutil.add_weight(
            pynutil.delete("সোয়া")
            + delete_space
            + (graph_digit | self.graph_two_digit)
            + pynutil.insert("২৫", weight=-0.1)
            + delete_space
            + delete_hundred,
            -0.1,
        )
        graph_in_hundreds |= pynutil.add_weight(
            pynutil.delete("পৌনে")
            + delete_space
            + graph_paune
            + pynutil.insert("৭৫", weight=-0.1)
            + delete_space
            + delete_hundred,
            -0.1,
        )
        graph_in_hundreds |= pynutil.add_weight(
            pynutil.delete("দেড়")
            + delete_space
            + pynutil.insert("১৫০", weight=-0.1)
            + delete_space
            + delete_hundred,
            -0.1,
        )
        graph_in_hundreds |= pynutil.add_weight(
            pynutil.delete("আড়াই") + delete_space + pynutil.insert("২৫০", weight=-0.1) + delete_space + delete_hundred,
            -0.1,
        )
        self.graph_hundreds = (
            graph_four_digit_colloquial
            | graph_hundred_component
            | graph_hundred_as_thousand
            | graph_in_hundreds
        )

        graph_teens_and_ties_component = pynini.union(
            graph_teens_and_ties | pynutil.insert("০০") + delete_space + (graph_digit | pynutil.insert("০")),
        )
        graph_ties_component_at_least_one_none_zero_digit = self.graph_two_digit @ (
            pynini.closure(NEMO_BN_DIGIT) + pynini.closure(NEMO_BN_DIGIT)
        )
        self.graph_ties_component_at_least_one_none_zero_digit = graph_ties_component_at_least_one_none_zero_digit

        # %% Indian numeric format simple https://en.wikipedia.org/wiki/Indian_numbering_system
        # This only covers "standard format".
        # Conventional format like thousand crores/lakh crores is yet to be implemented
        graph_in_thousands = pynini.union(
            self.graph_two_digit + delete_space + delete_thousand,  # For 2 digit like "চুয়াল্লিশ হাজার" or "চার হাজার"
            pynutil.insert("০০", weight=0.1),  # Default when no thousands present
        )
        self.graph_thousands = graph_in_thousands

        # Bengali: লক্ষ (lokkho) or লাখ (lakh) for lakh (100,000)
        graph_in_lakhs = pynini.union(
            self.graph_two_digit + delete_space + (pynutil.delete("লক্ষ") | pynutil.delete("লাখ")),
            pynutil.insert("০০", weight=0.1),  # Default when no lakhs present
        )

        # Bengali: কোটি (koti) for crore (10,000,000)
        # Support colloquial patterns like "নয় নয়শো সাতান্ন কোটি" (9957 crore)
        graph_in_crores = pynini.union(
            graph_four_digit_colloquial + delete_space + pynutil.delete("কোটি"),  # 4 digit colloquial
            graph_hundred_component + delete_space + pynutil.delete("কোটি"),  # 3 digit
            self.graph_two_digit + delete_space + pynutil.delete("কোটি"),  # 2 digit
            pynutil.insert("০০", weight=0.1),
        )

        # Bengali: আরব (arob) for arab
        graph_in_arabs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("আরব"),
            pynutil.insert("০০", weight=0.1),
        )

        # Bengali: খরব (khorob) for kharab
        graph_in_kharabs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("খরব"),
            pynutil.insert("০০", weight=0.1),
        )

        # Bengali: নীল (nil) for nil
        graph_in_nils = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("নীল"),
            pynutil.insert("০০", weight=0.1),
        )

        # Bengali: পদ্ম (podmo) for padma
        graph_in_padmas = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("পদ্ম"),
            pynutil.insert("০০", weight=0.1),
        )

        # Bengali: শঙ্খ (shongkho) for shankh
        graph_in_shankhs = pynini.union(
            self.graph_two_digit + delete_space + pynutil.delete("শঙ্খ"),
            pynutil.insert("০০", weight=0.1),
        )

        graph_ind = (
            graph_in_shankhs
            + delete_space
            + graph_in_padmas
            + delete_space
            + graph_in_nils
            + delete_space
            + graph_in_kharabs
            + delete_space
            + graph_in_arabs
            + delete_space
            + graph_in_crores
            + delete_space
            + graph_in_lakhs
            + delete_space
            + graph_in_thousands
        )
        graph_no_prefix = pynutil.add_weight(
            pynini.cross("শত", "১০০")
            | pynini.cross("হাজার", "১০০০")
            | pynini.cross("লক্ষ", "১০০০০০")
            | pynini.cross("লাখ", "১০০০০০")
            | pynini.cross("কোটি", "১০০০০০০০"),
            2,
        )

        graph = pynini.union(
            graph_ind + delete_space + self.graph_hundreds, graph_zero, graph_no_prefix
        )

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("০"))
            + pynini.difference(NEMO_BN_DIGIT, "০")
            + pynini.closure(NEMO_BN_DIGIT),
            "০",
        )

        labels_exception = [pynini.string_file(get_abs_path("data/numbers/labels_exception.tsv"))]

        graph_exception = pynini.union(*labels_exception).optimize()

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
