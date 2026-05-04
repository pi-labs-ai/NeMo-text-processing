# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.pa.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money (Punjabi)
        e.g. ਬਹੱਤਰ ਰੁਪਏ -> money { integer_part: "੭੨" currency: "₹" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        cardinal_single_and_double_digit_graph = cardinal.graph_digit | cardinal.graph_teens_and_ties
        decimal_graph = decimal.final_graph_wo_negative
        currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv")).invert()
        paune_graph = pynini.string_file(get_abs_path("data/numbers/paune.tsv")).invert()

        self.integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        self.integer_quarterly_measures = pynutil.insert("integer_part: \"") + cardinal_single_and_double_digit_graph
        self.integer_paune = pynutil.insert("integer_part: \"") + paune_graph
        self.paise = pynutil.insert("fractional_part: \"") + cardinal_graph + pynutil.insert("\"")
        self.fraction = decimal_graph
        self.currency = pynutil.insert("currency: \"") + currency_graph + pynutil.insert("\" ")
        
        # Punjabi word for "and"
        ate = pynutil.delete("ਅਤੇ")
        
        # Punjabi words for large numbers
        delete_hundred = pynutil.delete("ਸੌ")
        delete_lakh = pynutil.delete("ਲੱਖ")
        delete_hazar = pynutil.delete("ਹਜਾਰ") | pynutil.delete("ਹਜ਼ਾਰ")
        delete_crore = pynutil.delete("ਕਰੋੜ")

        graph_currency_decimal = self.fraction + delete_extra_space + self.currency
        graph_currency_cardinal = self.integer + delete_extra_space + self.currency

        graph_rupay_and_paisa = (
            graph_currency_cardinal
            + delete_extra_space
            + pynini.closure(ate + delete_extra_space, 0, 1)
            + self.paise
            + delete_extra_space
            + pynutil.delete(currency_graph)
        )
        
        # Cases for ਸਾਢੇ, ਸਵਾ with teens and ties
        graph_saade_teens_ties = (
            pynutil.delete("ਸਾਢੇ")
            + delete_space
            + self.integer_quarterly_measures
            + pynutil.insert("\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"੫੦\"")
            + delete_extra_space
            + self.currency
        )
        graph_sava_teens_ties = (
            pynutil.delete("ਸਵਾ")
            + delete_space
            + self.integer_quarterly_measures
            + pynutil.insert("\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"੨੫\"")
            + delete_extra_space
            + self.currency
        )
        graph_dedh = (
            pynini.union(pynutil.delete("ਡੇਢ") | pynutil.delete("ਡੇੜ"))
            + delete_space
            + pynutil.insert("integer_part: \"੧\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"੫੦\"")
            + delete_extra_space
            + self.currency
        )
        graph_dhaai = (
            pynutil.delete("ਢਾਈ")
            + delete_space
            + pynutil.insert("integer_part: \"੨\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"੫੦\"")
            + delete_extra_space
            + self.currency
        )
        graph_paune = (
            pynutil.delete("ਪੌਣੇ")
            + delete_space
            + self.integer_paune
            + pynutil.insert("\"")
            + delete_space
            + pynutil.insert(" fractional_part: \"੭੫\"")
            + delete_extra_space
            + self.currency
        )

        graph_quarterly_measures = (
            graph_saade_teens_ties | graph_sava_teens_ties | graph_dedh | graph_dhaai | graph_paune
        )

        graph = pynutil.add_weight(
            graph_currency_cardinal | graph_currency_decimal | graph_rupay_and_paisa | graph_quarterly_measures, 0.01
        )

        self.final_graph = graph
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
