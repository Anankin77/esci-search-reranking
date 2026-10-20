"""Number/unit, size, colour and negation parsing on tricky examples."""
from text_utils import (clean_text, colour_tokens, has_negation_cue, negated_terms,
                        negation_features, normalize_query, number_unit_features,
                        parse_numbers_units, size_features, size_tokens, tokenize)


# ---------------------------------------------------------------- cleaning
def test_clean_text_strips_html_and_entities():
    assert clean_text("<b>Soft</b>&nbsp;&amp; warm<br/>blanket") == "Soft & warm blanket"
    assert clean_text("&lt;p&gt;escaped tag&lt;/p&gt;") == "escaped tag"
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_clean_text_nfkc_and_budget():
    assert clean_text("ﬁne  \n  print") == "fine print"   # ligature -> 'fi'
    assert clean_text("one two three", budget=9) == "one two"


def test_tokenize_keeps_decimals():
    assert tokenize("Size 10.5, 2-Pack!") == ["size", "10.5", "2", "pack"]


def test_tokenize_drops_inner_apostrophes():
    assert tokenize("Levi's Men’s jeans") == ["levis", "mens", "jeans"]


def test_normalize_query_sorts_tokens():
    assert normalize_query("Red  Shoes, Men's") == "mens red shoes"
    assert normalize_query("b a") == normalize_query("A, B!")


# ---------------------------------------------------------------- numbers + units
def test_number_unit_parsing():
    assert (12.0, "oz") in parse_numbers_units("Coffee 12 oz bag")
    assert (12.0, "oz") in parse_numbers_units("12oz")
    assert (6.0, "pack") in parse_numbers_units("6-pack of soda")
    assert (4.0, "pack") in parse_numbers_units("pack of 4 batteries")
    assert (2.0, "tb") in parse_numbers_units("2TB hard drive")
    assert (65.0, "inch") in parse_numbers_units('65" tv')
    assert (9.0, "size") in parse_numbers_units("nike size 9")
    assert (5.0, None) in parse_numbers_units("iphone 5")


def test_two_in_one_is_not_inches():
    assert (2.0, "inch") not in parse_numbers_units("2 in 1 laptop")


def test_unit_does_not_eat_following_word():
    # "5 large" must not read as 5 litres
    assert (5.0, "l") not in parse_numbers_units("5 large eggs")


def test_number_unit_features():
    f = number_unit_features("12 oz coffee", "Coffee Beans, 12 Ounce Bag")
    assert f["num_match"] == 1 and f["unit_match"] == 1 and f["unit_conflict"] == 0
    f = number_unit_features("12 oz coffee", "Coffee Beans, 32 oz Bag")
    assert f["num_conflict"] == 1 and f["unit_conflict"] == 1 and f["unit_match"] == 0
    f = number_unit_features("coffee", "Coffee 32 oz")
    assert sum(f.values()) == 0  # query without numbers: all zeros


# ---------------------------------------------------------------- sizes, colours
def test_size_tokens():
    assert size_tokens("Mens shirt XL") == {"xl"}
    assert size_tokens("running shoes size 9.0") == {"size:9"}
    assert "l" in size_tokens("large dog bed")
    assert size_tokens("xlr cable") == set()  # 'xl' inside another word


def test_size_features():
    assert size_features("tshirt xl", "T-Shirt, X-Large")["size_match"] == 1
    assert size_features("tshirt xl", "T-Shirt, Small")["size_conflict"] == 1


def test_colour_tokens():
    assert colour_tokens("Gray hoodie, navy trim") == {"grey", "navy"}
    assert colour_tokens("redwood table") == set()  # whole tokens only


# ---------------------------------------------------------------- negation
def test_negated_terms():
    assert negated_terms("shampoo without sulfate") == ["sulfate"]
    assert negated_terms("gluten-free pasta") == ["gluten"]
    assert negated_terms("sugar free gum") == ["sugar"]
    assert negated_terms("non toxic paint") == ["toxic"]
    assert negated_terms("hands free headset") == []  # 'hands free' is not a negation of hands


def test_negation_cue():
    assert has_negation_cue("dairy-free cheese")
    assert has_negation_cue("phone case not pink")
    assert not has_negation_cue("nothing bundt cakes")  # 'nothing' is not 'not'


def test_negation_features():
    # the title states the same negation -> good sign, not a conflict
    f = negation_features("sulfate free shampoo", "Shampoo, Sulfate-Free, 12 oz")
    assert f == {"neg_cue": 1, "neg_phrase_in_title": 1, "neg_term_conflict": 0}
    # the title contains the avoided term without negation -> conflict
    f = negation_features("shampoo without sulfate", "Shampoo with Sodium Lauryl Sulfate")
    assert f["neg_term_conflict"] == 1
    f = negation_features("blue shirt", "Blue Shirt")
    assert f == {"neg_cue": 0, "neg_phrase_in_title": 0, "neg_term_conflict": 0}
