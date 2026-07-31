"""Sentence splitter contract, written before tuning.

Each case is drawn from patterns that break naive splitters in financial
prose: dollar amounts, corporate abbreviations, list markers, decimal
percentages, latin abbreviations, and whitespace-flattened table text. Tune
ticker.sentence_split against this file, not the other way around.
"""

from __future__ import annotations

from ticker.sentence_split import split_sentences


def test_dollar_amount_with_decimal_does_not_split_mid_number():
    text = (
        "Net revenue was $1.5 million in the current quarter. The prior "
        "quarter was $1.2 million."
    )
    result = split_sentences(text)
    assert result == [
        "Net revenue was $1.5 million in the current quarter.",
        "The prior quarter was $1.2 million.",
    ]


def test_billion_dollar_amount_stays_intact():
    text = (
        "The Company reported revenue of $0.5 billion, up from $0.4 billion "
        "a year earlier."
    )
    result = split_sentences(text)
    assert len(result) == 1
    assert result[0] == text


def test_inc_mid_sentence_does_not_split():
    text = (
        "The Company entered into a supply agreement with Acme Inc. covering "
        "the next three fiscal years."
    )
    assert len(split_sentences(text)) == 1


def test_corp_mid_sentence_does_not_split():
    text = "Intel Corp. reported quarterly revenue growth driven by data center demand."
    assert len(split_sentences(text)) == 1


def test_us_abbreviation_does_not_split():
    text = (
        "Substantially all of our manufacturing operations are located in "
        "the U.S. and Southeast Asia."
    )
    assert len(split_sentences(text)) == 1


def test_us_gaap_does_not_split():
    text = (
        "The financial statements are prepared in accordance with U.S. GAAP "
        "and are unaudited."
    )
    assert len(split_sentences(text)) == 1


def test_no_abbreviation_does_not_split():
    text = "See Note No. 3 to the consolidated financial statements for additional detail."
    assert len(split_sentences(text)) == 1


def test_numbered_item_heading_does_not_shred():
    text = (
        "Item 1. Business We design, develop, and market semiconductor "
        "products worldwide."
    )
    result = split_sentences(text)
    assert not any(len(s.split()) <= 2 for s in result)


def test_decimal_percentage_does_not_split():
    text = (
        "Gross margin was 45.2% in the current quarter compared to 43.8% in "
        "the prior quarter."
    )
    assert len(split_sentences(text)) == 1


def test_multiple_decimal_percentages_across_sentences():
    text = (
        "Gross margin expanded to 45.2% from 43.8%. Operating margin was "
        "21.5%, up from 19.1% a year ago."
    )
    assert len(split_sentences(text)) == 2


def test_eg_does_not_split():
    text = (
        "The Company monitors several non-GAAP measures, e.g. Adjusted "
        "EBITDA, on a quarterly basis."
    )
    assert len(split_sentences(text)) == 1


def test_ie_does_not_split():
    text = (
        "Working capital, i.e. Current assets less current liabilities, "
        "improved year over year."
    )
    assert len(split_sentences(text)) == 1


def test_eg_between_two_real_sentences_still_splits_at_the_period():
    text = (
        "Certain jurisdictions, e.g. California and New York, have enacted "
        "new privacy regulations. We are evaluating the impact on our "
        "operations."
    )
    result = split_sentences(text)
    assert len(result) == 2
    assert result[1] == "We are evaluating the impact on our operations."


def test_tabular_fragment_is_not_shredded_into_one_to_three_token_pieces():
    # Whitespace-flattened financial-statement table: numeric cells with
    # decimal points immediately followed by a capitalized row label, the
    # classic case that shreds a naive splitter into single-token sentences.
    text = (
        "Revenue 1,245.6 1,102.3 Cost of revenue 567.8 543.2 Gross profit "
        "677.8 559.1 Net income 312.4 289.1 Diluted EPS 1.25 1.10"
    )
    result = split_sentences(text)
    short = [s for s in result if len(s.split()) <= 3]
    assert not short, f"table fragment shredded into short pieces: {short}"


def test_dollar_sign_table_row_not_shredded():
    text = (
        "Cash and cash equivalents $ 1,204.3 $ 987.1 Short-term investments "
        "$ 402.5 $ 388.0"
    )
    result = split_sentences(text)
    short = [s for s in result if len(s.split()) <= 3]
    assert not short, f"table fragment shredded into short pieces: {short}"


def test_empty_text_returns_empty_list():
    assert split_sentences("") == []


def test_whitespace_only_returns_empty_list():
    assert split_sentences("   \n\t  ") == []
