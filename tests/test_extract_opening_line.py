from schemas import extract_opening_line


class TestBasicExtraction:
    def test_simple_sentence(self) -> None:
        text = "The room was empty and silent. She walked to the window."
        assert extract_opening_line(text) == "The room was empty and silent."

    def test_exclamation_mark(self) -> None:
        text = "What a terrible thing to behold! He turned away in horror."
        assert extract_opening_line(text) == "What a terrible thing to behold!"

    def test_question_mark(self) -> None:
        text = "Where had the years gone? She could not remember."
        assert extract_opening_line(text) == "Where had the years gone?"

    def test_newlines_replaced(self) -> None:
        text = "The wind howled through\nthe broken windows. It was winter."
        assert extract_opening_line(text) == "The wind howled through the broken windows."


class TestAbbreviations:
    def test_mr_not_split(self) -> None:
        text = "Mr. Smith walked into the garden. The sun was setting."
        assert extract_opening_line(text) == "Mr. Smith walked into the garden."

    def test_mrs_not_split(self) -> None:
        text = "Mrs. Dalloway said she would buy the flowers herself. And indeed she did."
        assert extract_opening_line(text) == "Mrs. Dalloway said she would buy the flowers herself."

    def test_dr_not_split(self) -> None:
        text = "Dr. Frankenstein had not slept in days. The creature stirred."
        assert extract_opening_line(text) == "Dr. Frankenstein had not slept in days."

    def test_multiple_abbreviations(self) -> None:
        text = "Mr. and Mrs. Bennet had five daughters. They lived in Longbourn."
        assert extract_opening_line(text) == "Mr. and Mrs. Bennet had five daughters."


class TestFallback:
    def test_no_sentence_boundary_truncates_at_word(self) -> None:
        text = "a " * 150  # no period, just words
        result = extract_opening_line(text, max_chars=50)
        assert result.endswith("\u2026")
        assert len(result) <= 52  # 50 chars + possible trailing ellipsis
        assert "  " not in result  # no broken words

    def test_truncation_does_not_break_mid_word(self) -> None:
        words = "extraordinary " * 20
        result = extract_opening_line(words, max_chars=50)
        assert result.endswith("\u2026")
        without_ellipsis = result[:-1]
        assert without_ellipsis.endswith("extraordinary")

    def test_short_first_sentence_not_used_as_boundary(self) -> None:
        text = "Hi. The morning light crept across the cold stone floor. She rose slowly."
        result = extract_opening_line(text)
        assert result == "Hi. The morning light crept across the cold stone floor."


class TestLongSentences:
    def test_long_sentence_within_scan_window(self) -> None:
        long_opening = "The studio was filled with the rich odour of roses " * 8
        long_opening = long_opening.strip() + ". The next sentence."
        result = extract_opening_line(long_opening)
        assert result.endswith(".")
        assert "The next sentence" not in result

    def test_quoted_sentence_ending(self) -> None:
        text = 'He whispered "goodbye." She did not look back.'
        result = extract_opening_line(text)
        assert result == 'He whispered "goodbye."'
