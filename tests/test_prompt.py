"""Tests for prompt template rendering."""

import chess
import pytest

from chessbench.models.chess_ai import ChessAI
from chessbench.prompts import KNOWN_VARIABLES


class MockChessAI(ChessAI):
    """Mock ChessAI for testing prompt rendering without API keys."""

    def __init__(self):
        super().__init__()
        self.name = "MockAI"

    async def _get_move_from_model(self, fen: str) -> str:
        return "e2e4"


class TestPromptTemplate:
    """Tests for prompt template rendering."""

    def test_prompt_template_renders_without_keyerror(self):
        """Test that _create_prompt renders without KeyError for all placeholders."""
        ai = MockChessAI()
        board = chess.Board()
        prompt = ai._create_prompt(board.fen())

        # Should not raise KeyError
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_contains_all_required_sections(self):
        """Test that prompt contains all expected sections."""
        ai = MockChessAI()
        board = chess.Board()
        prompt = ai._create_prompt(board.fen())

        required_sections = [
            "[GAME STATE]",
            "[INSTRUCTIONS]",
            "FEN:",
            "Turn:",
            "<think>",
            "<move>",
            "REASONING LEVEL:",
        ]

        for section in required_sections:
            assert section in prompt, f"Missing section: {section}"

    def test_prompt_contains_position_specific_info(self):
        """Test that prompt contains position-specific information."""
        ai = MockChessAI()
        board = chess.Board()
        prompt = ai._create_prompt(board.fen())

        # Should contain color
        assert "White" in prompt or "Black" in prompt
        assert "FEN:" in prompt

    def test_prompt_for_different_colors(self):
        """Test prompt generation for both colors."""
        ai = MockChessAI()

        # White to move
        board = chess.Board()
        white_prompt = ai._create_prompt(board.fen())
        assert "White" in white_prompt

        # Black to move
        board.push(chess.Move.from_uci("e2e4"))
        black_prompt = ai._create_prompt(board.fen())
        assert "Black" in black_prompt

    def test_prompt_fidelity_regression(self):
        """Fidelity regression: production payload keeps tags, contract, and directive intact — zero format artifacts."""
        from chessbench.prompts import create_safe_prompt_template

        system_prompt = "You are a professional chess engine playing as {color}."
        turn_prompt = """[STATE]
FEN: {fen}
Moves: {legal_moves_uci}

[INSTRUCTIONS]
Analyze the position and select the best move. Keep your reasoning concise (under 200 words).

You must format your response EXACTLY like this:
```
(Your reasoning here)
```

<move>
(Your chosen move in purely lower-case UCI notation, e.g. e2e4)
</move>

Failure to follow this exact format will result in disqualification.
IMPORTANT: Your move MUST be one of the legal UCI moves listed below."""
        template, validation = create_safe_prompt_template(system_prompt, turn_prompt)
        assert validation.is_valid, f"Template validation failed: {validation.errors}"

        ai = MockChessAI()
        ai.prompt_template = template
        messages = ai._create_messages(chess.Board().fen())
        payload = "\n".join(message.content for message in messages)

        # Template content survives rendering verbatim.
        assert "```" in payload, "Missing triple backticks section"
        assert "```\n(Your reasoning here)\n```" in payload, "Triple backticks section altered or missing"
        assert "<move>" in payload, "Missing <move> opening tag"
        assert "</move>" in payload, "Missing </move> closing tag"

        # Engine-appended layers are present and verbatim.
        assert "REASONING LEVEL:" in payload, "Missing reasoning level section"
        expected_directive = (
            "\n\n[REASONING LEVEL: HIGH]\n"
            "Perform deep step-by-step tactical calculation, candidate move evaluation, "
            "and king safety analysis in <think> tags before your move in <move>uci_move</move> tags."
        )
        assert expected_directive in payload, f"Reasoning directive altered. Expected: {expected_directive!r}"
        assert "You MUST format your output response strictly as follows:" in payload, "Missing system output contract"

        # The historical corruption: literal '<think>' must never degrade to '{}'.
        assert "in  {} tags" not in payload, "Format artifact detected: '<think>' degraded to '{}'"

    def test_prompt_with_complex_position(self):
        """Test prompt generation with a complex mid-game position."""
        ai = MockChessAI()

        # Complex position with captures available
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        prompt = ai._create_prompt(board.fen())

        assert isinstance(prompt, str)
        assert len(prompt) > 300  # Should be substantial
        assert "FEN: rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" in prompt

    def test_prompt_no_placeholder_leakage(self):
        """Test that no template placeholders remain unsubstituted."""
        ai = MockChessAI()
        board = chess.Board()
        prompt = ai._create_prompt(board.fen())

        # No unrendered placeholders should remain.
        # Check all variables that the v1_baseline template references.
        bad_patterns = [
            "{color}", "{fen}", "{ascii_board}",
            "{legal_moves_uci}", "{forcing_uci}",
            "{developing_uci}", "{positional_uci}",
            "{legal_moves_annotated}", "{last_move_san}",
            "{move_history_san}", "{white_pieces}", "{black_pieces}",
        ]

        for pattern in bad_patterns:
            assert pattern not in prompt, f"Unrendered placeholder: {pattern}"

    def test_prompt_only_computes_needed_variables(self):
        """Verify dead-weight evaluations are not called for v1_baseline."""
        from unittest.mock import patch

        ai = MockChessAI()
        board = chess.Board()

        # These evaluations are NOT referenced by v1_baseline and should
        # never be called during prompt construction.
        dead_methods = [
            "analyze_defense",
            "analyze_vulnerabilities",
            "analyze_captures",
            "analyze_king_safety",
            "analyze_undefended_pieces",
            "analyze_exposed_pieces",
            "get_material_count",
            "analyze_material_balance",
            "analyze_center_control",
            "analyze_development_status",
            "calculate_development_score",
        ]

        for method_name in dead_methods:
            with patch.object(ai.evaluator, method_name, wraps=getattr(ai.evaluator, method_name)) as mock_method:
                ai._create_prompt(board.fen())
                mock_method.assert_not_called(), f"{method_name} should not be called for v1_baseline"

    def test_create_messages_returns_system_and_user_roles(self):
        """Verify _create_messages returns system and user role ChatMessage list."""
        ai = MockChessAI()
        board = chess.Board()
        messages = ai._create_messages(board.fen())

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "professional chess engine" in messages[0].content
        assert "[GAME STATE]" in messages[1].content

    def test_rich_context_helpers(self):
        """Verify annotated legal moves, last move, move history, and piece locations."""
        ai = MockChessAI()
        board = chess.Board()

        annotated = ai._get_annotated_legal_moves(board)
        assert "e2e4 (e4)" in annotated or "g1f3 (Nf3)" in annotated

        last_move = ai._get_last_move_san(board)
        assert "None" in last_move

        board.push(chess.Move.from_uci("e2e4"))
        last_move = ai._get_last_move_san(board)
        assert "1. e4 (e2e4)" in last_move

        history = ai._get_move_history_san(board)
        assert "1. e4" in history

        w, b = ai._get_piece_locations_str(board)
        assert "King at e1" in w
        assert "King at e8" in b


class TestAnalyzePositionRepetition:
    """Regression tests for _analyze_position_repetition.

    Previously the function returned progress_score=1.0 unconditionally when
    history had fewer than 3 entries, masking early-game stagnation. The fix
    computes unique/total across the last 4 positions regardless of length.
    """

    def test_empty_history_returns_real_score(self):
        ai = MockChessAI()
        result = ai._analyze_position_repetition(chess.Board())
        assert result["progress_score"] == 1.0
        assert result["repetitions"] == 1
        assert result["is_stagnating"] is False

    def test_short_history_returns_real_score_not_always_one(self):
        ai = MockChessAI()
        # After one move, history has 1 entry; previously the function
        # short-circuited to progress_score=1.0 regardless of repetition.
        ai.move_history.append(chess.Board().fen().split(" ")[0])
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        result = ai._analyze_position_repetition(board)
        assert 0.0 < result["progress_score"] <= 1.0
        assert result["is_stagnating"] is False

    def test_repetition_triggers_stagnation_flag(self):
        ai = MockChessAI()
        # Simulate the board returning to the same FEN position.
        starting = chess.Board()
        ai.move_history.append(starting.fen().split(" ")[0])
        ai.move_history.append(starting.fen().split(" ")[0])
        ai.move_history.append(starting.fen().split(" ")[0])
        result = ai._analyze_position_repetition(starting)
        assert result["repetitions"] >= ai.stagnation_threshold
        assert result["is_stagnating"] is True
        assert result["progress_score"] < 1.0

    def test_all_unique_history_has_full_progress(self):
        ai = MockChessAI()
        board = chess.Board()
        ai.move_history.append(board.fen().split(" ")[0])
        board.push(chess.Move.from_uci("e2e4"))
        ai.move_history.append(board.fen().split(" ")[0])
        board.push(chess.Move.from_uci("e7e5"))
        ai.move_history.append(board.fen().split(" ")[0])
        board.push(chess.Move.from_uci("g1f3"))
        result = ai._analyze_position_repetition(board)
        assert result["progress_score"] == 1.0
        assert result["is_stagnating"] is False


class TestKnownVariablesProducibility:
    """Every registry variable must be producible by the context builder — no MISSING CONTEXT paths."""

    @staticmethod
    def _midgame_board() -> "chess.Board":
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
            board.push(chess.Move.from_uci(uci))
        return board

    @staticmethod
    def _template_referencing(variable: str):
        from chessbench.prompts import create_safe_prompt_template

        system_prompt = "You are a professional chess engine playing as {color}."
        turn_prompt = (
            "[STATE]\n"
            "FEN: {fen}\n"
            "Moves: {legal_moves_uci}\n"
            f"Value under test: {{{variable}}}"
        )
        template, validation = create_safe_prompt_template(system_prompt, turn_prompt)
        assert validation.is_valid, f"Fixture template invalid for {variable}: {validation.errors}"
        return template


@pytest.mark.parametrize("variable", sorted(KNOWN_VARIABLES))
def test_registry_variable_is_producible(variable):
    from chessbench.prompts.sample_context import build_sample_context

    fixture = TestKnownVariablesProducibility
    template = fixture._template_referencing(variable)
    context = build_sample_context(board=fixture._midgame_board(), prompt_template=template)

    assert variable in context, f"Registry advertises {{{variable}}} but context builder never produces it"

    rendered = template.render(context)
    assert "[MISSING CONTEXT" not in rendered, f"{{{variable}}} rendered as missing context"
    assert "[FORMAT ERROR" not in rendered, f"{{{variable}}} caused a format error"


class TestReasoningDirective:
    """Module-level directive export must match instance behavior byte-for-byte."""

    def test_returns_correct_strings(self):
        from chessbench.models.chess_ai import get_reasoning_directive

        assert get_reasoning_directive("low").startswith("\n\n[REASONING LEVEL: LOW]")
        mid = get_reasoning_directive("mid")
        high = get_reasoning_directive("high")
        assert "<think> tags" in mid
        assert "<think> tags" in high
        assert "{} tags" not in mid
        assert "{} tags" not in high

    def test_invalid_level_falls_back_to_high(self):
        from chessbench.models.chess_ai import get_reasoning_directive

        assert get_reasoning_directive("bogus") == get_reasoning_directive("high")
        assert get_reasoning_directive("") == get_reasoning_directive("high")

    def test_matches_instance_method(self):
        from chessbench.models.chess_ai import get_reasoning_directive

        ai = MockChessAI()
        for level in ("low", "mid", "high"):
            ai.reasoning_level = level
            assert ai._get_reasoning_directive() == get_reasoning_directive(level)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
