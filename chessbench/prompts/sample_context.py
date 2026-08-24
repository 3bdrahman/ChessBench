"""Sample context builder for generating production-identical prompt contexts."""

from __future__ import annotations

from typing import Any

import chess

from chessbench.models.evaluation import PositionEval, PositionEvaluator


def _extract_uci_moves(pos_eval: PositionEval | None) -> list[str]:
    """Extract UCI moves from a PositionEval's principal variation."""
    if not pos_eval or not pos_eval.pv:
        return []
    uci_moves = []
    for desc in pos_eval.pv:
        if "[" in desc and "]" in desc:
            uci = desc[desc.rfind("[") + 1 : desc.rfind("]")]
            if len(uci) in (4, 5):  # Valid UCI move length
                uci_moves.append(uci)
    return uci_moves


def build_sample_context(
    board: chess.Board,
    *,
    move_history: list[str] | None = None,
    reasoning_level: str = "high",
    stagnation_threshold: int = 3,
    prompt_template: Any = None,  # PromptTemplate type to avoid circular import
    evaluator: PositionEvaluator | None = None,
) -> dict[str, Any]:
    """Build a sample context dictionary identical to what _get_prompt_context would produce.

    Args:
        board: Current chess position
        move_history: List of FEN strings representing move history
        reasoning_level: Reasoning level ("low", "mid", "high")
        stagnation_threshold: Threshold for stagnation detection
        prompt_template: Template to determine which variables are needed
        evaluator: PositionEvaluator instance (creates new if None)

    Returns:
        Dictionary mapping variable names to their values for the given board state
    """
    if evaluator is None:
        evaluator = PositionEvaluator()

    # Determine which variables the active template actually references
    # so we only compute what's needed — no dead-weight evaluation.
    needed = prompt_template.referenced_variables() if prompt_template else set()

    # --- Always-cheap variables (trivial to compute) ---
    context: dict[str, Any] = {
        "fen": board.fen(),
        "color": "White" if board.turn == chess.WHITE else "Black",
        "reasoning_level": reasoning_level,
        "board": board.fen(),  # Alias for FEN board representation
    }

    # --- Rich context helpers ---
    if "legal_moves_annotated" in needed:
        context["legal_moves_annotated"] = _get_annotated_legal_moves(board)

    if "last_move_san" in needed:
        context["last_move_san"] = _get_last_move_san(board, move_history or [])

    if "move_history_san" in needed:
        context["move_history_san"] = _get_move_history_san(board, move_history or [])

    if needed & {"white_pieces", "black_pieces"}:
        w_str, b_str = _get_piece_locations_str(board, evaluator)
        context["white_pieces"] = w_str
        context["black_pieces"] = b_str

    if "legal_moves" in needed:
        context["legal_moves"] = ", ".join(m.uci() for m in board.legal_moves)

    # --- Move categorization (shared dependency) ---
    needs_moves = needed & {
        "forcing_moves", "developing_moves", "positional_moves",
        "legal_moves_uci", "forcing_uci", "developing_uci", "positional_uci",
    }
    moves = None
    if needs_moves:
        moves = evaluator.categorize_moves(board)

        if needed & {"forcing_moves", "developing_moves", "positional_moves"}:
            context["forcing_moves"] = moves["forcing_moves"]
            context["developing_moves"] = moves["developing_moves"]
            context["positional_moves"] = moves["positional_moves"]

        if needed & {"legal_moves_uci", "forcing_uci", "developing_uci", "positional_uci"}:
            forcing_uci = _extract_uci_moves(moves["forcing_moves"])
            developing_uci = _extract_uci_moves(moves["developing_moves"])
            positional_uci = _extract_uci_moves(moves["positional_moves"])

            context["legal_moves_uci"] = " ".join(forcing_uci + developing_uci + positional_uci)
            context["forcing_uci"] = " ".join(forcing_uci)
            context["developing_uci"] = " ".join(developing_uci)
            context["positional_uci"] = " ".join(positional_uci)

    # --- Board representation ---
    if "ascii_board" in needed:
        context["ascii_board"] = str(board)

    # --- Repetition / stagnation analysis ---
    if needed & {"position_repetitions", "stagnation_status", "position_progress"}:
        position_analysis = _analyze_position_repetition(
            board, move_history or [], stagnation_threshold
        )
        context["position_repetitions"] = position_analysis["repetitions"]
        context["stagnation_status"] = (
            "STAGNATING - Force dynamic play!" if position_analysis["is_stagnating"] else "Normal"
        )
        context["position_progress"] = f"{position_analysis['progress_score']:.2f}"

    # --- Evaluations: only computed when the template references them ---
    _eval_map: dict[str, Any] = {
        "material_tension": lambda: str(evaluator.analyze_material_tension(board)),
        "position_dynamism": lambda: str(evaluator.analyze_position_dynamism(board)),
        "development_score": lambda: str(evaluator.calculate_development_score(board)),
        "defense_analysis": lambda: str(evaluator.analyze_defense(board)),
        "vulnerability_analysis": lambda: str(evaluator.analyze_vulnerabilities(board)),
        "capture_analysis": lambda: str(evaluator.analyze_captures(board)),
        "king_safety": lambda: str(evaluator.analyze_king_safety(board)),
        "undefended_pieces": lambda: str(evaluator.analyze_undefended_pieces(board)),
        "exposed_pieces": lambda: str(evaluator.analyze_exposed_pieces(board)),
        "material_count": lambda: str(evaluator.get_material_count(board)),
        "material_balance": lambda: str(evaluator.analyze_material_balance(board)),
        "center_control": lambda: str(evaluator.analyze_center_control(board)),
        "development_status": lambda: str(evaluator.analyze_development_status(board)),
    }
    for var_name in needed & _eval_map.keys():
        context[var_name] = _eval_map[var_name]()

    return context


def _get_annotated_legal_moves(board: chess.Board) -> str:
    """Format legal moves pairing UCI with SAN, e.g., 'c8b7 (Bxb7), f6e5 (fxe5)'."""
    moves = [f"{m.uci()} ({board.san(m)})" for m in board.legal_moves]
    return ", ".join(moves)


def _get_last_move_san(board: chess.Board, move_history: list[str]) -> str:
    """Format opponent's previous move in SAN and UCI, e.g. '2... Nc6 (b8c6)'."""
    if not move_history:
        return "None (First move of the game)"
    # For sample context without move replay capability, we cannot determine
    # the actual last move, but we return a format-consistent placeholder
    # In production with full move history, this would return the actual last move
    return "None (Move history not available in sample context)"


def _get_move_history_san(board: chess.Board, move_history: list[str], max_moves: int = 10) -> str:
    """Reconstruct SAN game history, e.g. '1. e4 e5 2. Nf3 Nc6'."""
    if not move_history:
        return "None (Starting position)"
    # For sample context without move replay capability, we cannot determine
    # the actual move history, but we return a format-consistent placeholder
    # In production with full move history, this would return the actual SAN history
    return "... (Move history reconstruction not available in sample context)"


def _get_piece_locations_str(board: chess.Board, evaluator: PositionEvaluator) -> tuple[str, str]:
    """Format piece locations for White and Black."""
    w, b = evaluator.get_piece_locations(board)
    return ", ".join(w), ", ".join(b)


def _analyze_position_repetition(
    board: chess.Board,
    move_history: list[str],
    stagnation_threshold: int
) -> dict[str, Any]:
    """Analyze position repetition and stagnation."""
    current_fen = board.fen().split(' ')[0]

    recent_history = [*move_history[-7:], current_fen]
    repetitions = sum(1 for pos in recent_history if pos == current_fen)

    is_stagnating = repetitions >= stagnation_threshold

    recent_positions = [*move_history[-3:], current_fen]
    unique_positions = len(set(recent_positions))
    progress_score = unique_positions / len(recent_positions) if recent_positions else 0.0

    return {
        "repetitions": repetitions,
        "is_stagnating": is_stagnating,
        "progress_score": progress_score
    }
