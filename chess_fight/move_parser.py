"""Robust UCI move extraction from LLM outputs with promotion, disambiguation, and SAN support."""

from __future__ import annotations

import re

import chess

from chess_fight.common.common_types import MoveParseResult

# Promotion pieces
PROMOTION_PIECES = {
    'q': chess.QUEEN,
    'r': chess.ROOK,
    'b': chess.BISHOP,
    'n': chess.KNIGHT,
    'Q': chess.QUEEN,
    'R': chess.ROOK,
    'B': chess.BISHOP,
    'N': chess.KNIGHT,
}


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text, including unclosed think tags."""
    text = re.sub(r'<(?:think|thinking)>.*?</(?:think|thinking)>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<(?:think|thinking)>.*?(?=<move>)', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text


def _parse_move_candidate(candidate: str, board: chess.Board | None = None) -> MoveParseResult | None:
    """Parse a single move candidate string as UCI or SAN."""
    if not candidate:
        return None

    clean = candidate.strip().strip("`'\"*.:;")

    if board:
        # Try UCI
        res = _parse_uci(clean, board)
        if res and res.uci:
            return res
        # Try SAN
        res = _parse_san(clean, board)
        if res and res.uci:
            return res
        # Try natural language
        res = _parse_natural_language(clean, board)
        if res and res.uci:
            return res
    else:
        uci_match = re.search(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', clean.lower())
        if uci_match:
            return MoveParseResult(uci=uci_match.group(1), san=None, confidence=1.0, ambiguous=False)

    return None


def _parse_san(text: str, board: chess.Board) -> MoveParseResult | None:
    """
    Parse Standard Algebraic Notation (SAN) from text.
    Handles: Nf3, exd5, O-O, O-O-O, e8=Q, etc.
    """
    # Clean up text
    text = text.strip()

    # Castle patterns: O-O-O (queenside) MUST be checked before O-O (kingside)
    # because the kingside regex `\bO[-\s]?O\b` matches the O-O inside O-O-O
    # — checking queenside first avoids mis-firing.
    if re.search(r'\bO[-\s]?O[-\s]?O\b', text, re.IGNORECASE):
        # Queenside castle
        for move in board.legal_moves:
            if board.is_castling(move) and move.to_square < move.from_square:
                return MoveParseResult(
                    uci=move.uci(),
                    san=board.san(move),
                    confidence=0.9,
                    ambiguous=False,
                )
    if re.search(r'\bO[-\s]?O\b', text, re.IGNORECASE):
        # Kingside castle
        for move in board.legal_moves:
            if board.is_castling(move) and move.to_square > move.from_square:
                return MoveParseResult(
                    uci=move.uci(),
                    san=board.san(move),
                    confidence=0.9,
                    ambiguous=False,
                )

    # SAN pattern: [KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][=QRBN]?[+#]?
    san_pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b'
    matches = re.findall(san_pattern, text)

    for match in matches:
        try:
            move = board.parse_san(match)
            if move in board.legal_moves:
                return MoveParseResult(
                    uci=move.uci(),
                    san=match,
                    confidence=0.95,
                    ambiguous=False,
                    promotion_piece=(move.promotion and chess.piece_name(move.promotion).lower()) or None,
                )
        except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            continue

    return None


def _parse_uci(text: str, board: chess.Board | None = None) -> MoveParseResult | None:
    """Parse UCI notation from text."""
    text = text.lower()

    # UCI pattern: [a-h][1-8][a-h][1-8][qrbn]?
    uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
    matches = re.findall(uci_pattern, text)

    for match in matches:
        try:
            move = chess.Move.from_uci(match)
            promotion_piece = None
            if move.promotion:
                promotion_piece = chess.piece_name(move.promotion).lower()

            if board:
                if move in board.legal_moves:
                    return MoveParseResult(
                        uci=match,
                        san=board.san(move),
                        confidence=1.0,
                        ambiguous=False,
                        promotion_piece=promotion_piece,
                    )
            else:
                return MoveParseResult(
                    uci=match,
                    san=None,
                    confidence=0.9,
                    ambiguous=False,
                    promotion_piece=promotion_piece,
                )
        except ValueError:
            continue

    return None


def _piece_name_to_type(name: str) -> chess.PieceType | None:
    """Convert piece name to PieceType."""
    name = name.lower()
    mapping = {
        'knight': chess.KNIGHT,
        'bishop': chess.BISHOP,
        'rook': chess.ROOK,
        'queen': chess.QUEEN,
        'king': chess.KING,
        'pawn': chess.PAWN,
    }
    return mapping.get(name)


def _parse_natural_language(text: str, board: chess.Board) -> MoveParseResult | None:
    """
    Parse natural language move descriptions.
    Handles: "knight to f3", "move pawn to e4", "capture on d5", etc.
    """
    text = text.lower()

    # Patterns for natural language
    patterns = [
        # "knight to f3", "bishop to e7", etc.
        r'(?:move|play|choose|will play)\s+(?:the\s+)?(knight|bishop|rook|queen|king|pawn)\s+to\s+([a-h][1-8])',
        # "knight f3", "bishop e7"
        r'\b(knight|bishop|rook|queen|king|pawn)\s+([a-h][1-8])\b',
        # "to f3", "to e4"
        r'(?:move|play|choose|will play)\s+to\s+([a-h][1-8])',
        # "capture on d5", "take on e4"
        r'(?:capture|take)\s+on\s+([a-h][1-8])',
        # "move e2e4", "play e2e4"
        r'(?:move|play)\s+([a-h][1-8][a-h][1-8][qrbn]?)',
        # "best move e2e4"
        r'best move.{0,10}([a-h][1-8][a-h][1-8][qrbn]?)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in reversed(matches):
            if isinstance(match, tuple):
                if len(match) == 2:
                    piece_name, target_square = match
                    piece_type = _piece_name_to_type(piece_name)
                    if piece_type:
                        # Find moves of this piece type to target square
                        candidates = [
                            m for m in board.legal_moves
                            if chess.square_name(m.to_square) == target_square
                            and (piece := board.piece_at(m.from_square)) is not None
                            and piece.piece_type == piece_type
                        ]
                        if len(candidates) == 1:
                            move = candidates[0]
                            return MoveParseResult(
                                uci=move.uci(),
                                san=board.san(move),
                                confidence=0.8,
                                ambiguous=False,
                                promotion_piece=(move.promotion and chess.piece_name(move.promotion).lower()) or None,
                            )
                        elif len(candidates) > 1:
                            # Ambiguous - return first with low confidence
                            move = candidates[0]
                            return MoveParseResult(
                                uci=move.uci(),
                                san=board.san(move),
                                confidence=0.4,
                                ambiguous=True,
                                promotion_piece=(move.promotion and chess.piece_name(move.promotion).lower()) or None,
                            )
                elif len(match) == 1:
                    target_square = match[0]
                    # Just target square - very ambiguous
                    candidates = [
                        m for m in board.legal_moves
                        if chess.square_name(m.to_square) == target_square
                    ]
                    if len(candidates) == 1:
                        move = candidates[0]
                        return MoveParseResult(
                            uci=move.uci(),
                            san=board.san(move),
                            confidence=0.3,
                            ambiguous=True,
                            promotion_piece=(move.promotion and chess.piece_name(move.promotion).lower()) or None,
                        )
            else:
                # Single string match - could be UCI
                candidate = match
                if len(candidate) in (4, 5):
                    try:
                        move = chess.Move.from_uci(candidate)
                        if move in board.legal_moves:
                            promotion_piece = None
                            if move.promotion:
                                promotion_piece = chess.piece_name(move.promotion).lower()
                            return MoveParseResult(
                                uci=candidate,
                                san=board.san(move),
                                confidence=0.7,
                                ambiguous=False,
                                promotion_piece=promotion_piece,
                            )
                    except ValueError:
                        pass

    return None


def _resolve_disambiguation(text: str, board: chess.Board, candidates: list[chess.Move]) -> MoveParseResult | None:
    """
    Resolve ambiguous moves using context clues from the text.
    E.g., if two knights can go to d2, prefer "knight from b1" or "knight from f3"
    """
    if len(candidates) <= 1:
        return None

    text = text.lower()

    # Look for from-square hints: "from b1", "from f3", "b1 to d2", "f3d2"
    from_square_pattern = r'(?:from\s+)?([a-h][1-8])\s*(?:to|-)'
    from_matches = re.findall(from_square_pattern, text)

    for from_sq in from_matches:
        from_square = chess.parse_square(from_sq)
        filtered = [m for m in candidates if m.from_square == from_square]
        if len(filtered) == 1:
            move = filtered[0]
            return MoveParseResult(
                uci=move.uci(),
                san=board.san(move),
                confidence=0.85,
                ambiguous=False,
                promotion_piece=(move.promotion and chess.piece_name(move.promotion).lower()) or None,
            )

    # Look for piece-specific hints: "knight from b1", "white knight", "black bishop"
    piece_hints = {
        'white knight': chess.KNIGHT,
        'black knight': chess.KNIGHT,
        'white bishop': chess.BISHOP,
        'black bishop': chess.BISHOP,
        'white rook': chess.ROOK,
        'black rook': chess.ROOK,
        'white queen': chess.QUEEN,
        'black queen': chess.QUEEN,
        'white king': chess.KING,
        'black king': chess.KING,
    }

    for hint, piece_type in piece_hints.items():
        if hint in text:
            # Filter by piece type and color
            color = chess.WHITE if 'white' in hint else chess.BLACK
            filtered = [
                m for m in candidates
                if (piece := board.piece_at(m.from_square)) is not None
                and piece.piece_type == piece_type
                and piece.color == color
            ]
            if len(filtered) == 1:
                move = filtered[0]
                return MoveParseResult(
                    uci=move.uci(),
                    san=board.san(move),
                    confidence=0.8,
                    ambiguous=False,
                    promotion_piece=(move.promotion and chess.piece_name(move.promotion).lower()) or None,
                )

    return None


def parse_move(text: str, board: chess.Board | None = None) -> MoveParseResult:
    """
    Main entry point: parse a move from LLM output.

    Args:
        text: Raw LLM response text
        board: Optional chess board for validation and disambiguation

    Returns:
        MoveParseResult with uci, san, confidence, ambiguous, promotion_piece
    """
    if not text:
        return MoveParseResult(uci=None, san=None, confidence=0.0, ambiguous=False)

    import json
    try:
        # Check if text contains a JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                move_val = data.get("uci_move") or data.get("move") or data.get("move_uci")
                if move_val and isinstance(move_val, str):
                    res = _parse_move_candidate(move_val, board)
                    if res and res.uci:
                        res.confidence = 1.0
                        return res
    except json.JSONDecodeError:
        pass

    # 1. PRIORITY 1: Explicit <move>...</move> tag extraction
    move_match = re.search(r'<move>\s*(.*?)\s*</move>', text, re.IGNORECASE | re.DOTALL)
    if move_match:
        tag_content = move_match.group(1)
        res = _parse_move_candidate(tag_content, board)
        if res and res.uci:
            res.confidence = 1.0
            return res

    # 2. PRIORITY 2: Strip thinking blocks.
    # When the move tag was present but unparseable, also strip the <move> wrapper
    # so the inner text continues into the later fallback scanners.
    clean_text = _strip_thinking(text)
    if move_match:
        clean_text = re.sub(r'</?move>', '', clean_text, flags=re.IGNORECASE)

    # Strip untagged thinking headers like "Here's a thinking process:"
    thinking_headers = [
        r"(?:here's|here is)\s+a\s+thinking\s+process:?",
        r"thinking\s+process:?",
        r"analysis:?",
        r"reasoning:?",
    ]
    for header in thinking_headers:
        if re.search(header, clean_text, re.IGNORECASE):
            parts = re.split(header, clean_text, flags=re.IGNORECASE)
            if len(parts) > 1:
                clean_text = parts[-1]

    # Strip echoed legal moves list from prompt to avoid matching prompt echoes
    clean_text = re.sub(r'Legal moves(?:\s*\([^)]*\))?:[^\n]*', '', clean_text, flags=re.IGNORECASE)

    # 3. PRIORITY 3: Explicit Move Headers (e.g. "**Move:** e4", "Selected move: Nf3", "Play: Nf3")
    move_header_patterns = [
        r'(?:final\s+)?move\s*:\s*([^\n]+)',
        r'(?:play|choose|select)\s+([^\n]+)',
    ]
    uci_pattern_full = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
    first_uci_in_text = re.search(uci_pattern_full, clean_text.lower())
    first_uci_str = first_uci_in_text.group(1) if first_uci_in_text else None
    first_uci_legal = False
    if first_uci_str and board is not None:
        try:
            first_uci_legal = chess.Move.from_uci(first_uci_str) in board.legal_moves
        except ValueError:
            first_uci_legal = False
    for pattern in move_header_patterns:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            res = _parse_move_candidate(candidate, board)
            if res and res.uci:
                if first_uci_str and res.uci != first_uci_str and (first_uci_legal or board is None):
                    continue
                res.confidence = 0.9
                return res

    # 4. PRIORITY 4: Check the LAST non-empty lines (where models summarize their decision)
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    if lines:
        for line in reversed(lines[-3:]):
            res = _parse_move_candidate(line, board)
            if res and res.uci:
                return res

    # 5. PRIORITY 5: Full text SAN & UCI scan
    if board:
        result = _parse_san(clean_text, board)
        if result and result.uci:
            return result

    result = _parse_uci(clean_text, board)
    if result and result.uci:
        return result

    if board:
        result = _parse_natural_language(clean_text, board)
        if result and result.uci:
            return result

    # Fallback: forward search for legal UCI
    uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
    matches = re.findall(uci_pattern, clean_text.lower())
    for match in matches:
        try:
            move = chess.Move.from_uci(match)
            if board:
                if move in board.legal_moves:
                    return MoveParseResult(
                        uci=match,
                        san=board.san(move),
                        confidence=0.6,
                        ambiguous=False,
                    )
            else:
                return MoveParseResult(
                    uci=match,
                    san=None,
                    confidence=0.5,
                    ambiguous=False,
                )
        except ValueError:
            continue

    return MoveParseResult(uci=None, san=None, confidence=0.0, ambiguous=False)


def validate_move(move_str: str, board: chess.Board) -> str:
    """
    Validate a UCI move string against the board.

    Args:
        move_str: UCI move string
        board: Current chess board

    Returns:
        Validated UCI move string

    Raises:
        ValueError: If move is invalid or illegal
    """
    move_str = move_str.strip().lower()

    # Remove common response artifacts from START only
    prefixes = ["move:", "i choose", "my move is", "play", "'", '"', "`"]
    for prefix in prefixes:
        if move_str.startswith(prefix):
            move_str = move_str[len(prefix):].strip()

    # Remove trailing artifacts (quotes, backticks, punctuation)
    suffixes = ["'", '"', "`", ".", ",", ":", ";"]
    for suffix in suffixes:
        if move_str.endswith(suffix):
            move_str = move_str[:-len(suffix)].strip()

    # Basic UCI format validation
    if not (4 <= len(move_str) <= 5):
        raise ValueError(f"Invalid move format: {move_str}")

    # Create chess.Move object
    try:
        move = chess.Move.from_uci(move_str)
    except ValueError as err:
        raise ValueError(f"Invalid UCI format: {move_str}") from err

    # Check if move is legal in current position
    if move not in board.legal_moves:
        legal_moves = [m.uci() for m in board.legal_moves]
        raise ValueError(f"Illegal move {move_str}. Legal moves are: {', '.join(legal_moves)}")

    return move_str


def extract_move(text: str, legal_moves: list[chess.Move] | None = None) -> str | None:
    """
    Extract UCI move from LLM output. Returns None if not found.
    Kept for backward compatibility.

    Args:
        text: Raw LLM response text
        legal_moves: Optional list of legal moves to validate against

    Returns:
        UCI move string (e.g., "e2e4") or None if not found/ambiguous
    """
    board = None
    if legal_moves:
        # Construct board with legal moves if possible or check legal_moves set
        legal_uci = {m.uci() for m in legal_moves}
        move_match = re.search(r'<move>\s*(.*?)\s*</move>', text, re.IGNORECASE | re.DOTALL)
        if move_match:
            candidate = move_match.group(1).lower().strip()
            clean_cand = re.sub(r'[^a-h1-8qrbn]', '', candidate)
            if clean_cand in legal_uci:
                return clean_cand

    parsed = parse_move(text, board)
    if parsed and parsed.uci:
        if legal_moves:
            legal_uci = {m.uci() for m in legal_moves}
            if parsed.uci in legal_uci:
                return parsed.uci
        else:
            return parsed.uci

    return None
