"""Robust UCI move extraction from LLM outputs."""

import re

import chess


def extract_move(text: str, legal_moves: list[chess.Move] | None = None) -> str | None:
    """
    Extract UCI move from LLM output. Returns None if not found.
    
    Args:
        text: Raw LLM response text
        legal_moves: Optional list of legal moves to validate against
        
    Returns:
        UCI move string (e.g., "e2e4") or None if not found/ambiguous
    """
    # 1. Strip <thinking>...</thinking> blocks
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Find UCI pattern: [a-h][1-8][a-h][1-8][qrbn]?
    uci_pattern = r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b'
    matches = re.findall(uci_pattern, text.lower())

    if matches:
        # If we have legal moves, filter to only valid ones
        if legal_moves:
            legal_uci = {m.uci() for m in legal_moves}
            valid_matches = [m for m in matches if m in legal_uci]
            if valid_matches:
                # Return first valid match
                return valid_matches[0]
            return None
        # Return first match if no validation needed
        return matches[0]

    # 3. Fallback: "I will play e2e4" → regex capture
    fallback_patterns = [
        r'(?:play|move|choose|will play)\s+([a-h][1-8][a-h][1-8][qrbn]?)',
        r'best move.{0,10}([a-h][1-8][a-h][1-8][qrbn]?)',
        r'([a-h][1-8][a-h][1-8][qrbn]?)',
    ]

    for pattern in fallback_patterns:
        match = re.search(pattern, text.lower())
        if match:
            candidate = match.group(1)
            if legal_moves:
                legal_uci = {m.uci() for m in legal_moves}
                if candidate in legal_uci:
                    return candidate
            else:
                return candidate

    return None


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
    except ValueError:
        raise ValueError(f"Invalid UCI format: {move_str}")

    # Check if move is legal in current position
    if move not in board.legal_moves:
        legal_moves = [m.uci() for m in board.legal_moves]
        raise ValueError(f"Illegal move {move_str}. Legal moves are: {', '.join(legal_moves)}")

    return move_str
