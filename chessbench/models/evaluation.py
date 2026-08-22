"""Chess position evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from chessbench import constants


@dataclass
class PositionEval:
    """Typed position evaluation result."""

    cp_score: int | None = None
    mate_in: int | None = None
    best_move_uci: str | None = None
    pv: list[str] | None = None
    components: dict[str, float | str | None] = field(default_factory=dict)

    def __str__(self) -> str:
        """Human-readable string for backward compatibility."""
        parts = []
        if self.cp_score is not None:
            parts.append(f"cp: {self.cp_score}")
        if self.mate_in is not None:
            parts.append(f"mate in {self.mate_in}")
        if self.best_move_uci:
            parts.append(f"best: {self.best_move_uci}")
        if self.pv:
            parts.append(f"pv: {' '.join(self.pv)}")
        if self.components:
            comp_parts = []
            for k, v in self.components.items():
                if isinstance(v, float):
                    comp_parts.append(f"{k}={v:.1f}")
                else:
                    comp_parts.append(f"{k}={v}")
            comp_str = ", ".join(comp_parts)
            parts.append(f"components: {{{comp_str}}}")
        return " | ".join(parts) if parts else "No evaluation"

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable dictionary."""
        return {
            "cp_score": self.cp_score,
            "mate_in": self.mate_in,
            "best_move_uci": self.best_move_uci,
            "pv": self.pv,
            "components": self.components,
        }


class PositionEvaluator:
    """Utility class for chess position evaluation."""

    def __init__(self) -> None:
        self.piece_values: dict[chess.PieceType, int] = {
            chess.PAWN: constants.PIECE_VALUES_CP["PAWN"],
            chess.KNIGHT: constants.PIECE_VALUES_CP["KNIGHT"],
            chess.BISHOP: constants.PIECE_VALUES_CP["BISHOP"],
            chess.ROOK: constants.PIECE_VALUES_CP["ROOK"],
            chess.QUEEN: constants.PIECE_VALUES_CP["QUEEN"],
            chess.KING: constants.PIECE_VALUES_CP["KING"],
        }

        # New scoring weights for move evaluation
        self.eval_weights: dict[str, float] = constants.EVAL_WEIGHTS

    def get_piece_locations(self, board: chess.Board) -> tuple[list[str], list[str]]:
        """Get structured information about piece locations."""
        white_pieces: list[str] = []
        black_pieces: list[str] = []

        piece_symbols: dict[chess.PieceType, str] = {
            chess.PAWN: "Pawn",
            chess.KNIGHT: "Knight",
            chess.BISHOP: "Bishop",
            chess.ROOK: "Rook",
            chess.QUEEN: "Queen",
            chess.KING: "King"
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                square_name = chess.square_name(square)
                piece_name = piece_symbols[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_pieces.append(f"{piece_name} at {square_name}")
                else:
                    black_pieces.append(f"{piece_name} at {square_name}")

        return white_pieces, black_pieces

    def get_material_count(self, board: chess.Board) -> PositionEval:
        """Calculate material count for both sides."""
        piece_values: dict[chess.PieceType, int] = {
            chess.PAWN: constants.PIECE_VALUES_MATERIAL["PAWN"],
            chess.KNIGHT: constants.PIECE_VALUES_MATERIAL["KNIGHT"],
            chess.BISHOP: constants.PIECE_VALUES_MATERIAL["BISHOP"],
            chess.ROOK: constants.PIECE_VALUES_MATERIAL["ROOK"],
            chess.QUEEN: constants.PIECE_VALUES_MATERIAL["QUEEN"],
            chess.KING: constants.PIECE_VALUES_MATERIAL["KING"],
        }
        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        balance = white_material - black_material
        side_to_move_advantage = balance if board.turn == chess.WHITE else -balance

        return PositionEval(
            cp_score=side_to_move_advantage * constants.MATERIAL_BALANCE_MULTIPLIER,
            components={
                "material": float(side_to_move_advantage * constants.MATERIAL_BALANCE_MULTIPLIER),
                "white_material": float(white_material),
                "black_material": float(black_material),
            }
        )

    def analyze_material_tension(self, board: chess.Board) -> PositionEval:
        """Analyze pieces under attack and potential captures."""
        tension_score = 0
        exchanges: list[str] = []

        for move in board.legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                capturing_piece = board.piece_at(move.from_square)
                if captured_piece and capturing_piece:
                    value_diff = self.analyze_capture_value(board, move)
                    tension_score += abs(value_diff)
                    exchanges.append(f"{chess.piece_name(capturing_piece.piece_type)} x {chess.piece_name(captured_piece.piece_type)}")

        return PositionEval(
            cp_score=tension_score,
            components={
                "tension_score": float(tension_score),
                "exchanges_count": float(len(exchanges)),
            }
        )

    def annotate_moves(self, board: chess.Board) -> str:
        """Create annotated list of legal moves with piece information."""
        annotated_moves: list[str] = []
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece is None:
                continue
            piece_type = chess.piece_name(piece.piece_type).capitalize()
            from_square = chess.square_name(move.from_square)
            to_square = chess.square_name(move.to_square)
            capture = " captures" if board.is_capture(move) else " to"
            target_piece = board.piece_at(move.to_square)
            target_info = f" {chess.piece_name(target_piece.piece_type)}" if target_piece else ""

            promotion = f" promoting to {chess.piece_name(move.promotion)}" if move.promotion else ""
            check = " (check)" if board.gives_check(move) else ""

            annotation = f"{piece_type} {from_square}{capture}{target_info} {to_square}{promotion}{check} [{move.uci()}]"
            annotated_moves.append(annotation)

        return "\n".join(annotated_moves)

    def analyze_position_progress(self, board: chess.Board, move: chess.Move) -> float:
        """Evaluate if a move makes meaningful progress."""
        progress_score = 0.0

        if chess.square_rank(move.from_square) in [0, 1, 6, 7] and chess.square_rank(move.to_square) not in [0, 1, 6, 7]:
            progress_score += constants.PROGRESS_BACK_RANK_TO_CENTER_BONUS

        center_distance_before = min(
            chess.square_distance(move.from_square, chess.E4),
            chess.square_distance(move.from_square, chess.D4),
            chess.square_distance(move.from_square, chess.E5),
            chess.square_distance(move.from_square, chess.D5)
        )
        center_distance_after = min(
            chess.square_distance(move.to_square, chess.E4),
            chess.square_distance(move.to_square, chess.D4),
            chess.square_distance(move.to_square, chess.E5),
            chess.square_distance(move.to_square, chess.D5)
        )
        if center_distance_after < center_distance_before:
            progress_score += constants.PROGRESS_CENTER_BONUS

        return progress_score

    def analyze_position_dynamism(self, board: chess.Board) -> PositionEval:
        """Analyze how dynamic/static the position is."""
        dynamic_factors = []
        dynamism_score = 0

        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = sum(1 for sq in center_squares if board.is_attacked_by(board.turn, sq))
        dynamism_score += center_control * 2

        mobility = len(list(board.legal_moves))
        dynamism_score += mobility // 4

        pawn_count = 0
        for move in list(board.legal_moves):
            piece = board.piece_at(move.from_square)
            if piece is not None and piece.piece_type == chess.PAWN:
                pawn_count += 1
        dynamism_score += pawn_count

        if board.is_check():
            dynamic_factors.append("Check")
            dynamism_score += 5
        if any(board.is_capture(move) for move in board.legal_moves):
            dynamic_factors.append("Captures Available")
            dynamism_score += 3

        return PositionEval(
            cp_score=dynamism_score,
            components={
                "dynamism_score": float(dynamism_score),
                "center_control": float(center_control),
                "mobility": float(mobility),
                "pawn_moves": float(pawn_count),
                "in_check": 1.0 if board.is_check() else 0.0,
                "captures_available": 1.0 if any(board.is_capture(m) for m in board.legal_moves) else 0.0,
            }
        )

    def get_castling_rights(self, board: chess.Board) -> str:
        """Get readable castling rights."""
        rights = []
        if board.has_kingside_castling_rights(chess.WHITE):
            rights.append("White O-O")
        if board.has_queenside_castling_rights(chess.WHITE):
            rights.append("White O-O-O")
        if board.has_kingside_castling_rights(chess.BLACK):
            rights.append("Black O-O")
        if board.has_queenside_castling_rights(chess.BLACK):
            rights.append("Black O-O-O")
        return ", ".join(rights) if rights else "None"

    def analyze_capture_value(self, board: chess.Board, move: chess.Move) -> int:
        """Calculate the value difference of a capture move."""
        if not board.is_capture(move):
            return 0

        captured_piece = board.piece_at(move.to_square)
        capturing_piece = board.piece_at(move.from_square)

        if not captured_piece or not capturing_piece:
            return 0

        return (
            self.piece_values[captured_piece.piece_type]
            - self.piece_values[capturing_piece.piece_type]
        )

    def calculate_development_score(self, board: chess.Board) -> PositionEval:
        """Calculate development score based on piece positioning."""
        score = 0
        developed_pieces = []

        piece_development = {
            chess.KNIGHT: (2, ["b1", "g1"] if board.turn == chess.WHITE else ["b8", "g8"]),
            chess.BISHOP: (2, ["c1", "f1"] if board.turn == chess.WHITE else ["c8", "f8"]),
            chess.QUEEN: (1, ["d1"] if board.turn == chess.WHITE else ["d8"])
        }

        for piece_type, (value, initial_squares) in piece_development.items():
            for square_name in initial_squares:
                square = chess.parse_square(square_name)
                piece = board.piece_at(square)
                if not piece or piece.piece_type != piece_type:
                    score += value
                    developed_pieces.append(chess.piece_name(piece_type))

        if not board.has_kingside_castling_rights(board.turn):
            score += 3
            developed_pieces.append("Castled")

        center_files = ['d', 'e']
        back_rank = '2' if board.turn == chess.WHITE else '7'
        for file in center_files:
            square_name = file + back_rank
            square = chess.parse_square(square_name)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                score -= 1

        return PositionEval(
            cp_score=score * 10,
            components={
                "development_score": float(score * 10),
                "developed_pieces_count": float(len(developed_pieces)),
            }
        )

    def analyze_captures(self, board: chess.Board) -> PositionEval:
        """Analyze all possible captures and sort by value."""
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move):
                value_diff = self.analyze_capture_value(board, move)
                captured = board.piece_at(move.to_square)
                capturing = board.piece_at(move.from_square)
                if captured and capturing:
                    captures.append((
                        value_diff,
                        move.uci(),
                        f"{chess.piece_name(capturing.piece_type).capitalize()} takes "
                        f"{chess.piece_name(captured.piece_type)} on {chess.square_name(move.to_square)} "
                        f"(value: {value_diff/100:+.1f}) [{move.uci()}]"
                    ))

        if not captures:
            return PositionEval(
                cp_score=0,
                components={"captures_available": 0.0, "best_capture_uci": None}
            )

        captures.sort(key=lambda x: x[0], reverse=True)
        best_capture = captures[0]

        return PositionEval(
            cp_score=best_capture[0],
            components={
                "captures_available": float(len(captures)),
                "best_capture_value": float(best_capture[0]),
                "best_capture_uci": best_capture[1],
            }
        )

    def analyze_threats(self, board: chess.Board) -> PositionEval:
        """Analyze which pieces are under attack."""
        threats = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn and board.is_attacked_by(not board.turn, square):
                attackers = []
                for attacker_square in board.attackers(not board.turn, square):
                    attacker = board.piece_at(attacker_square)
                    if attacker:
                        attackers.append(chess.piece_name(attacker.piece_type))
                threats.append(
                    f"{chess.piece_name(piece.piece_type).capitalize()} on "
                    f"{chess.square_name(square)} threatened by {', '.join(attackers)}"
                )

        return PositionEval(
            cp_score=-len(threats) * 50,  # Negative = bad for side to move
            components={
                "threats_count": float(len(threats)),
                "threatened_pieces": float(len(threats)),
            }
        )

    def evaluate_capture(self, board: chess.Board, move: chess.Move) -> float:
        """Enhanced capture evaluation with positional considerations."""
        if not board.is_capture(move):
            return 0.0

        captured_piece = board.piece_at(move.to_square)
        capturing_piece = board.piece_at(move.from_square)

        if not captured_piece or not capturing_piece:
            return 0.0

        value_diff: float = float(
            self.piece_values[captured_piece.piece_type]
            - self.piece_values[capturing_piece.piece_type]
        )

        board.push(move)
        try:
            if board.is_attacked_by(not board.turn, move.to_square):
                defenders = len(list(board.attackers(board.turn, move.to_square)))
                attackers = len(list(board.attackers(not board.turn, move.to_square)))
                if attackers > defenders:
                    value_diff -= (
                        self.piece_values[capturing_piece.piece_type] * 0.8
                    )

            if chess.square_file(move.to_square) in [3, 4] and chess.square_rank(
                move.to_square
            ) in [3, 4]:
                value_diff += 50
        finally:
            board.pop()
        return float(value_diff)

    def categorize_moves(self, board: chess.Board) -> dict[str, PositionEval]:
        """Enhanced move categorization with stronger tactical awareness."""
        forcing_moves = []
        developing_moves = []
        positional_moves = []

        for move in board.legal_moves:
            move_str = move.uci()
            piece = board.piece_at(move.from_square)
            if not piece:
                continue

            capture_value = self.evaluate_capture(board, move)
            progress_score = self.analyze_position_progress(board, move)
            total_score = capture_value + progress_score

            board.push(move)
            try:
                if capture_value > 0 or board.is_check():
                    forcing_moves.append((
                        total_score,
                        f"{chess.piece_name(piece.piece_type)} "
                        f"{'captures' if board.is_capture(move) else 'checks'} "
                        f"(score: {total_score:+.1f}) [{move_str}]"
                    ))
                elif progress_score > 0:
                    developing_moves.append((
                        progress_score,
                        f"{chess.piece_name(piece.piece_type)} development "
                        f"(score: {progress_score:+.1f}) [{move_str}]"
                    ))
                else:
                    positional_moves.append((
                        total_score,
                        f"{chess.piece_name(piece.piece_type)} repositioning [{move_str}]"
                    ))
            finally:
                board.pop()

        forcing_moves.sort(key=lambda x: x[0], reverse=True)
        developing_moves.sort(key=lambda x: x[0], reverse=True)
        positional_moves.sort(key=lambda x: x[0], reverse=True)

        return {
            'forcing_moves': PositionEval(
                components={
                    "count": float(len(forcing_moves)),
                    "best_score": float(forcing_moves[0][0]) if forcing_moves else 0.0,
                },
                pv=[x[1] for x in forcing_moves]
            ),
            'developing_moves': PositionEval(
                components={
                    "count": float(len(developing_moves)),
                    "best_score": float(developing_moves[0][0]) if developing_moves else 0.0,
                },
                pv=[x[1] for x in developing_moves]
            ),
            'positional_moves': PositionEval(
                components={
                    "count": float(len(positional_moves)),
                    "best_score": float(positional_moves[0][0]) if positional_moves else 0.0,
                },
                pv=[x[1] for x in positional_moves]
            )
        }

    def analyze_defense(self, board: chess.Board) -> PositionEval:
        """Analyze defensive needs and immediate threats."""
        analysis = []
        mate_threats = 0
        undefended_under_attack = 0

        opponent_board = board.copy()
        opponent_board.turn = not board.turn
        for move in opponent_board.legal_moves:
            opponent_board.push(move)
            try:
                if opponent_board.is_checkmate():
                    mate_threats += 1
                    analysis.append(f"CRITICAL: Mate threat via {move.uci()}")
            finally:
                opponent_board.pop()

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = board.attackers(not board.turn, square)
                defenders = board.attackers(board.turn, square)
                if attackers and not defenders:
                    undefended_under_attack += 1
                    analysis.append(
                        f"URGENT: Undefended {chess.piece_name(piece.piece_type)} on "
                        f"{chess.square_name(square)} under attack"
                    )

        return PositionEval(
            cp_score=-(mate_threats * constants.MATE_THREAT_SCORE + undefended_under_attack * constants.UNDEFENDED_UNDER_ATTACK_SCORE),
            components={
                "mate_threats": float(mate_threats),
                "undefended_under_attack": float(undefended_under_attack),
                "defense_score": float(-(mate_threats * constants.MATE_THREAT_SCORE + undefended_under_attack * constants.UNDEFENDED_UNDER_ATTACK_SCORE)),
            }
        )

    def analyze_vulnerabilities(self, board: chess.Board) -> PositionEval:
        """Analyze opponent's weaknesses."""
        vulnerabilities = []
        undefended_count = 0
        pinned_count = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                attackers = board.attackers(board.turn, square)
                defenders = board.attackers(not board.turn, square)
                if not defenders and attackers:
                    undefended_count += 1
                    vulnerabilities.append(
                        f"Undefended {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"
                    )

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn and board.is_pinned(not board.turn, square):
                pinned_count += 1
                vulnerabilities.append(
                    f"Pinned {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"
                )

        return PositionEval(
            cp_score=(undefended_count * constants.VULNERABILITY_UNDEFENDED_SCORE + pinned_count * constants.VULNERABILITY_PINNED_SCORE),
            components={
                "undefended_opponent": float(undefended_count),
                "pinned_opponent": float(pinned_count),
                "vulnerability_score": float(undefended_count * constants.VULNERABILITY_UNDEFENDED_SCORE + pinned_count * constants.VULNERABILITY_PINNED_SCORE),
            }
        )

    def analyze_king_safety(self, board: chess.Board) -> PositionEval:
        """Analyze king safety for both sides."""
        def king_zone_attacks(king_color: chess.Color) -> int:
            king_square = board.king(king_color)
            if king_square is None:
                return 0

            attack_count = 0
            for square in chess.SQUARES:
                if chess.square_distance(king_square, square) <= 2 and board.is_attacked_by(not king_color, square):
                    attack_count += 1
            return attack_count

        own_king_attacks = king_zone_attacks(board.turn)
        opponent_king_attacks = king_zone_attacks(not board.turn)

        return PositionEval(
            cp_score=(opponent_king_attacks - own_king_attacks) * constants.KING_SAFETY_MULTIPLIER,
            components={
                "own_king_attacks": float(own_king_attacks),
                "opponent_king_attacks": float(opponent_king_attacks),
                "king_safety_delta": float((opponent_king_attacks - own_king_attacks) * constants.KING_SAFETY_MULTIPLIER),
            }
        )

    def is_pinned(self, board: chess.Board, square: int) -> bool:
        """Check if a piece is pinned to its king."""
        piece = board.piece_at(square)
        if not piece:
            return False

        color = piece.color
        king_square = board.king(color)
        if king_square is None:
            return False

        return bool(board.is_pinned(color, square))

    def analyze_pawn_structure(self, board: chess.Board) -> PositionEval:
        """Analyze pawn structure strengths and weaknesses."""
        analysis = []
        isolated_pawns = 0

        for file in range(8):
            pawns = []
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == board.turn:
                    pawns.append(rank)

            if pawns:
                has_neighbors = False
                for adjacent_file in [file - 1, file + 1]:
                    if 0 <= adjacent_file < 8:
                        for rank in range(8):
                            square = chess.square(adjacent_file, rank)
                            piece = board.piece_at(square)
                            if piece and piece.piece_type == chess.PAWN and piece.color == board.turn:
                                has_neighbors = True
                                break

                if not has_neighbors:
                    isolated_pawns += 1
                    analysis.append(f"Isolated pawn on file {chess.FILE_NAMES[file]}")

        return PositionEval(
            cp_score=-isolated_pawns * constants.ISOLATED_PAWN_PENALTY,
            components={
                "isolated_pawns": float(isolated_pawns),
                "pawn_structure_score": float(-isolated_pawns * constants.ISOLATED_PAWN_PENALTY),
            }
        )

    def analyze_undefended_pieces(self, board: chess.Board) -> PositionEval:
        """Analyze undefended pieces for the current side."""
        undefended = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = list(board.attackers(not board.turn, square))
                defenders = list(board.attackers(board.turn, square))
                if not defenders and attackers:
                    undefended.append(
                        f"{chess.piece_name(piece.piece_type)} on {chess.square_name(square)} "
                        f"attacked by {len(attackers)} piece(s)"
                    )

        return PositionEval(
            cp_score=-len(undefended) * constants.UNDEFENDED_PIECE_PENALTY,
            components={
                "undefended_count": float(len(undefended)),
                "undefended_score": float(-len(undefended) * constants.UNDEFENDED_PIECE_PENALTY),
            }
        )

    def analyze_exposed_pieces(self, board: chess.Board) -> PositionEval:
        """Analyze exposed pieces that could become vulnerable."""
        exposed = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = list(board.attackers(not board.turn, square))
                defenders = list(board.attackers(board.turn, square))
                if len(defenders) < len(attackers):
                    exposed.append(
                        f"{chess.piece_name(piece.piece_type)} on {chess.square_name(square)} "
                        f"({len(defenders)} defenders vs {len(attackers)} attackers)"
                    )

        return PositionEval(
            cp_score=-len(exposed) * constants.EXPOSED_PIECE_PENALTY,
            components={
                "exposed_count": float(len(exposed)),
                "exposed_score": float(-len(exposed) * constants.EXPOSED_PIECE_PENALTY),
            }
        )

    def analyze_material_balance(self, board: chess.Board) -> PositionEval:
        """Analyze material balance with piece-specific details."""
        piece_values = {
            chess.PAWN: constants.PIECE_VALUES_MATERIAL["PAWN"],
            chess.KNIGHT: constants.PIECE_VALUES_MATERIAL["KNIGHT"],
            chess.BISHOP: constants.PIECE_VALUES_MATERIAL["BISHOP"],
            chess.ROOK: constants.PIECE_VALUES_MATERIAL["ROOK"],
            chess.QUEEN: constants.PIECE_VALUES_MATERIAL["QUEEN"],
        }

        white_pieces = dict.fromkeys(piece_values, 0)
        black_pieces = dict.fromkeys(piece_values, 0)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                if piece.color == chess.WHITE:
                    white_pieces[piece.piece_type] += 1
                else:
                    black_pieces[piece.piece_type] += 1

        white_score = sum(count * piece_values[piece] for piece, count in white_pieces.items())
        black_score = sum(count * piece_values[piece] for piece, count in black_pieces.items())

        balance = white_score - black_score
        side_to_move_advantage = balance if board.turn == chess.WHITE else -balance

        return PositionEval(
            cp_score=side_to_move_advantage * constants.MATERIAL_BALANCE_MULTIPLIER,
            components={
                "material_balance_pawns": float(side_to_move_advantage),
                "white_material": float(white_score),
                "black_material": float(black_score),
            }
        )

    def analyze_center_control(self, board: chess.Board) -> PositionEval:
        """Analyze control of central squares."""
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        control = {chess.WHITE: 0, chess.BLACK: 0}

        for square in center_squares:
            white_attackers = len(list(board.attackers(chess.WHITE, square)))
            black_attackers = len(list(board.attackers(chess.BLACK, square)))
            control[chess.WHITE] += white_attackers
            control[chess.BLACK] += black_attackers

        side_to_move = board.turn
        opponent = not side_to_move

        return PositionEval(
            cp_score=(control[side_to_move] - control[opponent]) * constants.CENTER_CONTROL_MULTIPLIER,
            components={
                "center_control_self": float(control[side_to_move]),
                "center_control_opp": float(control[opponent]),
                "center_control_delta": float((control[side_to_move] - control[opponent]) * constants.CENTER_CONTROL_MULTIPLIER),
            }
        )

    def analyze_development_status(self, board: chess.Board) -> PositionEval:
        """Analyze piece development status."""
        def count_developed_pieces(color: chess.Color) -> int:
            developed = 0
            back_rank = 0 if color == chess.WHITE else 7

            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if (piece and piece.color == color and piece.piece_type == piece_type and
                        chess.square_rank(square) != back_rank):
                        developed += 1

            if board.has_castling_rights(color):
                developed += 1

            return developed

        own_developed = count_developed_pieces(board.turn)
        opponent_developed = count_developed_pieces(not board.turn)

        return PositionEval(
            cp_score=(own_developed - opponent_developed) * constants.DEVELOPMENT_BONUS,
            components={
                "own_developed": float(own_developed),
                "opponent_developed": float(opponent_developed),
                "development_delta": float((own_developed - opponent_developed) * constants.DEVELOPMENT_BONUS),
            }
        )
