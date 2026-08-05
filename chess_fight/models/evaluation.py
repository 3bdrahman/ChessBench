"""Chess position evaluation utilities."""


import chess


class PositionEvaluator:
    """Utility class for chess position evaluation."""

    def __init__(self):
        self.piece_values: dict[chess.PieceType, int] = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # New scoring weights for move evaluation
        self.eval_weights: dict[str, float] = {
            'capture_value': 1.0,
            'center_control': 0.8,
            'development': 0.7,
            'king_safety': 0.9,
            'pawn_structure': 0.6,
            'piece_activity': 0.75,
            'position_progress': 1.0
        }

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

    def get_material_count(self, board: chess.Board) -> str:
        """Calculate material count for both sides."""
        piece_values: dict[chess.PieceType, int] = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
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

        return f"White: {white_material} points, Black: {black_material} points"

    def analyze_material_tension(self, board: chess.Board) -> str:
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

        return f"Tension Score: {tension_score/100:.1f}, Possible Exchanges: {', '.join(exchanges[:3])}"

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
            progress_score += 100

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
            progress_score += 50

        return progress_score

    def analyze_position_dynamism(self, board: chess.Board) -> str:
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

        return f"Dynamism Score: {dynamism_score}, Factors: {', '.join(dynamic_factors)}"

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
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        if not board.is_capture(move):
            return 0

        captured_piece = board.piece_at(move.to_square)
        capturing_piece = board.piece_at(move.from_square)

        if not captured_piece or not capturing_piece:
            return 0

        return piece_values[captured_piece.piece_type] - piece_values[capturing_piece.piece_type]

    def calculate_development_score(self, board: chess.Board) -> str:
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

        return f"Development Score: {score}, Developed: {', '.join(developed_pieces)}"

    def analyze_captures(self, board: chess.Board) -> str:
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
                        f"{chess.piece_name(capturing.piece_type).capitalize()} takes "
                        f"{chess.piece_name(captured.piece_type)} on {chess.square_name(move.to_square)} "
                        f"(value: {value_diff/100:+.1f}) [{move.uci()}]"
                    ))

        if not captures:
            return "No captures available"

        captures.sort(key=lambda x: x[0], reverse=True)
        return "\n".join(capture[1] for capture in captures)

    def analyze_threats(self, board: chess.Board) -> str:
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

        return "\n".join(threats) if threats else "No pieces currently threatened"

    def evaluate_capture(self, board: chess.Board, move: chess.Move) -> float:
        """Enhanced capture evaluation with positional considerations."""
        if not board.is_capture(move):
            return 0.0

        captured_piece = board.piece_at(move.to_square)
        capturing_piece = board.piece_at(move.from_square)

        if not captured_piece or not capturing_piece:
            return 0.0

        value_diff: float = float(self.piece_values[captured_piece.piece_type] - self.piece_values[capturing_piece.piece_type])

        board.push(move)

        if board.is_attacked_by(not board.turn, move.to_square):
            defenders = len(list(board.attackers(board.turn, move.to_square)))
            attackers = len(list(board.attackers(not board.turn, move.to_square)))
            if attackers > defenders:
                value_diff -= self.piece_values[capturing_piece.piece_type] * 0.8

        if chess.square_file(move.to_square) in [3, 4] and chess.square_rank(move.to_square) in [3, 4]:
            value_diff += 50

        board.pop()
        return float(value_diff)

    def categorize_moves(self, board: chess.Board) -> dict[str, str]:
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

            board.pop()

        forcing_moves.sort(key=lambda x: x[0], reverse=True)
        developing_moves.sort(key=lambda x: x[0], reverse=True)
        positional_moves.sort(key=lambda x: x[0], reverse=True)

        return {
            'forcing_moves': "\n".join(move[1] for move in forcing_moves) if forcing_moves else "None available",
            'developing_moves': "\n".join(move[1] for move in developing_moves) if developing_moves else "None available",
            'positional_moves': "\n".join(move[1] for move in positional_moves) if positional_moves else "None available"
        }

    def analyze_defense(self, board: chess.Board) -> str:
        """Analyze defensive needs and immediate threats."""
        analysis = []

        opponent_board = board.copy()
        opponent_board.turn = not board.turn
        for move in opponent_board.legal_moves:
            opponent_board.push(move)
            if opponent_board.is_checkmate():
                analysis.append(f"CRITICAL: Mate threat via {move.uci()}")
            opponent_board.pop()

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == board.turn:
                attackers = board.attackers(not board.turn, square)
                defenders = board.attackers(board.turn, square)
                if attackers and not defenders:
                    analysis.append(
                        f"URGENT: Undefended {chess.piece_name(piece.piece_type)} on "
                        f"{chess.square_name(square)} under attack"
                    )

        return "\n".join(analysis) if analysis else "No immediate defensive concerns"

    def analyze_vulnerabilities(self, board: chess.Board) -> str:
        """Analyze opponent's weaknesses."""
        vulnerabilities = []

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                attackers = board.attackers(board.turn, square)
                defenders = board.attackers(not board.turn, square)
                if not defenders and attackers:
                    vulnerabilities.append(
                        f"Undefended {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"
                    )

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn and board.is_pinned(not board.turn, square):
                vulnerabilities.append(
                    f"Pinned {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"
                )

        return "\n".join(vulnerabilities) if vulnerabilities else "No major vulnerabilities found"

    def analyze_king_safety(self, board: chess.Board) -> str:
        """Analyze king safety for both sides."""
        def king_zone_attacks(king_color):
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

        return (
            f"Your king safety: {own_king_attacks} attacks in king zone\n"
            f"Opponent king safety: {opponent_king_attacks} attacks in king zone"
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

    def analyze_pawn_structure(self, board: chess.Board) -> str:
        """Analyze pawn structure strengths and weaknesses."""
        analysis = []

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
                    analysis.append(f"Isolated pawn on file {chess.FILE_NAMES[file]}")

        return "\n".join(analysis) if analysis else "Solid pawn structure"

    def analyze_undefended_pieces(self, board: chess.Board) -> str:
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
        return "\n".join(undefended) if undefended else "No undefended pieces"

    def analyze_exposed_pieces(self, board: chess.Board) -> str:
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
        return "\n".join(exposed) if exposed else "No exposed pieces"

    def analyze_material_balance(self, board: chess.Board) -> str:
        """Analyze material balance with piece-specific details."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
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

        return f"Material balance: {side_to_move_advantage:+d} ({'+' if side_to_move_advantage > 0 else ''}{side_to_move_advantage} pawns)"

    def analyze_center_control(self, board: chess.Board) -> str:
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
        return (
            f"Center control: {control[side_to_move]} squares attacked by you vs "
            f"{control[opponent]} by opponent"
        )

    def analyze_development_status(self, board: chess.Board) -> str:
        """Analyze piece development status."""
        def count_developed_pieces(color):
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

        return f"Developed pieces: {own_developed} vs opponent's {opponent_developed}"
