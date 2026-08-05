"""Base ChessAI class and legacy implementations."""

from abc import ABC, abstractmethod
from enum import Enum

import chess
import ollama
from anthropic import Anthropic
from openai import OpenAI

from chess_fight.common.common_types import CompletionResult
from chess_fight.config import ANTHROPIC_API_KEY
from chess_fight.models.evaluation import PositionEvaluator


class ModelType(Enum):
    CHATGPT_4O = "gpt-4o"
    CLAUDE_SONNET = "Claude Sonnet 3.5"
    LLAMA_3_2 = "Llama3.2"


class ChessAI(ABC):
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__
        self.move_history: list[str] = []
        self.position_history: set[str] = set()
        self.stagnation_threshold = 3

        self.last_completion_result: CompletionResult | None = None

        # Initialize position evaluator
        self.evaluator = PositionEvaluator()

        self.prompt_template = """
        You are playing chess as {color}. Current position critical analysis:

        MOVE HISTORY ANALYSIS:
        Previous Positions Repeated: {position_repetitions}
        Stagnation Warning: {stagnation_status}
        Position Progress Score: {position_progress}
        Material Tension: {material_tension}
        Position Dynamism: {position_dynamism}
        Development Score: {development_score}

        TACTICAL OPPORTUNITIES (MUST CONSIDER FIRST):
        Winning Captures Available:
        {capture_analysis}

        DEFENSE ANALYSIS:
        {defense_analysis}

        VULNERABILITY ANALYSIS:
        {vulnerability_analysis}

        Material Status:
        {material_count}
        Material Balance: {material_balance}

        POSITION EVALUATION:
        Center Control: {center_control}
        Development Status: {development_status}
        King Safety: {king_safety}
        Undefended Pieces: {undefended_pieces}
        Exposed Pieces: {exposed_pieces}
        Board: {ascii_board}

        Legal moves by priority:
        1. WINNING CAPTURES/CHECKS (Must play if available):
        {forcing_moves}

        2. DEVELOPING MOVES (Play if no winning captures):
        {developing_moves}

        3. POSITIONAL MOVES (Last resort):
        {positional_moves}

        CRITICAL: Select ONE move from the above categories.
        Respond ONLY with the UCI notation (e.g., 'e2e4').

        Decision Priority:
        1. Capitalize on opponent's undefended pieces.
        2. Defend against immediate threats/mate threats.
        3. Execute winning captures/tactics.
        4. Protect your vulnerable pieces.
        5. Avoid repetitions and play to win
        6. When your pieces are captured, you must capture back.

        Best move given state of the game(UCI notation only):
        """

    def _get_piece_locations(self, board: chess.Board) -> tuple[list[str], list[str]]:
        return self.evaluator.get_piece_locations(board)

    def _get_material_count(self, board: chess.Board) -> str:
        return self.evaluator.get_material_count(board)

    def _analyze_material_tension(self, board: chess.Board) -> str:
        return self.evaluator.analyze_material_tension(board)

    def _annotate_moves(self, board: chess.Board) -> str:
        return self.evaluator.annotate_moves(board)

    def _analyze_position_repetition(self, board: chess.Board) -> dict:
        current_fen = board.fen().split(' ')[0]

        recent_history = [*self.move_history[-7:], current_fen]
        repetitions = sum(1 for pos in recent_history if pos == current_fen)

        is_stagnating = repetitions >= self.stagnation_threshold

        if len(self.move_history) >= 3:
            recent_positions = [*self.move_history[-3:], current_fen]
            unique_positions = len(set(recent_positions))
            progress_score = unique_positions / 4.0
        else:
            progress_score = 1.0

        return {
            "repetitions": repetitions,
            "is_stagnating": is_stagnating,
            "progress_score": progress_score
        }

    def _analyze_position_progress(self, board: chess.Board, move: chess.Move) -> float:
        return self.evaluator.analyze_position_progress(board, move)

    def _analyze_position_dynamism(self, board: chess.Board) -> str:
        return self.evaluator.analyze_position_dynamism(board)

    def _get_castling_rights(self, board: chess.Board) -> str:
        return self.evaluator.get_castling_rights(board)

    def _analyze_capture_value(self, board: chess.Board, move: chess.Move) -> int:
        return self.evaluator.analyze_capture_value(board, move)

    def _calculate_development_score(self, board: chess.Board) -> str:
        return self.evaluator.calculate_development_score(board)

    def _analyze_captures(self, board: chess.Board) -> str:
        return self.evaluator.analyze_captures(board)

    def _analyze_threats(self, board: chess.Board) -> str:
        return self.evaluator.analyze_threats(board)

    def _evaluate_capture(self, board: chess.Board, move: chess.Move) -> float:
        return self.evaluator.evaluate_capture(board, move)

    def _categorize_moves(self, board: chess.Board) -> dict[str, str]:
        return self.evaluator.categorize_moves(board)

    def _analyze_defense(self, board: chess.Board) -> str:
        return self.evaluator.analyze_defense(board)

    def _analyze_vulnerabilities(self, board: chess.Board) -> str:
        return self.evaluator.analyze_vulnerabilities(board)

    def _analyze_king_safety(self, board: chess.Board) -> str:
        return self.evaluator.analyze_king_safety(board)

    def _is_pinned(self, board: chess.Board, square: int) -> bool:
        return self.evaluator.is_pinned(board, square)

    def _analyze_pawn_structure(self, board: chess.Board) -> str:
        return self.evaluator.analyze_pawn_structure(board)

    def _analyze_undefended_pieces(self, board: chess.Board) -> str:
        return self.evaluator.analyze_undefended_pieces(board)

    def _analyze_exposed_pieces(self, board: chess.Board) -> str:
        return self.evaluator.analyze_exposed_pieces(board)

    def _analyze_material_balance(self, board: chess.Board) -> str:
        return self.evaluator.analyze_material_balance(board)

    def _analyze_center_control(self, board: chess.Board) -> str:
        return self.evaluator.analyze_center_control(board)

    def _analyze_development_status(self, board: chess.Board) -> str:
        return self.evaluator.analyze_development_status(board)

    def _create_prompt(self, fen: str) -> str:
        board = chess.Board(fen)
        moves = self._categorize_moves(board)
        position_analysis = self._analyze_position_repetition(board)

        return self.prompt_template.format(
            color="White" if board.turn == chess.WHITE else "Black",
            position_repetitions=position_analysis["repetitions"],
            stagnation_status="STAGNATING - Force dynamic play!" if position_analysis["is_stagnating"] else "Normal",
            position_progress=f"{position_analysis['progress_score']:.2f}",
            material_tension=self.evaluator.analyze_material_tension(board),
            position_dynamism=self.evaluator.analyze_position_dynamism(board),
            development_score=self.evaluator.calculate_development_score(board),
            defense_analysis=self.evaluator.analyze_defense(board),
            vulnerability_analysis=self.evaluator.analyze_vulnerabilities(board),
            capture_analysis=self.evaluator.analyze_captures(board),
            king_safety=self.evaluator.analyze_king_safety(board),
            undefended_pieces=self.evaluator.analyze_undefended_pieces(board),
            exposed_pieces=self.evaluator.analyze_exposed_pieces(board),
            ascii_board=str(board),
            material_count=self.evaluator.get_material_count(board),
            material_balance=self.evaluator.analyze_material_balance(board),
            center_control=self.evaluator.analyze_center_control(board),
            development_status=self.evaluator.analyze_development_status(board),
            forcing_moves=moves['forcing_moves'],
            developing_moves=moves['developing_moves'],
            positional_moves=moves['positional_moves']
        )

    def _validate_move(self, move_str: str, board: chess.Board) -> str:
        move_str = move_str.strip().lower()

        prefixes = ["move:", "i choose", "my move is", "play", "'", '"', "`"]
        for prefix in prefixes:
            if move_str.startswith(prefix):
                move_str = move_str[len(prefix):].strip()

        suffixes = ["'", '"', "`", ".", ",", ":", ";"]
        for suffix in suffixes:
            if move_str.endswith(suffix):
                move_str = move_str[:-len(suffix)].strip()

        if not (4 <= len(move_str) <= 5):
            raise ValueError(f"Invalid move format: {move_str}")

        try:
            move = chess.Move.from_uci(move_str)
        except ValueError as err:
            raise ValueError(f"Invalid UCI format: {move_str}") from err

        if move not in board.legal_moves:
            legal_moves = [m.uci() for m in board.legal_moves]
            raise ValueError(f"Illegal move {move_str}. Legal moves are: {', '.join(legal_moves)}")

        return move_str

    def _is_valid_square(self, square: str) -> bool:
        if len(square) != 2:
            return False
        file, rank = square[0], square[1]
        return (
            file in 'abcdefgh' and
            rank in '12345678'
        )

    async def get_move(self, fen: str) -> str:
        board = chess.Board(fen)
        max_retries = 3
        errors = []

        for attempt in range(max_retries):
            try:
                move_str = await self._get_move_from_model(fen)
                validated_move = self._validate_move(move_str, board)

                current_fen = board.fen().split(' ')[0]
                self.move_history.append(current_fen)

                return validated_move
            except ValueError as e:
                errors.append(f"Attempt {attempt + 1}: {e!s}")
                continue

        legal_moves = list(board.legal_moves)
        if legal_moves:
            fallback_move = legal_moves[0].uci()
            current_fen = board.fen().split(' ')[0]
            self.move_history.append(current_fen)
            return fallback_move

        raise ValueError(f"Failed to get valid move after {max_retries} attempts. Errors: {'; '.join(errors)}")

    async def get_move_with_result(self, fen: str) -> tuple[str, "CompletionResult"]:
        board = chess.Board(fen)
        max_retries = 3
        errors = []

        for attempt in range(max_retries):
            try:
                move_str = await self._get_move_from_model(fen)
                validated_move = self._validate_move(move_str, board)

                current_fen = board.fen().split(' ')[0]
                self.move_history.append(current_fen)

                return validated_move, self.last_completion_result or CompletionResult(
                    text=move_str,
                    tokens_in=None,
                    tokens_out=None,
                    latency_ms=0,
                    raw_response=None,
                )
            except ValueError as e:
                errors.append(f"Attempt {attempt + 1}: {e!s}")
                continue

        legal_moves = list(board.legal_moves)
        if legal_moves:
            fallback_move = legal_moves[0].uci()
            current_fen = board.fen().split(' ')[0]
            self.move_history.append(current_fen)
            return fallback_move, self.last_completion_result or CompletionResult(
                text=fallback_move,
                tokens_in=None,
                tokens_out=None,
                latency_ms=0,
                raw_response=None,
            )

        raise ValueError(f"Failed to get valid move after {max_retries} attempts. Errors: {'; '.join(errors)}")

    @abstractmethod
    async def _get_move_from_model(self, fen: str) -> str:
        pass


class OpenAIChessAI(ChessAI):
    def __init__(self, model_type: ModelType):
        super().__init__(name=model_type.value)
        self.client = OpenAI()
        self.model = model_type.value

    async def _get_move_from_model(self, fen: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": self._create_prompt(fen)
            }]
        )
        text = (response.choices[0].message.content or "").strip()
        self.last_completion_result = CompletionResult(
            text=text,
            tokens_in=response.usage.prompt_tokens if response.usage else None,
            tokens_out=response.usage.completion_tokens if response.usage else None,
            latency_ms=0,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )
        return text


class AnthropicChessAI(ChessAI):
    def __init__(self, model_type: ModelType):
        super().__init__(name=model_type.value)
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

    async def _get_move_from_model(self, fen: str) -> str:
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": self._create_prompt(fen)
            }]
        )
        text = (response.content[0].text if response.content and hasattr(response.content[0], 'text') and response.content[0].text else "").strip()
        self.last_completion_result = CompletionResult(
            text=text,
            tokens_in=response.usage.input_tokens if response.usage else None,
            tokens_out=response.usage.output_tokens if response.usage else None,
            latency_ms=0,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )
        return text


class LlamaChessAI(ChessAI):
    def __init__(self, model_type: ModelType):
        super().__init__(name=model_type.value)
        self.model_name = model_type.value.lower()

    async def _get_move_from_model(self, fen: str) -> str:
        response = ollama.generate(
            model=self.model_name,
            prompt=self._create_prompt(fen)
        )
        text = response['response'].strip()
        self.last_completion_result = CompletionResult(
            text=text,
            tokens_in=response.get('prompt_eval_count'),
            tokens_out=response.get('eval_count'),
            latency_ms=0,
            raw_response=dict(response),
        )
        return str(text)
