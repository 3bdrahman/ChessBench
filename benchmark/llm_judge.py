"""LLM-as-Judge for evaluating chess reasoning quality."""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from providers.chess_ai import ProviderChessAI
from providers.base import ChatMessage, CompletionResult
from providers import get_provider
import re


class JudgmentCriteria(Enum):
    """Criteria for evaluating chess reasoning."""
    TACTICAL_AWARENESS = "tactical_awareness"
    POSITIONAL_UNDERSTANDING = "positional_understanding"
    CALCULATION_DEPTH = "calculation_depth"
    MOVE_JUSTIFICATION = "move_justification"
    ERROR_RECOGNITION = "error_recognition"
    OVERALL_QUALITY = "overall_quality"


@dataclass
class ReasoningJudgment:
    """Judgment of a single move's reasoning."""
    move_number: int
    player: str
    fen: str
    move_uci: str
    reasoning_trace: str
    scores: Dict[JudgmentCriteria, float]  # 0-10 scale
    overall_score: float
    feedback: str
    judge_model: str


@dataclass
class GameJudgment:
    """Complete game judgment."""
    game_id: str
    white_player: str
    black_player: str
    move_judgments: List[ReasoningJudgment]
    white_avg_scores: Dict[JudgmentCriteria, float]
    black_avg_scores: Dict[JudgmentCriteria, float]
    correlation_with_result: float
    timestamp: str


class LLMJudge:
    """LLM-as-Judge for evaluating chess reasoning quality."""
    
    JUDGE_PROMPT = """You are an expert chess analyst evaluating the quality of an LLM's chess reasoning.

You will be given:
1. The position (FEN)
2. The move played (UCI)
3. The LLM's reasoning trace (their thought process)

Evaluate the reasoning on these criteria (score 0-10 each):

**Tactical Awareness (0-10):** Does the reasoning identify tactical motifs (pins, forks, skewers, discovered attacks, mate threats)? Does it calculate forcing sequences accurately?

**Positional Understanding (0-10):** Does the reasoning show understanding of pawn structure, piece activity, king safety, space, weak squares, and strategic plans?

**Calculation Depth (0-10):** How deep and accurate is the calculation? Does it consider opponent replies? Does it see 2-3+ moves ahead?

**Move Justification (0-10):** Is the chosen move well-justified by the reasoning? Does the reasoning logically lead to the move?

**Error Recognition (0-10):** Does the reasoning acknowledge risks, alternative moves, or potential downsides of the chosen move?

**Overall Quality (0-10):** Holistic assessment of reasoning quality.

Return ONLY a JSON object with this exact structure:
{
  "tactical_awareness": <score>,
  "positional_understanding": <score>,
  "calculation_depth": <score>,
  "move_justification": <score>,
  "error_recognition": <score>,
  "overall_quality": <score>,
  "feedback": "<brief explanation of strengths/weaknesses>"
}

Do not include any other text. Scores must be numbers 0-10 (can be decimal)."""

    def __init__(self, judge_provider: str, judge_model: str, judge_api_key: str, **params):
        self.judge_provider = judge_provider
        self.judge_model = judge_model
        self.judge_api_key = judge_api_key
        self.params = params
        self.provider = get_provider(judge_provider)
        if not self.provider:
            raise ValueError(f"Unknown provider: {judge_provider}")
        if not self.provider.validate_key(judge_api_key):
            raise ValueError(f"Invalid API key for {judge_provider}")
        self.judge_model_name = f"{judge_provider}:{judge_model}"
    
    async def _complete_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Complete a prompt with retry logic."""
        for attempt in range(max_retries):
            try:
                result = await self.provider.complete(
                    self.judge_api_key,
                    self.judge_model,
                    [ChatMessage(role="user", content=prompt)],
                    **self.params
                )
                if result.text and result.text.strip():
                    return result.text.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
        raise ValueError(f"Failed to get valid response after {max_retries} attempts")
    
    async def judge_reasoning(
        self, 
        fen: str, 
        move_uci: str, 
        reasoning_trace: str,
        move_number: int,
        player: str
    ) -> ReasoningJudgment:
        """Judge a single move's reasoning."""
        
        # Build the evaluation prompt
        eval_prompt = f"""{self.JUDGE_PROMPT}

Position (FEN): {fen}
Move played: {move_uci}
Move number: {move_number}
Player: {player}

Reasoning trace:
{reasoning_trace}"""
        
        # Get judgment from judge LLM with retry logic
        try:
            result = await self._complete_with_retry(eval_prompt)
            
            # Extract JSON from response
            judgment_data = self._extract_json(result)
            
            if judgment_data:
                scores = {
                    JudgmentCriteria.TACTICAL_AWARENESS: judgment_data.get("tactical_awareness", 0),
                    JudgmentCriteria.POSITIONAL_UNDERSTANDING: judgment_data.get("positional_understanding", 0),
                    JudgmentCriteria.CALCULATION_DEPTH: judgment_data.get("calculation_depth", 0),
                    JudgmentCriteria.MOVE_JUSTIFICATION: judgment_data.get("move_justification", 0),
                    JudgmentCriteria.ERROR_RECOGNITION: judgment_data.get("error_recognition", 0),
                    JudgmentCriteria.OVERALL_QUALITY: judgment_data.get("overall_quality", 0),
                }
                feedback = judgment_data.get("feedback", "")
                overall = scores[JudgmentCriteria.OVERALL_QUALITY]
            else:
                # Fallback scoring
                scores = {c: 5.0 for c in JudgmentCriteria}
                feedback = "Failed to parse judgment"
                overall = 5.0
            
        except Exception as e:
            scores = {c: 0.0 for c in JudgmentCriteria}
            feedback = f"Judgment error: {str(e)}"
            overall = 0.0
        
        return ReasoningJudgment(
            move_number=move_number,
            player=player,
            fen=fen,
            move_uci=move_uci,
            reasoning_trace=reasoning_trace,
            scores=scores,
            overall_score=overall,
            feedback=feedback,
            judge_model=self.judge_model_name
        )
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object from text."""
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON in code blocks
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass
        
        return None
    
    async def judge_game(
        self, 
        game_id: str,
        white_player: str,
        black_player: str,
        moves_data: List[Dict]
    ) -> GameJudgment:
        """Judge all moves in a game."""
        move_judgments = []
        
        for move_data in moves_data:
            judgment = await self.judge_reasoning(
                fen=move_data["fen_before"],
                move_uci=move_data["move_uci"],
                reasoning_trace=move_data.get("thinking_trace", ""),
                move_number=move_data["move_number"],
                player=move_data["player"]
            )
            move_judgments.append(judgment)
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)
        
        # Calculate average scores per player
        white_scores = {c: 0.0 for c in JudgmentCriteria}
        black_scores = {c: 0.0 for c in JudgmentCriteria}
        white_count = 0
        black_count = 0
        
        for j in move_judgments:
            if j.player == white_player:
                for c in JudgmentCriteria:
                    white_scores[c] += j.scores[c]
                white_count += 1
            else:
                for c in JudgmentCriteria:
                    black_scores[c] += j.scores[c]
                black_count += 1
        
        for c in JudgmentCriteria:
            white_scores[c] = white_scores[c] / white_count if white_count > 0 else 0
            black_scores[c] = black_scores[c] / black_count if black_count > 0 else 0
        
        # Calculate correlation with result (simplified)
        # Higher reasoning quality should correlate with better results
        white_overall = white_scores[JudgmentCriteria.OVERALL_QUALITY]
        black_overall = black_scores[JudgmentCriteria.OVERALL_QUALITY]
        
        # Simple correlation: if white won and white_overall > black_overall, positive correlation
        correlation = 0.0  # Would need actual game result
        
        return GameJudgment(
            game_id=game_id,
            white_player=white_player,
            black_player=black_player,
            move_judgments=move_judgments,
            white_avg_scores=white_scores,
            black_avg_scores=black_scores,
            correlation_with_result=correlation,
            timestamp=""
        )


async def demo():
    """Demo the LLM judge with sample reasoning."""
    import os
    
    # This would need actual API keys
    # judge = LLMJudge("openai", "gpt-4o", os.getenv("OPENAI_API_KEY"))
    
    # Sample reasoning to test JSON extraction
    sample_reasoning = """{
  "tactical_awareness": 8.5,
  "positional_understanding": 7.0,
  "calculation_depth": 8.0,
  "move_justification": 9.0,
  "error_recognition": 6.5,
  "overall_quality": 7.8,
  "feedback": "Strong tactical vision with accurate calculation of the knight fork. Good justification for the move. Could improve by acknowledging the opponent's counterplay on the c-file."
}"""
    
    judge = LLMJudge("mock", "mock", "")
    extracted = judge._extract_json(sample_reasoning)
    print("Extracted judgment:", extracted)
    
    # Test with text wrapper
    wrapped = f"Here is my evaluation:\n\n```json\n{sample_reasoning}\n```\n\nEnd."
    extracted2 = judge._extract_json(wrapped)
    print("Extracted from wrapper:", extracted2)


if __name__ == "__main__":
    asyncio.run(demo())