"""Demo-game generation for the hosted Streamlit experience.

The hosted demo on Streamlit Cloud must show the product at its best — even when
product at its best — even when the visitor has no API key. The genuine replay
material comes from real benchmark runs in ``runs/`` (auto-generated), but a
fresh deploy may land without any real run yet. This module seeds one
clearly-labeled demo run so the landing page, history, and analytical
dashboard always have something dimensional to render.

Implementation honesty:
- Moves come from real Stockfish playing itself (real legal moves, real
  game progression, real terminations) — no scripted moves.
- Eval/quality fields come from the real ``StockfishEvaluator`` on every
  position the demo visits — same code path the benchmark runner uses.
- Thinking traces are short, clearly-tagged演示 traces (``is_demo=True``
  in the run config) cribbed from real LLM chess transcripts. They exist
  purely so the "Thinking Trace Viewer" analytics section demonstrates
  what it will look like with real LLM data. They are NOT a claim that
  a model produced them — the run config says so.
"""
