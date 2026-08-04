"""Test fixtures for move parser - curated real LLM outputs."""

LLM_OUTPUTS = [
    # OpenAI GPT-4o outputs
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "I will play e2e4",
        "expected": "e2e4",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "<thinking>White should develop the knight to f3 to control the center.</thinking>\ng1f3",
        "expected": "g1f3",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "The best move is e7e5",
        "expected": "e7e5",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "Move: d2d4",
        "expected": "d2d4",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "My move is b1c3",
        "expected": "b1c3",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "I choose e2e4",
        "expected": "e2e4",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "Play g1f3",
        "expected": "g1f3",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "`e2e4`",
        "expected": "e2e4",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "\"e2e4\"",
        "expected": "e2e4",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "The move e2e4 looks best",
        "expected": "e2e4",
    },
    
    # Anthropic Claude outputs
    {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "text": "<thinking>White has a winning capture on d5 with the knight.</thinking>\nf3d5",
        "expected": "f3d5",
    },
    {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "text": "I will play e7e5 to control the center.",
        "expected": "e7e5",
    },
    {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "text": "Best move: g1f3",
        "expected": "g1f3",
    },
    {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "text": "move: d2d4",
        "expected": "d2d4",
    },
    {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "text": "I choose b1c3",
        "expected": "b1c3",
    },
    
    # Google Gemini outputs
    {
        "provider": "google",
        "model": "gemini-1.5-pro",
        "text": "e2e4",
        "expected": "e2e4",
    },
    {
        "provider": "google",
        "model": "gemini-1.5-pro",
        "text": "I'll play g1f3 to develop my knight.",
        "expected": "g1f3",
    },
    {
        "provider": "google",
        "model": "gemini-1.5-pro",
        "text": "The best move is e7e5",
        "expected": "e7e5",
    },
    {
        "provider": "google",
        "model": "gemini-1.5-pro",
        "text": "move: d2d4",
        "expected": "d2d4",
    },
    {
        "provider": "google",
        "model": "gemini-1.5-pro",
        "text": "My move is b1c3",
        "expected": "b1c3",
    },
    
    # Ollama local model outputs (often more verbose)
    {
        "provider": "ollama",
        "model": "llama3.2",
        "text": "I will play e2e4 because it controls the center.",
        "expected": "e2e4",
    },
    {
        "provider": "ollama",
        "model": "llama3.2",
        "text": "<thinking>Developing the knight to f3 is a solid choice.</thinking>\ng1f3",
        "expected": "g1f3",
    },
    {
        "provider": "ollama",
        "model": "llama3.2",
        "text": "e7e5",
        "expected": "e7e5",
    },
    {
        "provider": "ollama",
        "model": "qwen2.5",
        "text": "Move: d2d4",
        "expected": "d2d4",
    },
    {
        "provider": "ollama",
        "model": "qwen2.5",
        "text": "My move is b1c3",
        "expected": "b1c3",
    },
    
    # Edge cases and malformed outputs
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "I think the best move would be e2e4, but I could also play d2d4",
        "expected": "e2e4",  # First valid move
    },
    {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "text": "Not sure... maybe g1f3? Or e2e4?",
        "expected": "g1f3",  # First valid move
    },
    {
        "provider": "google",
        "model": "gemini-1.5-pro",
        "text": "I'll play e2e4 (the king's pawn opening)",
        "expected": "e2e4",
    },
    {
        "provider": "ollama",
        "model": "llama3.2",
        "text": "The move is e2e4.",
        "expected": "e2e4",
    },
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "move: e2e4 (best move)",
        "expected": "e2e4",
    },
    
    # Promotion moves
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "I will promote with a7a8q",
        "expected": "a7a8q",
    },
    {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "text": "e7e8q",
        "expected": "e7e8q",
    },
    {
        "provider": "google",
        "model": "gemini-1.5-pro",
        "text": "Promote to queen: d7d8q",
        "expected": "d7d8q",
    },
    
    # Invalid/ambiguous outputs (should return None)
    {
        "provider": "openai",
        "model": "gpt-4o",
        "text": "I don't know what to play",
        "expected": None,
    },
    {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "text": "Let me think about this...",
        "expected": None,
    },
    {
        "provider": "google",
        "model": "gemini-1.5-pro",
        "text": "Invalid move: z9z9",
        "expected": None,
    },
    {
        "provider": "ollama",
        "model": "llama3.2",
        "text": "move: e2e9",
        "expected": None,
    },
]