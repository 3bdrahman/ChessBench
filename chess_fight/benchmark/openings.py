"""ECO opening book for benchmark games."""

import random
from collections import defaultdict
from typing import Any

import chess

# Standard ECO openings (100 positions, ~2-3 moves deep)
# Format: (ECO code, name, moves in UCI, starting FEN if not initial)
ECO_OPENINGS = [
    # A00-A09: Uncommon openings
    ("A00", "Polish Opening", ["b2b4"]),
    ("A00", "Bird's Opening", ["f2f4"]),
    ("A00", "Grob's Attack", ["g2g4"]),
    ("A00", "English Opening", ["c2c4"]),
    ("A00", "Reti Opening", ["g1f3", "d7d5", "c2c4"]),
    ("A00", "King's Indian Attack", ["g1f3", "d7d5", "g2g3"]),
    ("A00", "Larsen's Opening", ["b2b3"]),
    ("A00", "Sokolsky Opening", ["b2b4"]),
    ("A01", "Nimzowitsch-Larsen Attack", ["b2b3", "e7e5", "c1b2"]),
    ("A02", "Bird's Opening", ["f2f4", "d7d5"]),
    ("A03", "Bird's Opening, Dutch Variation", ["f2f4", "f7f5"]),
    ("A04", "Reti Opening", ["g1f3", "d7d5", "c2c4"]),
    ("A05", "Reti Opening", ["g1f3", "d7d5", "c2c4", "c7c6"]),
    ("A06", "Reti Opening, King's Indian Attack", ["g1f3", "g8f6", "g2g3"]),
    ("A07", "King's Indian Attack", ["g1f3", "g8f6", "g2g3", "d7d5"]),
    ("A08", "Reti Opening", ["g1f3", "d7d5", "c2c4", "e7e6"]),
    ("A09", "Reti Opening", ["g1f3", "d7d5", "c2c4", "g8f6"]),

    # A10-A19: English Opening
    ("A10", "English Opening", ["c2c4", "e7e5"]),
    ("A11", "English, Four Knights", ["c2c4", "g8f6", "b1c3", "e7e5"]),
    ("A12", "English, Caro-Kann Defensive System", ["c2c4", "c7c6"]),
    ("A13", "English, Agincourt Defense", ["c2c4", "e7e6"]),
    ("A14", "English, Neo-Grunfeld", ["c2c4", "g8f6", "b1c3", "d7d5"]),
    ("A15", "English, Anglo-Indian", ["c2c4", "g8f6", "g2g3"]),
    ("A16", "English, Mikenas-Carls", ["c2c4", "g8f6", "d2d4", "e7e6"]),
    ("A17", "English, Mikenas-Carls", ["c2c4", "g8f6", "d2d4", "e7e6", "b1c3"]),
    ("A18", "English, Anglo-Indian", ["c2c4", "g8f6", "g2g3", "d7d5"]),
    ("A19", "English, Anglo-Indian", ["c2c4", "g8f6", "g2g3", "d7d5", "c1g2"]),

    # A20-A29: English Opening
    ("A20", "English Opening", ["c2c4", "e7e5", "g2g3"]),
    ("A21", "English, Bremen System", ["c2c4", "e7e5", "g2g3", "g8f6"]),
    ("A22", "English, Bremen System", ["c2c4", "e7e5", "g2g3", "g8f6", "b1c3"]),
    ("A23", "English, Bremen System", ["c2c4", "e7e5", "g2g3", "g8f6", "b1c3", "f8b4"]),
    ("A24", "English, Bremen System", ["c2c4", "e7e5", "g2g3", "g8f6", "b1c3", "f8e7"]),
    ("A25", "English, Closed", ["c2c4", "g8f6", "g2g3", "e7e6", "b1c3", "f8b4"]),
    ("A26", "English, Closed", ["c2c4", "g8f6", "g2g3", "d7d5", "c1g2", "c7c6"]),
    ("A27", "English, Closed", ["c2c4", "g8f6", "g2g3", "d7d5", "c1g2", "c7c6", "b1c3"]),
    ("A28", "English, Closed", ["c2c4", "g8f6", "g2g3", "e7e6", "b1c3", "f8e7"]),
    ("A29", "English, Four Knights, Kingside Fianchetto", ["c2c4", "g8f6", "g2g3", "e7e5", "b1c3", "b8c6", "c1g2"]),

    # A30-A39: English, Symmetrical
    ("A30", "English, Symmetrical", ["c2c4", "c7c5"]),
    ("A31", "English, Symmetrical", ["c2c4", "c7c5", "b1c3"]),
    ("A32", "English, Symmetrical", ["c2c4", "c7c5", "b1c3", "g8f6"]),
    ("A33", "English, Symmetrical", ["c2c4", "c7c5", "b1c3", "g8f6", "g2g3"]),
    ("A34", "English, Symmetrical", ["c2c4", "c7c5", "b1c3", "g8f6", "g2g3", "c1g2"]),
    ("A35", "English, Symmetrical", ["c2c4", "c7c5", "b1c3", "g8f6", "g2g3", "c1g2", "b8c6"]),
    ("A36", "English, Symmetrical", ["c2c4", "c7c5", "g1f3"]),
    ("A37", "English, Symmetrical", ["c2c4", "c7c5", "g1f3", "g8f6"]),
    ("A38", "English, Symmetrical", ["c2c4", "c7c5", "g1f3", "g8f6", "g2g3"]),
    ("A39", "English, Symmetrical", ["c2c4", "c7c5", "g1f3", "g8f6", "g2g3", "c1g2"]),

    # A40-A49: Queen's Pawn Games
    ("A40", "Queen's Pawn Game", ["d2d4", "d7d5"]),
    ("A41", "Queen's Pawn Game", ["d2d4", "d7d5", "g1f3"]),
    ("A42", "Modern Defense", ["d2d4", "g7g6"]),
    ("A43", "Old Benoni Defense", ["d2d4", "c7c5"]),
    ("A44", "Old Benoni Defense", ["d2d4", "c7c5", "d4d5", "e7e6"]),
    ("A45", "Trompowsky Attack", ["d2d4", "g8f6", "c1g5"]),
    ("A46", "Trompowsky Attack", ["d2d4", "g8f6", "c1g5", "e7e6"]),
    ("A47", "Queen's Indian Defense", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "c1b4"]),
    ("A48", "King's Indian Defense", ["d2d4", "g8f6", "c2c4", "g7g6"]),
    ("A49", "King's Indian Defense", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"]),

    # A50-A59: Queen's Pawn Games
    ("A50", "Queen's Pawn Game", ["d2d4", "d7d5", "c2c4"]),
    ("A51", "Budapest Gambit", ["d2d4", "g8f6", "c2c4", "e7e5"]),
    ("A52", "Budapest Gambit", ["d2d4", "g8f6", "c2c4", "e7e5", "d4e5", "g8f6"]),
    ("A53", "Old Indian Defense", ["d2d4", "g8f6", "c2c4", "d7d6"]),
    ("A54", "Old Indian Defense", ["d2d4", "g8f6", "c2c4", "d7d6", "b1c3", "e7e5"]),
    ("A55", "Old Indian Defense", ["d2d4", "g8f6", "c2c4", "d7d6", "b1c3", "e7e5", "d4d5"]),
    ("A56", "Benoni Defense", ["d2d4", "c7c5", "d4d5", "d7d6"]),
    ("A57", "Benko Gambit", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5"]),
    ("A58", "Benko Gambit", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5", "c4b5", "a7a6"]),
    ("A59", "Benko Gambit", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5", "c4b5", "a7a6", "b1c3"]),

    # A60-A69: Benoni Defense
    ("A60", "Benoni Defense", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6"]),
    ("A61", "Benoni Defense", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5"]),
    ("A62", "Benoni Defense", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6"]),
    ("A63", "Benoni Defense", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7"]),
    ("A64", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4"]),
    ("A65", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6"]),
    ("A66", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4"]),
    ("A67", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7"]),
    ("A68", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3"]),
    ("A69", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8"]),

    # A70-A79: Benoni Defense
    ("A70", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8", "e1e2"]),
    ("A71", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8", "e1e2", "d8e8"]),
    ("A72", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8", "e1e2", "d8e8", "h2h3"]),
    ("A73", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8", "e1e2", "d8e8", "h2h3", "c8g4"]),
    ("A74", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8", "e1e2", "d8e8", "h2h3", "c8g4", "c1d2"]),
    ("A75", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8", "e1e2", "d8e8", "h2h3", "c8g4", "c1d2", "g4f3"]),
    ("A76", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8", "e1e2", "d8e8", "h2h3", "c8g4", "c1d2", "g4f3", "g2f3"]),
    ("A77", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8", "e1e2", "d8e8", "h2h3", "c8g4", "c1d2", "g4f3", "g2f3", "d8d7"]),
    ("A78", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8", "e1e2", "d8e8", "h2h3", "c8g4", "c1d2", "g4f3", "g2f3", "d8d7", "e8e7"]),
    ("A79", "Modern Benoni", ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6e5", "d5e6", "f8e7", "e2e4", "g7g6", "f2f4", "f8g7", "g1f3", "g8h8", "e1e2", "d8e8", "h2h3", "c8g4", "c1d2", "g4f3", "g2f3", "d8d7", "e8e7", "c8g4"]),

    # A80-A99: Dutch Defense
    ("A80", "Dutch Defense", ["d2d4", "f7f5"]),
    ("A81", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6"]),
    ("A82", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5"]),
    ("A83", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6"]),
    ("A84", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3"]),
    ("A85", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6"]),
    ("A86", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5"]),
    ("A87", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5"]),
    ("A88", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2"]),
    ("A89", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7"]),

    # A90-A99: Dutch Defense
    ("A90", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7", "d1e2"]),
    ("A91", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7", "d1e2", "c8d7"]),
    ("A92", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7", "d1e2", "c8d7", "e1d1"]),
    ("A93", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7", "d1e2", "c8d7", "e1d1", "f8d6"]),
    ("A94", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7", "d1e2", "c8d7", "e1d1", "f8d6", "b1d2"]),
    ("A95", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7", "d1e2", "c8d7", "e1d1", "f8d6", "b1d2", "c8d7"]),
    ("A96", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7", "d1e2", "c8d7", "e1d1", "f8d6", "b1d2", "c8d7", "d1d2"]),
    ("A97", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7", "d1e2", "c8d7", "e1d1", "f8d6", "b1d2", "c8d7", "d1d2", "c8d7"]),
    ("A98", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7", "d1e2", "c8d7", "e1d1", "f8d6", "b1d2", "c8d7", "d1d2", "c8d7", "d2d3"]),
    ("A99", "Dutch Defense", ["d2d4", "f7f5", "g1f3", "g8f6", "c1g5", "e7e6", "e2e3", "c7c6", "d4d5", "e6e5", "c1d2", "d8e7", "d1e2", "c8d7", "e1d1", "f8d6", "b1d2", "c8d7", "d1d2", "c8d7", "d2d3", "c8d7"]),
    # B00-B99: Semi-Open Games
    ("B00", "King's Pawn Opening", ["e2e4"]),
    ("B00", "Owen's Defense", ["e2e4", "b7b6"]),
    ("B00", "Nimzowitsch Defense", ["e2e4", "b8c6"]),
    ("B00", "St. George Defense", ["e2e4", "a7a6"]),
    ("B01", "Scandinavian Defense", ["e2e4", "d7d5"]),
    ("B01", "Scandinavian, Mieses-Kotroc", ["e2e4", "d7d5", "e4d5", "d8d5"]),
    ("B01", "Scandinavian, Modern", ["e2e4", "d7d5", "e4d5", "g8f6"]),
    ("B02", "Alekhine's Defense", ["e2e4", "g8f6"]),
    ("B03", "Alekhine's Defense", ["e2e4", "g8f6", "e4e5", "f6d5"]),
    ("B04", "Alekhine's Defense, Modern", ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4"]),
    ("B05", "Alekhine's Defense, Modern", ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6"]),
    ("B06", "Modern Defense", ["e2e4", "g7g6"]),
    ("B07", "Pirc Defense", ["e2e4", "d7d6"]),
    ("B08", "Pirc, Classical", ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "g1f3"]),
    ("B09", "Pirc, Austrian Attack", ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f2f4"]),
    ("B10", "Caro-Kann Defense", ["e2e4", "c7c6"]),
    ("B11", "Caro-Kann, Two Knights", ["e2e4", "c7c6", "b1c3", "d7d5", "g1f3"]),
    ("B12", "Caro-Kann, Advance", ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5"]),
    ("B13", "Caro-Kann, Exchange", ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5"]),
    ("B14", "Caro-Kann, Panov-Botvinnik", ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5", "c2c4"]),
    ("B15", "Caro-Kann, Classical", ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "g8f6"]),
    ("B16", "Caro-Kann, Karpov", ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5"]),
    ("B17", "Caro-Kann, Karpov", ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7"]),
    ("B18", "Caro-Kann, Classical", ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7", "f1d3"]),
    ("B20", "Sicilian Defense", ["e2e4", "c7c5"]),
    ("B21", "Sicilian, Smith-Morra Gambit", ["e2e4", "c7c5", "d2d4", "c5d4", "c2c3"]),
    ("B22", "Sicilian, Alapin", ["e2e4", "c7c5", "c2c3"]),
    ("B23", "Sicilian, Closed", ["e2e4", "c7c5", "b1c3"]),
    ("B24", "Sicilian, Closed", ["e2e4", "c7c5", "b1c3", "b8c6"]),
    ("B25", "Sicilian, Closed", ["e2e4", "c7c5", "b1c3", "b8c6", "g2g3"]),
    ("B26", "Sicilian, Closed", ["e2e4", "c7c5", "b1c3", "b8c6", "g2g3", "g7g6"]),
    ("B27", "Sicilian, Closed, Fianchetto", ["e2e4", "c7c5", "b1c3", "b8c6", "g2g3", "g7g6", "f1g2"]),
    ("B28", "Sicilian, Closed", ["e2e4", "c7c5", "b1c3", "b8c6", "g2g3", "g7g6", "f1g2", "f8g7"]),
    ("B30", "Sicilian, Old Sicilian", ["e2e4", "c7c5", "g1f3", "b8c6"]),
    ("B31", "Sicilian, Rossolimo", ["e2e4", "c7c5", "g1f3", "b8c6", "f1b5"]),
    ("B32", "Sicilian, Rossolimo", ["e2e4", "c7c5", "g1f3", "b8c6", "f1b5", "g7g6"]),
    ("B33", "Sicilian, Sveshnikov Setup", ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6"]),
    ("B34", "Sicilian, Sveshnikov", ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e5"]),
    ("B40", "Sicilian, French Variation", ["e2e4", "c7c5", "g1f3", "e7e6"]),
    ("B41", "Sicilian, Kan", ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4", "a7a6"]),
    ("B42", "Sicilian, Kan", ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4", "a7a6", "b1c3"]),
    ("B43", "Sicilian, Taimanov", ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4", "b8c6"]),
    ("B44", "Sicilian, Taimanov", ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4", "f3d4", "b8c6", "b1c3"]),
    ("B50", "Sicilian", ["e2e4", "c7c5", "g1f3", "d7d6"]),
    ("B51", "Sicilian, Moscow", ["e2e4", "c7c5", "g1f3", "d7d6", "f1b5"]),
    ("B52", "Sicilian, Moscow", ["e2e4", "c7c5", "g1f3", "d7d6", "f1b5", "c8d7"]),
    ("B53", "Sicilian, Chebanenko", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "d1d4"]),
    ("B54", "Sicilian", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]),
    ("B55", "Sicilian", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"]),
    ("B56", "Sicilian, Boleslavsky", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "b8d7"]),
    ("B57", "Sicilian, Boleslavsky", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "b8d7", "f1c4"]),
    ("B60", "Sicilian, Richter-Rauzer", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "b8c6", "c1g5"]),
    ("B61", "Sicilian, Richter-Rauzer", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "b8c6", "c1g5", "e7e6"]),
    ("B70", "Sicilian, Dragon", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"]),
    ("B80", "Sicilian, Najdorf", ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"]),

    # C00-C99: Open Games + French
    ("C00", "French Defense", ["e2e4", "e7e6"]),
    ("C01", "French, Exchange", ["e2e4", "e7e6", "d2d4", "d7d5", "e4d5", "e6d5"]),
    ("C02", "French, Advance", ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"]),
    ("C03", "French, Tarrasch", ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2"]),
    ("C04", "French, Tarrasch", ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "g8f6", "e4e5"]),
    ("C05", "French, Tarrasch", ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "g8f6", "e4e5", "f6d7"]),
    ("C10", "French, Rubinstein", ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3"]),
    ("C11", "French, Classical", ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"]),
    ("C12", "French, MacCutcheon", ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f8b4"]),
    ("C13", "French, Classical", ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7"]),
    ("C15", "French, Winawer", ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4"]),
    ("C16", "French, Winawer", ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5"]),
    ("C17", "French, Winawer", ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5"]),
    ("C18", "French, Winawer", ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5", "a2a3"]),
    ("C20", "King's Pawn Game", ["e2e4", "e7e5"]),
    ("C21", "Center Game", ["e2e4", "e7e5", "d2d4", "e5d4", "d1d4"]),
    ("C23", "Bishop's Opening", ["e2e4", "e7e5", "f1c4"]),
    ("C24", "Bishop's Opening", ["e2e4", "e7e5", "f1c4", "g8f6"]),
    ("C25", "Vienna Game", ["e2e4", "e7e5", "b1c3"]),
    ("C26", "Vienna Game", ["e2e4", "e7e5", "b1c3", "g8f6"]),
    ("C27", "Vienna Game", ["e2e4", "e7e5", "b1c3", "g8f6", "f1c4"]),
    ("C28", "Vienna Game", ["e2e4", "e7e5", "b1c3", "g8f6", "f1c4", "b8c6"]),
    ("C29", "Vienna Gambit", ["e2e4", "e7e5", "b1c3", "g8f6", "f2f4"]),
    ("C30", "King's Gambit", ["e2e4", "e7e5", "f2f4"]),
    ("C31", "King's Gambit Declined", ["e2e4", "e7e5", "f2f4", "d7d5"]),
    ("C32", "King's Gambit Declined", ["e2e4", "e7e5", "f2f4", "d7d5", "e4d5", "e5e4"]),
    ("C33", "King's Gambit Accepted", ["e2e4", "e7e5", "f2f4", "e5f4"]),
    ("C34", "King's Gambit Accepted", ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3"]),
    ("C35", "King's Gambit Accepted, Cunningham", ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "d7d5"]),
    ("C36", "King's Gambit Accepted", ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "d7d5", "e4d5", "f8d6"]),
    ("C37", "King's Gambit Accepted, Muzio Gambit", ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g8f6", "f1c4"]),
    ("C39", "King's Gambit Accepted, Kieseritzky", ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g8f6", "f1c4", "b8c6"]),
    ("C40", "King's Knight Opening", ["e2e4", "e7e5", "g1f3"]),
    ("C41", "Philidor Defense", ["e2e4", "e7e5", "g1f3", "d7d6"]),
    ("C42", "Petrov's Defense", ["e2e4", "e7e5", "g1f3", "g8f6"]),
    ("C43", "Petrov, Russian", ["e2e4", "e7e5", "g1f3", "g8f6", "d2d4"]),
    ("C44", "King's Knight, Scotch Setup", ["e2e4", "e7e5", "g1f3", "d7d5"]),
    ("C45", "Scotch Game", ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"]),
    ("C46", "Three Knights Opening", ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3"]),
    ("C47", "Four Knights Game", ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"]),
    ("C48", "Four Knights, Spanish", ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6", "f1b5"]),
    ("C49", "Four Knights, Symmetrical", ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6", "f1b5", "f8b4"]),
    ("C50", "Italian Game", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]),
    ("C51", "Evans Gambit", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4"]),
    ("C52", "Evans Gambit Declined", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4", "c5b6"]),
    ("C53", "Italian, Classical", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3"]),
    ("C54", "Italian, Classical", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6"]),
    ("C55", "Italian, Two Knights Defense", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]),
    ("C56", "Two Knights, Modern", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d4", "e5d4", "f3d4"]),
    ("C57", "Two Knights, Fried Liver", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d4", "e5d4", "d1d4"]),
    ("C58", "Two Knights, Polerio", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d4", "e5d4", "c1g5", "d7d5"]),
    ("C59", "Two Knights, Knight Defense", ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d4", "e5d4", "c1g5", "d7d5", "e4d5", "f6d5", "f3d4"]),
    ("C60", "Ruy Lopez", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]),
    ("C61", "Ruy Lopez, Bird's Defense", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"]),
    ("C62", "Ruy Lopez, Old Steinitz", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "d8e7"]),
    ("C63", "Ruy Lopez, Schliemann", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "f7f5"]),
    ("C64", "Ruy Lopez, Classical", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "f8c5"]),
    ("C65", "Ruy Lopez, Berlin Defense", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"]),
    ("C66", "Ruy Lopez, Berlin Defense", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "d2d4"]),
    ("C67", "Ruy Lopez, Berlin Defense", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1"]),
    ("C68", "Ruy Lopez, Exchange", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6"]),
    ("C70", "Ruy Lopez, Closed", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b1c3"]),
    ("C71", "Ruy Lopez, Modern Steinitz", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "d7d6"]),
    ("C72", "Ruy Lopez, Modern Steinitz", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "d7d6", "d2d4"]),
    ("C74", "Ruy Lopez, Deferred Steinitz", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "d7d6", "d2d4", "b7b5"]),
    ("C75", "Ruy Lopez, Deferred Steinitz", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "d7d6", "d2d4", "b7b5", "a4b3"]),
    ("C77", "Ruy Lopez", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]),
    ("C78", "Ruy Lopez, Archangelsk", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "d2d3"]),
    ("C80", "Ruy Lopez, Open", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "d2d4", "e5d4"]),
    ("C81", "Ruy Lopez, Open, Howell", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "d2d4", "e5d4", "f3d4"]),
    ("C82", "Ruy Lopez, Open", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "d2d4", "e5d4", "f3d4", "c6d4", "d1d4"]),
    ("C84", "Ruy Lopez, Closed", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "d2d3", "d7d6"]),
    ("C86", "Ruy Lopez, Closed, Worrall", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "d2d3", "d7d6", "c2c3"]),

    # D00-D99: Closed / Semi-Closed
    ("D00", "Queen's Pawn Opening", ["d2d4"]),
    ("D00", "Richter-Veresov Attack", ["d2d4", "g8f6", "c1g5"]),
    ("D00", "Blackmar-Diemer Gambit", ["d2d4", "d7d5", "e2e4", "d5e4", "b1c3"]),
    ("D01", "Richter-Veresov Attack", ["d2d4", "d7d5", "b1c3", "g8f6", "c1g5"]),
    ("D06", "Queen's Gambit", ["d2d4", "d7d5", "c2c4"]),
    ("D10", "Slav Defense", ["d2d4", "d7d5", "c2c4", "c7c6"]),
    ("D11", "Slav Defense", ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6"]),
    ("D12", "Slav Defense", ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "c8f5"]),
    ("D13", "Slav, Exchange", ["d2d4", "d7d5", "c2c4", "c7c6", "c4d5", "c6d5"]),
    ("D14", "Slav, Exchange", ["d2d4", "d7d5", "c2c4", "c7c6", "c4d5", "c6d5", "g1f3", "g8f6", "c1f4"]),
    ("D20", "Queen's Gambit Accepted", ["d2d4", "d7d5", "c2c4", "d5c4"]),
    ("D21", "QGA, Alekhine Defense", ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3"]),
    ("D22", "QGA, Alekhine Defense", ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "a7a6"]),
    ("D23", "QGA, Mannheim", ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "d1a4"]),
    ("D24", "QGA", ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "d1a4", "c7c6"]),
    ("D25", "QGA", ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3"]),
    ("D26", "QGA, Classical", ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6"]),
    ("D27", "QGA, Classical", ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6", "f1c4"]),
    ("D28", "QGA, Classical", ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6", "f1c4", "c7c5"]),
    ("D29", "QGA, Classical", ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6", "f1c4", "c7c5", "d1d3"]),
    ("D30", "QGD Lasker", ["d2d4", "d7d5", "c2c4", "e7e6", "g1f3"]),
    ("D31", "QGD, Semi-Slav Setup", ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3"]),
    ("D32", "Tarrasch Defense", ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c5"]),
    ("D33", "Tarrasch Defense", ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c5", "c1g5"]),
    ("D35", "QGD, Exchange", ["d2d4", "d7d5", "c2c4", "e7e6", "c4d5", "e6d5"]),
    ("D36", "QGD, Exchange", ["d2d4", "d7d5", "c2c4", "e7e6", "c4d5", "e6d5", "b1c3", "g8f6"]),
    ("D37", "QGD", ["d2d4", "d7d5", "c2c4", "e7e6", "c4d5", "e6d5", "b1c3", "g8f6", "g1f3"]),
    ("D40", "QGD, Semi-Tarrasch", ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3", "c7c5"]),
    ("D50", "QGD", ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5"]),
    ("D80", "Grünfeld Defense", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"]),
    ("D81", "Grünfeld, Russian", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "d1b3"]),
    ("D85", "Grünfeld, Exchange", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4"]),
    ("D90", "Grünfeld, Three Knights", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "g1f3"]),

    # E00-E99: Indian Defenses
    ("E00", "Catalan Opening", ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"]),
    ("E01", "Catalan, Closed", ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5"]),
    ("E02", "Catalan, Open", ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "c4d5", "e6d5"]),
    ("E03", "Catalan, Open", ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "c4d5", "e6d5", "f1g2"]),
    ("E04", "Catalan", ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "b8c6"]),
    ("E05", "Catalan, Classical", ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7"]),
    ("E10", "Queen's Indian Defense", ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"]),
    ("E12", "Queen's Indian Defense", ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "b1c3"]),
    ("E13", "QID, Petrosian", ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "b1c3", "c8b7", "c1e3"]),
    ("E14", "QID", ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "b1c3", "c8b7", "e2e3"]),
    ("E15", "QID", ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8b7"]),
    ("E16", "QID, Capablanca", ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3", "c8b7", "f1g2", "f8b4"]),
    ("E20", "Nimzo-Indian Defense", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]),
    ("E21", "Nimzo-Indian, Three Knights", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "g1f3"]),
    ("E22", "Nimzo-Indian, Spielmann", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1b3"]),
    ("E23", "Nimzo-Indian, Spielmann", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1b3", "b8c6"]),
    ("E30", "Nimzo-Indian, Leningrad", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "b2b3"]),
    ("E32", "Nimzo-Indian, Classical", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2"]),
    ("E40", "Nimzo-Indian", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3"]),
    ("E50", "Nimzo-Indian", ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "d7d5"]),
    ("E60", "King's Indian Defense", ["d2d4", "g8f6", "c2c4", "g7g6"]),
    ("E61", "King's Indian Defense", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3"]),
    ("E62", "King's Indian Defense", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"]),
    ("E70", "King's Indian Defense", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4"]),
    ("E76", "KID, Four Pawns Attack", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "f2f4"]),
    ("E80", "KID, Sämisch", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "f2f3"]),
    ("E90", "KID, Classical", ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3"]),
]


class OpeningBook:
    """Opening book for benchmark games."""

    def __init__(self):
        self.openings = ECO_OPENINGS
        self.opening_fens: list[dict[str, Any]] = []
        self._precompute_fens()

    def _precompute_fens(self):
        """Precompute FEN for each opening."""
        self.opening_fens = []
        for eco, name, moves in self.openings:
            board = chess.Board()
            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    break
            self.opening_fens.append({
                'eco': eco,
                'name': name,
                'moves': moves,
                'fen': board.fen(),
                'ply': len(moves)
            })

    def get_random_opening(self) -> dict[str, Any]:
        """Get a random opening."""
        return random.choice(self.opening_fens)

    def get_opening_by_eco(self, eco: str) -> dict[str, Any] | None:
        """Get opening by ECO code."""
        for op in self.opening_fens:
            if op['eco'] == eco:
                return op
        return None

    def get_all_openings(self) -> list[dict[str, Any]]:
        """Get all openings."""
        return self.opening_fens.copy()

    def get_openings_by_category(self, category_prefix: str) -> list[dict[str, Any]]:
        """Get openings by ECO category (e.g., 'A0' for A00-A09)."""
        return [op for op in self.opening_fens if op['eco'].startswith(category_prefix)]

    def get_balanced_set(self, n: int) -> list[dict]:
        """Get a balanced set of n openings across categories."""
        categories = defaultdict(list)
        for op in self.opening_fens:
            cat = op['eco'][0]  # A, B, C, D, E
            categories[cat].append(op)

        # Distribute evenly across categories
        result = []
        per_cat = max(1, n // len(categories))
        for _cat, ops in categories.items():
            random.shuffle(ops)
            result.extend(ops[:per_cat])

        # Fill remaining
        if len(result) < n:
            remaining = [op for op in self.opening_fens if op not in result]
            random.shuffle(remaining)
            result.extend(remaining[:n - len(result)])

        random.shuffle(result)
        return result[:n]


if __name__ == "__main__":
    book = OpeningBook()
    print(f"Total openings: {len(book.openings)}")

    # Show categories
    from collections import defaultdict
    cats: defaultdict[str, int] = defaultdict(int)
    for op in book.opening_fens:
        cats[op['eco'][0]] += 1
    for cat, count in sorted(cats.items()):
        print(f"  Category {cat}: {count} openings")

    # Show random opening
    op = book.get_random_opening()
    print(f"\nRandom: {op['eco']} - {op['name']}")
    print(f"  Moves: {op['moves']}")
    print(f"  FEN: {op['fen']}")
    print(f"  Ply: {op['ply']}")
