#!/usr/bin/env python3
"""Minimal UCI engine stub for StockfishProvider integration tests.

Speaks the real UCI protocol over stdin/stdout — **not** a unit-test mock.
The subprocess is launched by python-chess's `SimpleEngine.popen_uci(command)`,
which means the StockfishProvider exercises its real code path (subprocess
spawn, async `await asyncio.to_thread(engine.play, ...)`). This script
tracks the board state and plays the first legal move, so tests are
reproducible and valid for both colors.

Invoke as: ``python -m tests.fixtures.uci_stub_engine``
"""

import sys

import chess


def uci_loop() -> None:
    board = chess.Board()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = line.strip()
        if cmd == "uci":
            sys.stdout.write("id name StubUCI\n")
            sys.stdout.write("id author tests\n")
            sys.stdout.write("uciok\n")
            sys.stdout.flush()
        elif cmd == "isready":
            sys.stdout.write("readyok\n")
            sys.stdout.flush()
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd.startswith("position"):
            parts = cmd.split()
            if parts[1] == "startpos":
                board = chess.Board()
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                    for move_uci in parts[moves_idx + 1 :]:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
            elif parts[1] == "fen":
                # position fen <FEN> [moves ...]
                fen_parts = parts[2:8]  # FEN has 6 fields
                fen = " ".join(fen_parts)
                board = chess.Board(fen)
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                    for move_uci in parts[moves_idx + 1 :]:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
        elif cmd.startswith("go"):
            # Play the first legal move for the current position.
            legal_moves = list(board.legal_moves)
            if legal_moves:
                move = legal_moves[0]
                sys.stdout.write(f"bestmove {move.uci()}\n")
            else:
                sys.stdout.write("bestmove 0000\n")
            sys.stdout.flush()
        elif cmd == "quit":
            break


if __name__ == "__main__":
    uci_loop()
