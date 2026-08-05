/**
 * Chessboard.js Streamlit Component
 * Interactive chess board with drag-and-drop, animations, and coordinates
 */

// Chessboard component for Streamlit
class ChessboardComponent {
    constructor() {
        this.board = null;
        this.game = null;
        this.config = {
            position: 'start',
            orientation: 'white',
            showCoordinates: true,
            draggable: true,
            animations: true,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
        };
        this.onMove = null;
        this.onSnapEnd = null;
    }

    init(container, config = {}) {
        this.config = { ...this.config, ...config };
        
        // Create chessboard.js board
        this.board = Chessboard(container, {
            position: this.config.position,
            orientation: this.config.orientation,
            showCoordinates: this.config.showCoordinates,
            draggable: this.config.draggable,
            pieceTheme: this.config.pieceTheme,
            onDrop: this.handleDrop.bind(this),
            onSnapEnd: this.handleSnapEnd.bind(this),
            onMoveEnd: this.handleMoveEnd.bind(this),
        });

        // Add animation CSS
        this.injectStyles();
    }

    injectStyles() {
        if (document.getElementById('chessboard-animations')) return;
        
        const style = document.createElement('style');
        style.id = 'chessboard-animations';
        style.textContent = `
            .chessboard-piece {
                transition: transform 0.2s ease-out, opacity 0.1s ease-out !important;
            }
            .chessboard-piece.animating {
                transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
            }
            .square-55d63.highlight-last-move {
                box-shadow: inset 0 0 0 4px #ffcc00 !important;
            }
            .square-55d63.highlight-check {
                box-shadow: inset 0 0 0 4px #ff3333 !important;
            }
            .square-55d63.highlight-legal {
                background: rgba(0, 200, 0, 0.15) !important;
            }
            .chessboard-squares .square-55d63 {
                transition: background-color 0.1s ease, box-shadow 0.1s ease;
            }
        `;
        document.head.appendChild(style);
    }

    handleDrop(source, target, piece, newPos, oldPos, orientation) {
        // Validate move
        const move = source + target;
        
        // Check if move is legal (basic UCI validation)
        if (!this.isValidUCI(move)) {
            return 'snapback';
        }

        // Call custom move handler
        if (this.onMove) {
            const result = this.onMove(move, source, target, piece);
            if (result === false) {
                return 'snapback';
            }
        }

        return move;
    }

    handleSnapEnd() {
        if (this.onSnapEnd) {
            this.onSnapEnd(this.board.position());
        }
    }

    handleMoveEnd(oldPos, newPos) {
        // Highlight last move
        this.highlightLastMove(oldPos, newPos);
    }

    highlightLastMove(from, to) {
        // Remove previous highlights
        document.querySelectorAll('.square-55d63.highlight-last-move').forEach(el => {
            el.classList.remove('highlight-last-move');
        });
        
        // Add new highlights
        const fromEl = document.querySelector(`.square-${from}`);
        const toEl = document.querySelector(`.square-${to}`);
        
        if (fromEl) fromEl.classList.add('highlight-last-move');
        if (toEl) toEl.classList.add('highlight-last-move');
    }

    highlightCheck(square) {
        const el = document.querySelector(`.square-${square}`);
        if (el) el.classList.add('highlight-check');
    }

    highlightLegalMoves(squares) {
        squares.forEach(sq => {
            const el = document.querySelector(`.square-${sq}`);
            if (el) el.classList.add('highlight-legal');
        });
    }

    clearHighlights() {
        document.querySelectorAll('.square-55d63.highlight-last-move, .square-55d63.highlight-check, .square-55d63.highlight-legal').forEach(el => {
            el.classList.remove('highlight-last-move', 'highlight-check', 'highlight-legal');
        });
    }

    isValidUCI(move) {
        // Basic UCI format: e2e4, e7e8q
        return /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(move);
    }

    setPosition(fen, animate = true) {
        this.board.position(fen, animate);
    }

    getPosition() {
        return this.board.position();
    }

    flip() {
        this.board.flip();
    }

    resize() {
        this.board.resize();
    }

    destroy() {
        this.board.destroy();
    }

    // Animation helpers
    animateMove(from, to, piece) {
        const pieceEl = document.querySelector(`.square-${from} .chessboard-piece`);
        if (!pieceEl) return Promise.resolve();

        const fromSquare = document.querySelector(`.square-${from}`);
        const toSquare = document.querySelector(`.square-${to}`);
        
        if (!fromSquare || !toSquare) return Promise.resolve();

        const fromRect = fromSquare.getBoundingClientRect();
        const toRect = toSquare.getBoundingClientRect();

        const dx = toRect.left - fromRect.left;
        const dy = toRect.top - fromRect.top;

        pieceEl.classList.add('animating');
        pieceEl.style.transform = `translate(${dx}px, ${dy}px)`;

        return new Promise(resolve => {
            setTimeout(() => {
                pieceEl.classList.remove('animating');
                pieceEl.style.transform = '';
                resolve();
            }, 300);
        });
    }
}

// Export for Streamlit
window.ChessboardComponent = ChessboardComponent;

// Streamlit component interface
function createChessboardComponent() {
    return {
        name: 'chessboard',
        render: function(container, props) {
            if (!window.chessboardInstance) {
                window.chessboardInstance = new ChessboardComponent();
                window.chessboardInstance.init(container, props.config);
                
                // Set up callbacks
                window.chessboardInstance.onMove = function(move, source, target, piece) {
                    if (props.onMove) {
                        props.onMove(move, source, target, piece);
                    }
                };
                
                window.chessboardInstance.onSnapEnd = function(position) {
                    if (props.onSnapEnd) {
                        props.onSnapEnd(position);
                    }
                };
            } else {
                // Update config if changed
                if (props.config) {
                    if (props.config.position) {
                        window.chessboardInstance.setPosition(props.config.position, props.config.animate !== false);
                    }
                    if (props.config.orientation) {
                        window.chessboardInstance.board.orientation(props.config.orientation);
                    }
                }
            }

            // Handle programmatic moves
            if (props.lastMove) {
                const [from, to] = [props.lastMove.slice(0, 2), props.lastMove.slice(2, 4)];
                window.chessboardInstance.highlightLastMove(from, to);
            }

            if (props.checkSquare) {
                window.chessboardInstance.highlightCheck(props.checkSquare);
            }

            if (props.legalMoves) {
                window.chessboardInstance.highlightLegalMoves(props.legalMoves);
            }

            if (props.clearHighlights) {
                window.chessboardInstance.clearHighlights();
            }

            return container;
        },
        dispose: function() {
            if (window.chessboardInstance) {
                window.chessboardInstance.destroy();
                window.chessboardInstance = null;
            }
        }
    };
}

// Register with Streamlit if available
if (typeof Streamlit !== 'undefined') {
    Streamlit.components.v1.registerComponent('chessboard', createChessboardComponent());
}