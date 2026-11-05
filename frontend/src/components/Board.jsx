import { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import { BOARD_THEMES, uciSquares } from "../utils";
import { EvalBar } from "./ui";

const BRASS = "rgba(224,169,59,0.45)";

/**
 * A chessboard that sizes itself to its container.
 *  fen          position to show ("start" is allowed)
 *  interactive  allow the person to move pieces (drag, or click two squares)
 *  onMove(uci)  called with a move that is legal in `fen`; promotions become queens
 *  arrows       [{uci, color}] drawn over the board
 *  lastMove     uci of the last move, highlighted
 */
export default function Board({ fen, orientation = "white", interactive = false, onMove, arrows = [], lastMove, theme = "classic", id = "board" }) {
  const box = useRef(null);
  const [width, setWidth] = useState(480);
  const [picked, setPicked] = useState(null);

  useEffect(() => {
    if (!box.current) return undefined;
    const ro = new ResizeObserver(([e]) => setWidth(Math.floor(e.contentRect.width)));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  useEffect(() => setPicked(null), [fen]);

  const colors = BOARD_THEMES[theme] ?? BOARD_THEMES.classic;
  const full = fen === "start" ? undefined : fen;

  const tryMove = (from, to) => {
    if (!interactive || !full) return false;
    try {
      const m = new Chess(full).move({ from, to, promotion: "q" });
      onMove?.(`${from}${to}${m.promotion || ""}`);
      return true;
    } catch { return false; }
  };

  const squareStyles = useMemo(() => {
    const s = {};
    if (lastMove) uciSquares(lastMove).forEach((q) => { s[q] = { background: BRASS }; });
    if (picked && interactive && full) {
      s[picked] = { background: "rgba(109,179,212,0.55)" };
      try {
        new Chess(full).moves({ square: picked, verbose: true }).forEach((m) => {
          s[m.to] = { ...s[m.to], background: "radial-gradient(circle, rgba(15,26,28,.45) 22%, transparent 24%)" };
        });
      } catch { /* empty square */ }
    }
    return s;
  }, [lastMove, picked, full, interactive]);

  const onSquareClick = (sq) => {
    if (!interactive || !full) return;
    if (picked && picked !== sq && tryMove(picked, sq)) { setPicked(null); return; }
    try {
      const c = new Chess(full);
      const p = c.get(sq);
      setPicked(p && p.color === c.turn() ? sq : null);
    } catch { setPicked(null); }
  };

  return (
    <div className="board-box" ref={box}>
      <Chessboard
        id={id}
        position={fen}
        boardWidth={width}
        boardOrientation={orientation}
        arePiecesDraggable={interactive}
        onPieceDrop={(from, to) => tryMove(from, to)}
        onSquareClick={onSquareClick}
        customSquareStyles={squareStyles}
        customArrows={arrows.map((a) => [...uciSquares(a.uci), a.color || "#e0a93b"])}
        customDarkSquareStyle={{ backgroundColor: colors.dark }}
        customLightSquareStyle={{ backgroundColor: colors.light }}
        animationDuration={160}
      />
    </div>
  );
}

/**
 * Board with the evaluation bar beside it, same height.
 *  fit  fill the parent (which must be position: relative with a definite size) and
 *       use the largest square that fits, so the page never needs to scroll.
 */
export function BoardWithEval({ showEval = true, evalValue, orientation, fit = false, ...rest }) {
  const holder = useRef(null);
  const area = useRef(null);
  const [h, setH] = useState(undefined);
  const [side, setSide] = useState(null);

  useEffect(() => {
    if (fit || !holder.current) return undefined;
    const ro = new ResizeObserver(([e]) => setH(Math.floor(e.contentRect.height)));
    ro.observe(holder.current);
    return () => ro.disconnect();
  }, [fit]);

  useEffect(() => {
    if (!fit || !area.current) return undefined;
    const barSpace = showEval ? 26 + 12 : 0;
    const ro = new ResizeObserver(([e]) => {
      const { width, height } = e.contentRect;
      setSide(Math.max(160, Math.floor(Math.min(width - barSpace, height))));
    });
    ro.observe(area.current);
    return () => ro.disconnect();
  }, [fit, showEval]);

  if (fit) {
    return (
      <div className="board-wrap fit" ref={area}>
        {side && (
          <>
            {showEval && <EvalBar height={side} value={evalValue} orientation={orientation} />}
            <div style={{ width: side, height: side }}>
              <Board orientation={orientation} {...rest} />
            </div>
          </>
        )}
      </div>
    );
  }
  return (
    <div className="board-wrap">
      {showEval && <EvalBar height={h} value={evalValue} orientation={orientation} />}
      <div ref={holder} style={{ width: "100%", maxWidth: 640 }}>
        <Board orientation={orientation} {...rest} />
      </div>
    </div>
  );
}
