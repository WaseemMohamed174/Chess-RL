import { Chess } from "chess.js";

/** Same arithmetic as backend/rl/arch.py, so the Models page can preview a size before creating it. */
export function countParams(blocks, f) {
  const stem = 18 * 9 * f + 2 * f;
  const block = 2 * (9 * f * f) + 4 * f;
  const policy = f * 32 + 64 + 32 * 64 * 4672 + 4672;
  const value = f + 2 + 64 * 256 + 256 + 256 + 1;
  return stem + blocks * block + policy + value;
}

export const fmtInt = (n) => (n == null ? "-" : Math.round(n).toLocaleString("en-US"));
export const fmtParams = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : `${(n / 1e3).toFixed(0)}K`);
export const fmtEval = (v) => (v == null ? "0.00" : `${v > 0 ? "+" : ""}${v.toFixed(2)}`);
export const fmtDate = (iso) => (iso ? new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "-");

export const BOARD_THEMES = {
  classic: { name: "Classic", light: "#e9dcc0", dark: "#a67c52" },
  green: { name: "Green", light: "#eeeed2", dark: "#6a9a5b" },
  slate: { name: "Slate", light: "#c9d3d6", dark: "#5f7a86" },
  midnight: { name: "Midnight", light: "#8d9aa6", dark: "#3d4f63" },
};

export function download(filename, text, mime = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** UCI "e2e4" -> ["e2","e4"] */
export const uciSquares = (uci) => [uci.slice(0, 2), uci.slice(2, 4)];

/** The position the engine searched before it played the last move in `moves` (a list of UCI). */
export function fenBeforeLast(moves) {
  try {
    const c = new Chess();
    moves.slice(0, -1).forEach((m) => c.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] }));
    return c.fen();
  } catch { return null; }
}

/** Position after the first `ply` moves of `moves` (a list of UCI). ply 0 = the start position. */
export function fenAtPly(moves, ply) {
  try {
    const c = new Chess();
    moves.slice(0, ply).forEach((m) => c.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] }));
    return c.fen();
  } catch { return null; }
}

/** 3.2s, 45s, 1m 05s */
export function fmtSecs(s) {
  if (s == null) return "";
  if (s < 10) return `${s.toFixed(1)}s`;
  if (s < 60) return `${Math.round(s)}s`;
  return `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s`;
}
