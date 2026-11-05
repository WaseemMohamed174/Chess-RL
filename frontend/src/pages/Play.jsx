import { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { useSearchParams } from "react-router-dom";
import { Download, Flag, Lightbulb, Plus, Undo2 } from "lucide-react";
import { api } from "../api";
import PageHead from "../components/PageHead";
import { BoardWithEval } from "../components/Board";
import { Clocks, KingDead } from "../components/Extras";
import { Empty, ModelSelect, MoveList, PlayerCard, Segmented, SectionTitle, Thoughts, Tip } from "../components/ui";
import { useApp, useToast } from "../context";
import { useI18n } from "../i18n";
import { play, playSan } from "../sound";
import { download, fenAtPly, fenBeforeLast } from "../utils";

const step = (c, u) => c.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] });

export default function Play() {
  const { t } = useI18n();
  const toast = useToast();
  const { models, ui } = useApp();
  const [sp] = useSearchParams();
  const [modelId, setModelId] = useState(() => Number(sp.get("model")) || null);   // ?model=ID comes from the Models page
  const [color, setColor] = useState("white");
  const [game, setGame] = useState(null);             // latest state from the server
  const [pendingFen, setPendingFen] = useState(null); // your move, shown while the model thinks
  const [busy, setBusy] = useState(false);
  const [thoughts, setThoughts] = useState(null);     // {fen, top}
  const [hint, setHint] = useState(null);             // uci of the suggested move
  const [yourColor, setYourColor] = useState("white");
  const [closedKing, setClosedKing] = useState(null);  // game id whose "king is dead" card was closed
  const [viewPly, setViewPly] = useState(null);     // index of the move being looked at; null = following the live game
  const sounded = useRef(0);                      // how many moves of this game already made a sound

  const selected = modelId ?? models[0]?.id ?? null;
  const opponentId = game ? (game.white ?? game.black) : selected;
  const opponent = models.find((m) => m.id === opponentId);

  /** Take a new server state; keep what the model was thinking when it just moved. */
  const adopt = (g, { clear = false } = {}) => {
    if (g.san.length > sounded.current) playSan(g.san[g.san.length - 1]);   // the model's move
    sounded.current = g.san.length;
    setGame(g);
    setViewPly(null);
    setPendingFen(null);
    setHint(null);
    const fen = g.top?.length && g.moves.length ? fenBeforeLast(g.moves) : null;
    if (fen && g.human_to_move) setThoughts({ fen, top: g.top });
    else if (clear || !g.human_to_move) setThoughts(null);
  };

  const start = async () => {
    const mine = color === "random" ? (Math.random() < 0.5 ? "white" : "black") : color;
    setBusy(true);
    try {
      const g = await api("/api/games/new", { method: "POST", body: { model_id: selected, color: mine } });
      setYourColor(mine);
      sounded.current = 0;
      play("start");
      adopt(g, { clear: true });
    } catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };

  const onMove = async (uci) => {
    const c = new Chess(game.fen);
    const mv = step(c, uci);
    playSan(mv.san);
    sounded.current = game.san.length + 1;
    setPendingFen(c.fen());
    setBusy(true);
    try {
      adopt(await api("/api/games/play", { method: "POST", body: { game_id: game.game_id, move_uci: uci } }));
    } catch (e) {
      setPendingFen(null);
      toast(e.message, "error");
    } finally { setBusy(false); }
  };

  const call = async (action) => {
    setBusy(true);
    try {
      adopt(await api(`/api/games/${game.game_id}/${action}`, { method: "POST" }), { clear: action === "undo" });
    } catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };

  const askHint = async () => {
    setBusy(true);
    try {
      const h = await api(`/api/games/${game.game_id}/hint`, { method: "POST" });
      setHint(h.move);
      setThoughts({ fen: game.fen, top: h.top });
    } catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };

  const exportPgn = async () => {
    try { download(`game-${game.game_id}.pgn`, await api(`/api/games/${game.game_id}/pgn`)); }
    catch (e) { toast(e.message, "error"); }
  };

  const reviewing = !!game && viewPly != null && viewPly < game.moves.length - 1;
  const reviewFen = reviewing ? fenAtPly(game.moves, viewPly + 1) : null;
  const shownFen = reviewFen ?? pendingFen ?? game?.fen ?? "start";
  const shownLast = reviewing ? game.moves[viewPly] : game?.last_move;

  const pick = (i) => {
    if (!game) return;
    if (i >= game.moves.length - 1) { setViewPly(null); return; }
    setViewPly(i);
    if (game.san[i]) playSan(game.san[i]);
  };
  // Left / right arrow keys walk through the moves, Esc returns to the live position.
  useEffect(() => {
    const onKey = (e) => {
      if (!game || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
      const cur = viewPly ?? game.moves.length - 1;
      if (e.key === "ArrowLeft") { e.preventDefault(); pick(Math.max(0, cur - 1)); }
      else if (e.key === "ArrowRight") { e.preventDefault(); pick(Math.min(game.moves.length - 1, cur + 1)); }
      else if (e.key === "Escape") setViewPly(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // Green arrow = hint for you. Brass arrow = the move the model just played.
  const arrows = useMemo(() => {
    if (reviewing || !ui.show_arrows || !game || game.over) return [];
    if (hint) return [{ uci: hint, color: "#5fbf8f" }];
    if (game.last_move && game.human_to_move) return [{ uci: game.last_move, color: "#e0a93b" }];
    return [];
  }, [ui.show_arrows, hint, game, reviewing]);

  // Undo would leave a black player with nobody to move, so it needs a move of yours to take back.
  const canUndo = game && !game.over && (yourColor === "white" ? game.moves.length > 0 : game.moves.length > 1);

  const mated = !!game?.over && !!game.san.length && game.san[game.san.length - 1].endsWith("#");
  const loser = game?.result === "1-0" ? "black" : "white";
  const youWon = game?.over && game.result !== "1/2-1/2" && ((game.result === "1-0") === (yourColor === "white"));
  const outcome = !game || !game.over ? null
    : game.result === "1/2-1/2" ? ["", t("Draw")] : youWon ? ["win", t("You won")] : ["loss", t("You lost")];
  const modelColor = yourColor === "white" ? "black" : "white";
  const modelTurn = !!game && !game.over && (game.turn === modelColor || !!pendingFen);
  const myTurn = !!game && !game.over && game.turn === yourColor && !pendingFen;

  return (
    <>
      <PageHead title="Play" subtitle="Play a game against one of your trained models. It searches ahead before every move; the bar beside the board shows who it thinks is winning." />
      <div className="fit-grid">
        <div className="sidecol">
          {game && <Clocks gameKey={game.game_id} moveKey={`${game.san.length}-${pendingFen ? 1 : 0}`} running={!game.over} />}

          {outcome && (
            <div className={`banner ${outcome[0]}`} aria-live="polite">
              <strong>{outcome[1]}</strong>
              <span className="muted">{game.result}</span>
            </div>
          )}

          {game && (
            <div className="stack" style={{ gap: 6 }}>
              <PlayerCard color={modelColor} name={opponent?.name ?? "Model"} kind={t("Model")}
                detail={`${Math.round(opponent?.elo ?? 0)} Elo`} active={modelTurn} turnText={t("Thinking...")} />
              <PlayerCard color={yourColor} name={t("Player")} kind={t("Human")} active={myTurn} turnText={t("Your turn")} />
            </div>
          )}

          {(!game || game.over) && (
            <section className="card pad-lg stack">
              <SectionTitle help="Pick which trained model to play against and which side you want. Elo is its strength: higher is stronger.">{t("New game")}</SectionTitle>
              {models.length === 0 ? (
                <Empty title="No models yet">
                  <span>{t("The first model is created when the engine finishes starting. You can also make one on the Models page.")}</span>
                </Empty>
              ) : (
                <>
                  <ModelSelect id="opp" label={t("Opponent")} models={models} value={selected} onChange={setModelId} />
                  <div className="field">
                    <span className="label">{t("Your color")}</span>
                    <Segmented label={t("Your color")} value={color} onChange={setColor}
                      options={[{ value: "white", label: t("White") }, { value: "black", label: t("Black") }, { value: "random", label: t("Random") }]} />
                  </div>
                  <button className="btn primary" onClick={start} disabled={busy || !selected}><Plus size={17} />{t("Start game")}</button>
                </>
              )}
            </section>
          )}

          {game && (
            <section className="card grow">
              <SectionTitle help="Click a move to see the board at that moment. Small numbers are how long each move took."
                right={<Tip text="Download this game as a PGN file, which any chess program can open."><button className="btn small ghost" onClick={exportPgn}><Download size={14} />{t("Export PGN")}</button></Tip>}>
                {t("Moves")}
              </SectionTitle>
              {reviewing && (
                <div className="review-bar">
                  <span>{t("Viewing an earlier move")}</span>
                  <button className="btn small" onClick={() => setViewPly(null)}>{t("Back to current move")}</button>
                </div>
              )}
              <MoveList san={game.san} secs={game.move_secs} selected={viewPly} onSelect={pick} fill />
            </section>
          )}

          {game && !game.over && (
            <div className="btn-row" style={{ flexWrap: "nowrap" }}>
              <Tip text="Take back your last move (and the model's reply)." side="top">
                <button className="btn" onClick={() => call("undo")} disabled={busy || !canUndo}><Undo2 size={16} />{t("Undo")}</button>
              </Tip>
              <Tip text="Ask the model what it would play for you. It is drawn as a green arrow." side="top">
                <button className="btn" onClick={askHint} disabled={busy || !game.human_to_move}><Lightbulb size={16} />{t("Hint")}</button>
              </Tip>
              <Tip text="Give up this game. It counts as a loss." side="top" align="end">
                <button className="btn danger" onClick={() => call("resign")} disabled={busy}><Flag size={16} />{t("Resign")}</button>
              </Tip>
            </div>
          )}

          {game && (
            <section className="card">
              <SectionTitle help="The model's favourite moves in the position before its last move. A longer bar means it spent more of its thinking on that move.">{t("What the model sees")}</SectionTitle>
              <Thoughts fen={thoughts?.fen} top={thoughts?.top} limit={4} />
            </section>
          )}
        </div>

        <div className="boardcol">
          <BoardWithEval fit
            fen={shownFen} orientation={yourColor} theme={ui.board_theme}
            interactive={!!game && !game.over && game.human_to_move && !busy && !reviewing}
            onMove={onMove} lastMove={shownLast} arrows={arrows}
            showEval={ui.show_eval} evalValue={game?.eval ?? 0}
          />
          {mated && closedKing !== game.game_id && <KingDead loser={loser} onClose={() => setClosedKing(game.game_id)} />}
        </div>
      </div>
    </>
  );
}
