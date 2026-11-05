import { useEffect, useMemo, useRef, useState } from "react";
import { Play as PlayIcon, Plus, Square } from "lucide-react";
import { api, useSocket } from "../api";
import PageHead from "../components/PageHead";
import { BoardWithEval } from "../components/Board";
import { Clocks, KingDead } from "../components/Extras";
import { Empty, InfoTip, ModelSelect, MoveList, PlayerCard, SectionTitle, Stat, Thoughts } from "../components/ui";
import { useApp, useToast } from "../context";
import { useI18n } from "../i18n";
import { play, playSan, unlockSound } from "../sound";
import { fenAtPly, fenBeforeLast } from "../utils";

const delayFor = (speed) => Math.round(2000 * ((100 - speed) / 100)); // speed 0..100 -> 2000..0 ms

export default function Spectate() {
  const { t } = useI18n();
  const toast = useToast();
  const { models, loadModels, ui } = useApp();
  const [aId, setA] = useState(null);
  const [bId, setB] = useState(null);
  const [games, setGames] = useState(6);
  const [sims, setSims] = useState(100);
  const [speed, setSpeed] = useState(80);
  const [busy, setBusy] = useState(false);

  const [match, setMatch] = useState(null);     // {id, a, b, aId, bId, total}
  const [running, setRunning] = useState(false);
  const [score, setScore] = useState({ a: 0, b: 0, d: 0 });
  const [index, setIndex] = useState(0);
  const [live, setLive] = useState(null);       // latest game state
  const [final, setFinal] = useState(null);     // match_end payload
  const [viewPly, setViewPly] = useState(null); // move being looked at; null = following the live game
  const [setupOpen, setSetupOpen] = useState(false);
  const [closedKing, setClosedKing] = useState(null);
  const speedTimer = useRef(null);
  const sounded = useRef(0);

  const a = models.find((m) => m.id === (aId ?? models[0]?.id));
  const b = models.find((m) => m.id === (bId ?? models.find((m2) => m2.id !== a?.id)?.id));
  const byId = (id) => models.find((m) => m.id === id);
  const nameOf = (id) => byId(id)?.name ?? "?";

  useSocket(match ? `/ws/match/${match.id}` : null, (m) => {
    switch (m.type) {
      case "hello":
        if (m.match) setScore({ a: m.match.white_wins, b: m.match.black_wins, d: m.match.draws });
        break;
      case "game_start":
        setIndex(m.index);
        setLive(null);
        setViewPly(null);
        sounded.current = 0;
        play("start");
        break;
      case "move":
        if (m.san && m.san.length > sounded.current) playSan(m.san[m.san.length - 1]);
        sounded.current = m.san ? m.san.length : sounded.current;
        setLive(m);
        if (m.index != null) setIndex(m.index);
        break;
      case "game_end":
        setScore({ a: m.a_wins, b: m.b_wins, d: m.draws });
        break;
      case "match_end":
        setScore({ a: m.a_wins, b: m.b_wins, d: m.draws });
        setFinal(m);
        setRunning(false);
        loadModels();
        break;
      case "match_cancelled":
        setRunning(false);
        loadModels();
        break;
      case "match_error":
        setRunning(false);
        toast(m.error, "error");
        break;
      default:
    }
  });

  useEffect(() => () => clearTimeout(speedTimer.current), []);

  const start = async () => {
    unlockSound();
    setBusy(true);
    try {
      const row = await api("/api/arena/match", {
        method: "POST",
        body: { white_id: a.id, black_id: b.id, games, simulations: sims, delay_ms: delayFor(speed) },
      });
      setMatch({ id: row.id, a: a.name, b: b.name, aId: a.id, bId: b.id, total: games, startedAt: Date.now() });
      setScore({ a: 0, b: 0, d: 0 });
      setIndex(0);
      setLive(null);
      setFinal(null);
      setViewPly(null);
      setSetupOpen(false);
      setRunning(true);
    } catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };

  const stop = async () => {
    try { await api(`/api/arena/match/${match.id}/cancel`, { method: "POST" }); }
    catch (e) { toast(e.message, "error"); }
  };

  const changeSpeed = (v) => {
    setSpeed(v);
    if (!running || !match) return;
    clearTimeout(speedTimer.current);
    speedTimer.current = setTimeout(() => {
      api(`/api/arena/match/${match.id}/speed`, { method: "PUT", body: { delay_ms: delayFor(v) } }).catch(() => {});
    }, 250);
  };

  // ---- looking back at an earlier move (like chess.com) ----
  const lastIdx = (live?.moves?.length ?? 0) - 1;
  const reviewing = !!live && viewPly != null && viewPly < lastIdx;
  const shownFen = reviewing ? fenAtPly(live.moves, viewPly + 1) : live?.fen ?? "start";
  const shownLast = reviewing ? live.moves[viewPly] : live?.last_move;
  const pick = (i) => {
    if (!live) return;
    if (i >= lastIdx) { setViewPly(null); return; }
    setViewPly(i);
    if (live.san[i]) playSan(live.san[i]);
  };
  useEffect(() => {
    const onKey = (e) => {
      if (!live || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
      const cur = viewPly ?? lastIdx;
      if (e.key === "ArrowLeft") { e.preventDefault(); pick(Math.max(0, cur - 1)); }
      else if (e.key === "ArrowRight") { e.preventDefault(); pick(Math.min(lastIdx, cur + 1)); }
      else if (e.key === "Escape") setViewPly(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const fenThen = live ? fenBeforeLast(live.moves) : null;
  const total = score.a + score.b + score.d;
  const mated = !!live?.over && !!live.san?.length && live.san[live.san.length - 1].endsWith("#");
  const arrows = useMemo(() => (
    !reviewing && ui.show_arrows && live?.last_move && !live.over ? [{ uci: live.last_move, color: "#e0a93b" }] : []
  ), [reviewing, ui.show_arrows, live]);

  // Who plays which colour in the game on the board, and how it ended.
  const whiteM = live ? byId(live.white) : null;
  const blackM = live ? byId(live.black) : null;
  const tagOf = (id) => (match && id === match.aId ? t("Model A") : t("Model B"));
  const gameOutcome = !live?.over ? null
    : live.result === "1/2-1/2" ? ["", t("Draw")]
    : ["win", `${nameOf(live.result === "1-0" ? live.white : live.black)} ${t("won")}`];

  const showSetup = !match || (!running && setupOpen);
  const showWatch = !!match && !showSetup;
  const aLead = score.a > score.b, bLead = score.b > score.a;

  const setup = (
    <section className="card pad-lg stack">
      <SectionTitle help={t("Pick two models and how many games they play. Colors swap every game, so neither model gets an unfair advantage.")}>
        {t("New match")}
      </SectionTitle>
      <ModelSelect id="ma" label={t("Model A")} models={models} value={a?.id ?? null} onChange={setA} exclude={b?.id} />
      <ModelSelect id="mb" label={t("Model B")} models={models} value={b?.id ?? null} onChange={setB} exclude={a?.id} />
      <div className="grid2">
        <div className="field">
          <label htmlFor="g">{t("Games")}<InfoTip text={t("How many games the two models play against each other (1 to 200).")} /></label>
          <input id="g" type="number" min="1" max="200" value={games} onChange={(e) => setGames(Math.max(1, Math.min(200, Number(e.target.value) || 1)))} />
        </div>
        <div className="field">
          <label htmlFor="s">{t("Simulations")}<InfoTip text={t("How many positions each model searches before every move. More means stronger and slower play (10 to 800).")} /></label>
          <input id="s" type="number" min="10" max="800" step="10" value={sims} onChange={(e) => setSims(Math.max(10, Math.min(800, Number(e.target.value) || 10)))} />
        </div>
      </div>
      <div className="field">
        <label htmlFor="speed2">{t("Speed")}<InfoTip text={t("How fast moves appear on the board. You can change it while the match is running.")} /></label>
        <input id="speed2" type="range" min="0" max="100" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
        <div className="rangeval muted"><span>{t("Slower")}</span><span>{t("Faster")}</span></div>
      </div>
      <button className="btn primary" onClick={start} disabled={busy || !a || !b}><PlayIcon size={16} />{t("Start match")}</button>
    </section>
  );

  return (
    <>
      <PageHead title="Spectate" subtitle="Pick two models and watch them play each other. Colors swap every game, so neither one has an advantage." />
      {models.length < 2 ? (
        <div className="dock"><Empty title={t("You need two models to start a match")}>
          <span>{t("Create another model on the Models page, then come back.")}</span>
        </Empty></div>
      ) : (
        <div className="fit-grid">
          <div className="sidecol">
            {match && <Clocks gameKey={`${match.id}-${index}`} moveKey={live?.moves?.length ?? 0} sessionKey={match.id} running={running} />}

            {match && (
              <div className={`banner ${final ? "win" : ""}`} aria-live="polite">
                <strong>{match.a} {t("vs")} {match.b}</strong>
                <span className="muted">
                  {final ? t("Match finished") : running ? `${t("Game")} ${Math.min(index + 1, match.total)} / ${match.total}` : t("Match stopped")}
                </span>
              </div>
            )}

            {live && !showSetup && (
              <div className="stack" style={{ gap: 6 }}>
                <PlayerCard color="black" name={blackM?.name ?? "?"} kind={tagOf(live.black)} detail={`${Math.round(blackM?.elo ?? 0)} Elo`}
                  active={!live.over && live.turn === "black"} turnText={t("Thinking...")} />
                <PlayerCard color="white" name={whiteM?.name ?? "?"} kind={tagOf(live.white)} detail={`${Math.round(whiteM?.elo ?? 0)} Elo`}
                  active={!live.over && live.turn === "white"} turnText={t("Thinking...")} />
              </div>
            )}

            {gameOutcome && !showSetup && (
              <div className={`banner ${gameOutcome[0]}`}><strong>{gameOutcome[1]}</strong><span className="muted">{live.result}</span></div>
            )}

            {match && (
              <section className="card">
                <SectionTitle help={t("Wins are counted per model, not per color. Each model plays White in half of the games.")}>{t("Score")}</SectionTitle>
                <div className="stats">
                  <Stat label={match.a} value={score.a} />
                  <Stat label={t("Draws")} value={score.d} hint={t("Games nobody won.")} />
                  <Stat label={match.b} value={score.b} />
                </div>
                {total > 0 && (
                  <div className="share" style={{ marginTop: 8 }} title={`${score.a} - ${score.d} - ${score.b}`}><i style={{ width: `${(score.a / total) * 100}%` }} /></div>
                )}
                {final && (
                  <p className="help" style={{ marginTop: 8 }}>
                    {aLead ? `${match.a} ${t("won the match")}.` : bLead ? `${match.b} ${t("won the match")}.` : `${t("The match is a draw")}.`}{" "}
                    Elo: {match.a} {Math.round(final.elo_a)} · {match.b} {Math.round(final.elo_b)}
                  </p>
                )}
              </section>
            )}

            {running && (
              <section className="card stack" style={{ gap: 8 }}>
                <div className="field">
                  <label htmlFor="speed">{t("Speed")}<InfoTip side="top" text={t("How fast moves appear on the board. You can change it while the match is running.")} /></label>
                  <input id="speed" type="range" min="0" max="100" value={speed} onChange={(e) => changeSpeed(Number(e.target.value))} />
                  <div className="rangeval muted"><span>{t("Slower")}</span><span>{t("Faster")}</span></div>
                </div>
                <button className="btn danger" onClick={stop}><Square size={16} />{t("Stop match")}</button>
              </section>
            )}

            {!running && match && !showSetup && (
              <button className="btn primary" onClick={() => setSetupOpen(true)}><Plus size={16} />{t("New match")}</button>
            )}

            {showSetup && setup}
            {showSetup && match && (
              <button className="btn ghost" onClick={() => setSetupOpen(false)}>{t("Back to the finished match")}</button>
            )}

            {showWatch && live && (
              <section className="card grow">
                <SectionTitle help={t("Click a move to see the board at that moment. Small numbers are how long each move took.")}>{t("Moves")}</SectionTitle>
                {reviewing && (
                  <div className="review-bar">
                    <span>{t("Viewing an earlier move")}</span>
                    <button className="btn small" onClick={() => setViewPly(null)}>{t("Back to current move")}</button>
                  </div>
                )}
                <MoveList san={live.san} secs={live.move_secs} selected={viewPly} onSelect={pick} fill />
              </section>
            )}

            {showWatch && live && (
              <section className="card">
                <SectionTitle help={t("The model's favourite moves in the position before its last move. A longer bar means it spent more of its thinking on that move.")}>{t("What the model sees")}</SectionTitle>
                <Thoughts fen={fenThen} top={live.top} limit={3} />
              </section>
            )}
          </div>

          <div className="boardcol">
            <BoardWithEval fit
              fen={shownFen} orientation="white" theme={ui.board_theme}
              lastMove={shownLast} showEval={ui.show_eval} evalValue={live?.eval ?? 0} arrows={arrows}
            />
            {mated && closedKing !== `${match?.id}-${index}` && (
              <KingDead loser={live.result === "1-0" ? "black" : "white"} onClose={() => setClosedKing(`${match?.id}-${index}`)} />
            )}
          </div>
        </div>
      )}
    </>
  );
}
