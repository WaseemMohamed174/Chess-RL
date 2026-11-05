import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play as PlayIcon, Square } from "lucide-react";
import { api, useSocket } from "../api";
import PageHead from "./PageHead";
import { BoardWithEval } from "./Board";
import { LineChartBox } from "./Charts";
import CapacityDialog, { isCapacity } from "./CapacityDialog";
import { reportStart, startAllTraining } from "../bulk";
import { Empty, InfoTip, ModelSelect, SectionTitle, Stat, Thoughts, Tip } from "./ui";
import { useApp, useToast } from "../context";
import { useI18n } from "../i18n";
import { fmtInt } from "../utils";

const ACTIVE = ["running", "paused", "starting"];
const HOLD_MS = 4000;   // a finished game stays visible this long before the next one replaces it

const STATE_LABEL = { running: "Running", paused: "Paused", starting: "Starting", stopped: "Stopped", failed: "Failed", done: "Done" };
const STATE_CLS = { running: "ok", paused: "warn", starting: "warn", failed: "bad", done: "ok" };
const PHASES = [
  ["selfplay", "Self-play", "The model plays games against itself. These games are its training material."],
  ["learning", "Learning", "The network studies the games it just played and adjusts itself to play better."],
  ["arena", "Arena", "The new version plays the current best version. It only replaces it if it wins enough."],
];

function resultInfo(live) {
  if (!live?.ended) return null;
  if (live.result === "draw") return { cls: "draw", text: "Draw" };
  return { cls: "won", text: live.result === "white" ? "White won" : "Black won", side: live.result };
}

/** One training model: a numbered button with everything you need at a glance. */
function RunCard({ n, r, selected, onSelect, t }) {
  const res = resultInfo(r.live);
  const ph = r.phase;
  const running = ACTIVE.includes(r.state);
  const pct = ph?.total ? Math.min(100, Math.round((ph.done / ph.total) * 100)) : 0;
  const phaseName = PHASES.find((p) => p[0] === ph?.name);
  return (
    <button type="button" className={`runcard${selected ? " sel" : ""}${res ? ` ${res.cls}` : ""}`}
            aria-pressed={selected} onClick={onSelect}>
      <span className="rc-num">{n}</span>
      <span className="rc-body">
        <span className="rc-top">
          <b className="rc-name">{r.model.name}</b>
          <span className={`pill small ${STATE_CLS[r.state] || ""}`}><i />{t(STATE_LABEL[r.state] ?? r.state)}</span>
        </span>
        <span className="rc-meta">
          <Tip text={t("Strength rating of this model. It goes up when a new version beats the old one.")}>{t("Elo")} {r.elo != null ? Math.round(r.elo) : "-"}</Tip>
          <span>·</span>
          <Tip text={t("How many full cycles of play, learn and test this model has finished.")}>{t("Iteration")} {r.iteration}</Tip>
        </span>
        <span className="rc-game">
          {res ? (
            <span className={`rc-result ${res.cls}`}>
              {res.side && <i className={`dot ${res.side}`} aria-hidden="true" />}{t(res.text)}
            </span>
          ) : running && ph?.name && ph.name !== "selfplay" ? (
            <span className="muted">{t(phaseName?.[1] ?? "Working")}…</span>
          ) : r.live ? (
            <>
              <Tip text={t("White pieces still on the board in the game shown for this model.")}><span className="pc"><i className="dot white" aria-hidden="true" />{r.live.white_pieces}</span></Tip>
              <Tip text={t("Black pieces still on the board in the game shown for this model.")}><span className="pc"><i className="dot black" aria-hidden="true" />{r.live.black_pieces}</span></Tip>
              <span className="muted">{t("Move")} {Math.ceil(r.live.ply / 2)}</span>
            </>
          ) : <span className="muted">{t("Waiting for the first game…")}</span>}
        </span>
        {ph?.total > 0 && running && (
          <span className="rc-prog" aria-label={`${pct}%`}><i style={{ width: `${pct}%` }} /></span>
        )}
      </span>
    </button>
  );
}

const SERVER_STATE = { READY: ["Ready", "ok"], STARTING: ["Starting", "warn"], BACKOFF: ["Restarting", "warn"],
  CRASHED: ["Restarting", "warn"], FAILED: ["Failed", "bad"], STOPPED: ["Stopped", ""] };

/** The one inference server all models share: live rates computed from two samples. */
function ServerPanel({ t }) {
  const [info, setInfo] = useState(null);
  const [rate, setRate] = useState(null);
  const prev = useRef(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      if (document.hidden) return;                       // nobody is looking: do not ask
      try {
        const d = await api("/api/training/server");
        if (!alive) return;
        setInfo(d);
        const s = d.stats;
        if (!s) { prev.current = null; setRate(null); return; }
        const p = prev.current;
        if (p && s.uptime_s < p.uptime_s) { prev.current = s; setRate(null); return; }   // server restarted
        const dt = p ? s.uptime_s - p.uptime_s : 0;
        if (dt > 0) {
          const dPos = s.positions - p.positions, dCalls = s.calls - p.calls, dReq = s.requests - p.requests;
          setRate({
            pos: dPos / dt,
            batch: dCalls > 0 ? dPos / dCalls : null,
            wait: dReq > 0 ? (1000 * (s.wait_s - p.wait_s)) / dReq : null,
            busy: Math.min(100, (100 * (s.busy_s - p.busy_s)) / dt),
          });
          prev.current = s;
        } else if (!p) prev.current = s;
      } catch { /* the page keeps working without this panel */ }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  if (!info?.enabled) return null;
  const [label, cls] = SERVER_STATE[info.state] ?? ["Starting", "warn"];
  const dash = (v, f) => (v == null ? "-" : f(v));
  const per = info.runs?.map((r) => r.workers).filter((n) => n > 0) ?? [];
  return (
    <div className="dock" style={{ marginTop: 14 }}><section>
      <SectionTitle
        help={t("One shared program that runs every model's network. Many light workers play the games and send it positions; it answers them in big batches, so several models train at once without each one needing its own copy of the network.")}
        right={<span className={`pill small ${cls}`}><i />{t(label)}{info.restarts > 0 ? ` \u00b7 ${info.restarts}\u00d7` : ""}</span>}>
        {t("Shared inference server")}
      </SectionTitle>
      <div className="stats">
        <Stat label={t("Positions/s")} value={dash(rate?.pos, (v) => fmtInt(Math.round(v)))}
              hint={t("How many board positions the server evaluates per second, over the last few seconds. Higher means more training progress.")} />
        <Stat label={t("Average batch")} value={dash(rate?.batch, (v) => Math.round(v))} unit={t("positions")}
              hint={t("Positions answered in one go. Bigger batches use the device better; very small ones mean it is waiting for work.")} />
        <Stat label={t("Queue wait")} value={dash(rate?.wait, (v) => v.toFixed(1))} unit="ms"
              hint={t("How long a request waits in the server before it is answered. It should stay small.")} />
        <Stat label={t("Server busy")} value={dash(rate?.busy, (v) => Math.round(v))} unit="%"
              hint={t("Share of time the server spends computing. Near 100% the server is the limit; low means the workers (CPU) are.")} />
        <Stat label={t("Workers per model")} value={per.length ? per.join(" / ") : "-"}
              hint={t("Self-play workers each training model has right now. They are shared evenly and rebalanced when a model starts or stops.")} />
        <Stat label={t("Models loaded")} value={info.stats?.loaded?.length ?? "-"}
              hint={t("Networks the server keeps ready: two per training model (the best and the candidate).")} />
      </div>
    </section></div>
  );
}

/** play -> learn -> test: shows which step the selected model is in. */
function PhaseSteps({ ph, active, t }) {
  return (
    <ol className="steps">
      {PHASES.map(([key, label, help], i) => (
        <li key={key} className={active && ph?.name === key ? "now" : ""}>
          <span className="sn">{i + 1}</span>
          <span>{t(label)}{key === ph?.name && ph.total > 0 && active ? ` ${ph.done}/${ph.total}` : ""}</span>
          <InfoTip text={t(help)} side="bottom" />
        </li>
      ))}
    </ol>
  );
}

/**
 * The live training window: one button per model that is training, the game it plays, its
 * numbers and its charts. The same window is used for single models (scope = "single") and
 * for the agents of one population (scope = the population's id), so the two never mix.
 */
export default function TrainingRuns({ scope = "single" }) {
  const isPop = scope !== "single";
  const { t } = useI18n();
  const toast = useToast();
  const { models, loadModels, settings, ui } = useApp();
  const [runs, setRuns] = useState({});            // run id -> everything we show for it
  const [boards, setBoards] = useState({});        // run id -> latest full board event
  const [sel, setSel] = useState(null);
  const [maxRuns, setMaxRuns] = useState(4);
  const [lastRun, setLastRun] = useState(null);
  const [busyModels, setBusyModels] = useState([]);   // every model that trains now, whoever started it
  const [newModel, setNewModel] = useState(null);
  const [gamesPerIter, setGamesPerIter] = useState("");
  const [sims, setSims] = useState("");
  const [busy, setBusy] = useState(false);
  const [cap, setCap] = useState(null);                // capacity warning for a single-model start
  const [capAll, setCapAll] = useState(false);          // ...true when it is for "Start all"
  const hold = useRef({});                         // run id -> time until which a finished game stays
  const watching = useRef(null);

  // Prefill the two start options from Settings (only until the person edits them).
  const v = settings?.values;
  useEffect(() => {
    if (!v) return;
    setGamesPerIter((x) => (x === "" ? String(v.train.games_per_iteration) : x));
    setSims((x) => (x === "" ? String(v.mcts.simulations) : x));
  }, [v]);

  const fromStatus = (x) => ({
    id: x.run.id, model: x.model, run: x.run, state: x.live?.state ?? x.run.status,
    error: x.live?.error ?? null, metrics: x.metrics || [],
    last: x.live?.last ?? x.metrics?.at(-1) ?? null,          // a stopped run still shows its last iteration
    live: x.live?.live ?? null, phase: x.live?.phase ?? null,
    games: x.live?.games ?? x.metrics?.at(-1)?.games ?? 0,
    buffer: x.live?.buffer ?? x.metrics?.at(-1)?.buffer ?? null, elo: x.live?.elo ?? x.metrics?.at(-1)?.elo ?? x.model.elo,
    iteration: x.live?.iteration ?? x.metrics?.at(-1)?.iteration ?? 0,
  });

  const refresh = useCallback(async () => {
    try {
      const s = await api(`/api/training/status?population_id=${scope}`);
      const map = {};
      s.runs.forEach((x) => { map[x.run.id] = fromStatus(x); });
      setRuns(map);
      setLastRun(s.last ? fromStatus(s.last) : null);
      setMaxRuns(s.max_parallel);
      setBusyModels(s.busy_models ?? []);
      setSel((cur) => (cur != null && map[cur] ? cur : s.watch != null && map[s.watch] ? s.watch : Number(Object.keys(map)[0]) || null));
    } catch (e) { toast(e.message, "error"); }
  }, [toast, scope]);

  // A different population (or the other view) starts from a clean window.
  useEffect(() => {
    setRuns({}); setBoards({}); setLastRun(null); setSel(null); watching.current = null;
    refresh();
  }, [refresh]);

  // The agents of a population are started by the server, one after another: follow it.
  useSocket(isPop ? "/ws/population" : null, (m) => { if (m.type === "population") refresh(); });
  useEffect(() => {
    if (!isPop) return undefined;
    const id = setInterval(() => { if (!document.hidden) refresh(); }, 5000);
    return () => clearInterval(id);
  }, [isPop, refresh]);

  const patch = (id, fn) => setRuns((rs) => (rs[id] ? { ...rs, [id]: { ...rs[id], ...fn(rs[id]) } } : rs));

  useSocket("/ws/training", (m) => {
    const id = m.run_id;
    switch (m.type) {
      case "hello": refresh(); break;
      case "status":
        patch(id, () => ({ state: m.state, error: m.error ?? null }));
        if (!ACTIVE.includes(m.state)) { loadModels(); refresh(); }
        else if (id != null && !runs[id]) refresh();      // a run we did not know yet
        break;
      case "metrics":
        patch(id, (r) => ({
          last: m, games: m.games, buffer: m.buffer, elo: m.elo, iteration: m.iteration,
          metrics: [...r.metrics.filter((x) => x.iteration !== m.iteration), m].sort((a, b) => a.iteration - b.iteration),
        }));
        break;
      case "game": patch(id, (r) => ({ games: r.games + 1 })); break;
      case "phase": patch(id, () => ({ phase: { name: m.name, done: m.done, total: m.total } })); break;
      case "live": {
        const now = Date.now();
        if (m.ended) {
          hold.current[id] = now + HOLD_MS;
          if (m.fen) setBoards((b) => ({ ...b, [id]: { ...(b[id] || {}), fen: m.fen } }));
        } else if (now < (hold.current[id] || 0)) break;   // let the finished game stay visible
        patch(id, () => ({ live: m }));
        break;
      }
      case "selfplay_move":
        if (Date.now() < (hold.current[id] || 0)) break;
        setBoards((b) => ({ ...b, [id]: m }));
        break;
      default:
    }
  });

  const list = useMemo(() => Object.values(runs).sort((a, b) => a.id - b.id), [runs]);
  const cur = runs[sel] ?? list[0] ?? null;
  const shown = cur ?? lastRun;                       // after everything stopped, show the last run
  const active = !!cur && ACTIVE.includes(cur.state);
  const board = cur ? boards[cur.id] : null;
  const res = resultInfo(cur?.live);

  // Only the model open on this page streams its board; the others train faster without one.
  useEffect(() => {
    const id = cur?.id ?? null;
    if (id != null && watching.current !== id) {
      watching.current = id;
      api("/api/training/watch", { method: "POST", body: { run_id: id } }).catch(() => {});
    }
  }, [cur?.id]);
  useEffect(() => () => {
    watching.current = null;
    fetch("/api/training/watch", { method: "POST", keepalive: true, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run_id: null }) }).catch(() => {});
  }, []);

  const act = async (id, name) => {
    setBusy(true);
    try { await api(`/api/training/runs/${id}/${name}`, { method: "POST" }); }
    catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };
  const actAll = async (name) => {
    setBusy(true);
    try {                                    // only the runs of this window, never the other one's
      await Promise.all(list.filter((r) => ACTIVE.includes(r.state))
        .map((r) => api(`/api/training/runs/${r.id}/${name}`, { method: "POST" })));
    }
    catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };

  const idle = models.filter((m) => !busyModels.includes(m.id) && m.population_id == null);   // population agents are trained on the Population tab
  const picked = idle.some((m) => m.id === newModel) ? newModel : idle[0]?.id ?? null;
  const over = busyModels.length >= maxRuns;               // only a suggestion: the warning window decides

  const start = async (force = false) => {
    const overrides = {};
    const g = Number(gamesPerIter), s = Number(sims);
    if (g > 0) overrides.train = { games_per_iteration: g };
    if (s > 0) overrides.mcts = { simulations: s };
    setBusy(true);
    try {
      const r = await api("/api/training/start", { method: "POST", body: { model_id: picked, overrides, force } });
      setCap(null);
      await refresh();
      setSel(r.id);
    } catch (e) {
      if (isCapacity(e)) setCap(e.detail);
      else { setCap(null); toast(e.message, "error"); }
    } finally { setBusy(false); }
  };

  const startAll = async (force = false) => {
    setBusy(true);
    try {
      const r = await startAllTraining(force);
      setCap(null); setCapAll(false);
      reportStart(r, t, toast);
      await refresh();
      if (r.started[0]) setSel(r.started[0].run_id);
    } catch (e) {
      if (isCapacity(e)) { setCap(e.detail); setCapAll(true); }
      else { setCap(null); setCapAll(false); toast(e.message, "error"); }
    } finally { setBusy(false); }
  };

  const last = shown?.last;
  const arena = last?.arena;
  const threshold = shown?.run?.hyperparams?.train?.arena_threshold ?? 0.55;
  const metrics = shown?.metrics ?? [];
  const lastRow = metrics.at(-1);
  const pausedAny = list.some((r) => r.state === "paused");

  const emptyMsg = !cur ? t("Games appear here while a model is training.")
    : cur.state === "paused" ? t("Paused. Press Resume to continue the game.")
    : cur.phase?.name === "learning" || cur.phase?.name === "arena"
      ? t("The model is not playing self-play games right now (it is learning or being tested). Games return in the next iteration.")
      : t("Waiting for the next move…");

  if (isPop && !shown) return null;

  return (
    <>
      {!isPop && <PageHead title="Training"
        subtitle="Each model plays against itself, learns from those games, and a new version only replaces the old one if it wins a match. You can train several models at once.">
        <span className="pill"><i />{busyModels.length} / {maxRuns} {t("training now")}</span>
      </PageHead>}

      {!isPop && models.length === 0 ? (
        <div className="dock"><Empty title="No models yet"><span>{t("Create a model on the Models page, then start training it.")}</span></Empty></div>
      ) : (
        <div className="stack" style={{ gap: 22 }}>
          {isPop && (
            <SectionTitle help={t("The agents of this population while they train, in the same window as single models. Click an agent to watch its games. Single models are on the other tab and do not appear here.")}>
              {t("Live training of this population")}
            </SectionTitle>
          )}

          {/* 1: start another model (single models only) */}
          {!isPop && (
          <div className="dock"><section>
            <SectionTitle help={t("Pick a model that is not training yet. It starts right away and gets its own button below. Several models can train at the same time; more models means each one is a bit slower.")}
              right={idle.length > 1 && (
                <Tip text={t("Start every model that is not training yet, with the saved settings. You are warned first if the computer cannot hold them all.")} side="bottom" align="end">
                  <button className="btn small" onClick={() => startAll()} disabled={busy}><PlayIcon size={14} />{t("Start all")} ({idle.length})</button>
                </Tip>
              )}>{t("Start training a model")}</SectionTitle>
            {idle.length === 0 ? (
              <p className="muted">{t("Every model is already training. Create another one on the Models page to train more.")}</p>
            ) : (
              <div className="formgrid" style={{ alignItems: "end" }}>
                <ModelSelect id="tm" label={t("Model")} models={idle} value={picked} onChange={setNewModel} />
                <div className="field">
                  <label htmlFor="gpi">{t("Games per iteration")}<InfoTip text={t("How many self-play games the model plays before each learning step. More games means better training data but a longer iteration.")} /></label>
                  <input id="gpi" type="number" min="1" value={gamesPerIter} onChange={(e) => setGamesPerIter(e.target.value)} />
                </div>
                <div className="field">
                  <label htmlFor="sim">{t("Simulations")}<InfoTip text={t("How many positions the model searches before every move. More simulations means stronger, slower play (50 to 800).")} /></label>
                  <input id="sim" type="number" min="50" max="800" value={sims} onChange={(e) => setSims(e.target.value)} />
                </div>
                <button className="btn primary" onClick={() => start()} disabled={busy || !picked}>
                  <PlayIcon size={16} />{busy ? t("Starting...") : t("Start training")}
                </button>
              </div>
            )}
            {over && <p className="help" style={{ marginTop: 10 }}>{t("That is already the number of models this computer handles well. You can start more, but you will be warned first and each one will be slower.")}</p>}
          </section></div>
          )}

          {/* 2: one numbered button per model that is training */}
          {list.length > 0 && (
            <div>
              <SectionTitle
                help={t("One button per model that is training. Click a button to watch that model's games below. Only the model you are watching draws its board, so the others train faster.")}
                right={list.length > 1 && (
                  <>
                    {pausedAny
                      ? <button className="btn small" onClick={() => actAll("resume")} disabled={busy}><PlayIcon size={14} />{t("Resume all")}</button>
                      : <button className="btn small" onClick={() => actAll("pause")} disabled={busy}><Pause size={14} />{t("Pause all")}</button>}
                    {!isPop && <button className="btn small danger" onClick={() => actAll("stop")} disabled={busy}><Square size={14} />{t("Stop all")}</button>}
                  </>
                )}>
                {isPop ? t("Agents training now") : t("Models training now")}
              </SectionTitle>
              <div className="runcards">
                {list.map((r, i) => <RunCard key={r.id} n={i + 1} r={r} t={t} selected={cur?.id === r.id} onSelect={() => setSel(r.id)} />)}
              </div>
              <p className="help" style={{ marginTop: 8 }}>
                <i className="dot white" aria-hidden="true" /> / <i className="dot black" aria-hidden="true" /> {t("= pieces left for White / Black in the game being shown.")}
                {" "}<span className="legend draw">{t("Draw")}</span> <span className="legend won">{t("a side won")}</span> {t("= that game just ended.")}
              </p>
              <ServerPanel t={t} />
            </div>
          )}

          {!shown ? (
            <div className="dock"><Empty title="No training yet"><span>{t("Pick a model above and press Start training. Charts fill in after the first iteration.")}</span></Empty></div>
          ) : (
            <>
              {/* 3: the model we are looking at */}
              <div className="dock">
                <section className="row between" style={{ flexWrap: "wrap", gap: 16 }}>
                  <div>
                    <div className="row" style={{ gap: 10 }}>
                      <h2 style={{ margin: 0 }}>{cur ? `${list.findIndex((r) => r.id === cur.id) + 1}. ` : ""}{shown.model.name}</h2>
                      <span className={`pill ${STATE_CLS[shown.state] || ""}`}><i />{t(STATE_LABEL[shown.state] ?? shown.state)}</span>
                    </div>
                    <div style={{ marginTop: 10 }}><PhaseSteps ph={shown.phase} active={active && shown.state === "running"} t={t} /></div>
                  </div>
                  {active && (
                    <div className="btn-row">
                      {cur.state === "paused"
                        ? <button className="btn primary" onClick={() => act(cur.id, "resume")} disabled={busy}><PlayIcon size={16} />{t("Resume")}</button>
                        : <button className="btn" onClick={() => act(cur.id, "pause")} disabled={busy}><Pause size={16} />{t("Pause")}</button>}
                      {!isPop && <button className="btn danger" onClick={() => act(cur.id, "stop")} disabled={busy}><Square size={16} />{t("Stop")}</button>}
                    </div>
                  )}
                </section>
                {shown.state === "failed" && shown.error && <section><pre className="help" style={{ whiteSpace: "pre-wrap", color: "var(--bad)" }}>{shown.error}</pre></section>}
                <section className="stats" style={{ maxWidth: 760 }}>
                  <Stat label={t("Elo")} value={shown.elo != null ? Math.round(shown.elo) : "-"} hint={t("Strength rating. Every time a new version beats the old one it goes up.")} />
                  <Stat label={t("Iteration")} value={shown.iteration} hint={t("One full cycle: play games, learn from them, then test the new version.")} />
                  <Stat label={t("Games played")} value={fmtInt(shown.games)} hint={t("Total self-play games finished in this training run.")} />
                  <Stat label={t("Speed")} value={lastRow?.games_per_min != null ? lastRow.games_per_min.toFixed(1) : "-"} unit={t("games/min")} hint={t("How many self-play games finish per minute. Higher means faster training.")} />
                  <Stat label={t("Memory")} value={fmtInt(shown.buffer)} unit={t("positions")} hint={t("Positions from recent games that the model learns from. The oldest ones are dropped when it is full.")} />
                </section>
              </div>

              {/* 4: live game (big) + explanations */}
              <div className="live-grid">
                <div className="dock"><section>
                  <SectionTitle help={t("A real game the selected model is playing against itself right now. Training data comes from games like this one.")}
                    right={res && <span className={`rc-result ${res.cls}`}>{res.side && <i className={`dot ${res.side}`} aria-hidden="true" />}{t(res.text)}</span>}>
                    {t("Live self-play")}
                  </SectionTitle>
                  {cur && board?.fen ? (
                    <div className="live-stage">
                      <BoardWithEval fit id="live" fen={board.fen} orientation="white" theme={ui.board_theme}
                        lastMove={board.uci} showEval={ui.show_eval} evalValue={board.eval} />
                      {res && (
                        <div className={`board-result ${res.cls}`} role="status">
                          {res.side && <i className={`dot ${res.side}`} aria-hidden="true" />}<b>{t(res.text)}</b>
                        </div>
                      )}
                    </div>
                  ) : <p className="muted live-empty">{emptyMsg}</p>}
                  <p className="help" style={{ marginTop: 8 }}>{t("The bar beside the board shows who the model thinks is winning. Only the model you selected above shows its board.")}</p>
                </section></div>

                <aside className="dock">
                  <section>
                    <SectionTitle help={t("What happened in the most recent finished iteration of this model.")}>{t("Latest iteration")}</SectionTitle>
                    {last ? (
                      <dl className="kv">
                        <dt>{t("Iteration")}</dt><dd>{last.iteration}</dd>
                        <dt>{t("Policy loss")}<InfoTip side="top" text={t("How far the model's first move choices were from the better choices found by its own search. Lower is better; it should fall over time.")} /></dt>
                        <dd>{last.policy_loss != null ? last.policy_loss.toFixed(3) : "-"}</dd>
                        <dt>{t("Value loss")}<InfoTip side="top" text={t("How wrong the model's guess of who wins was, compared with how the games really ended. Lower is better.")} /></dt>
                        <dd>{last.value_loss != null ? last.value_loss.toFixed(3) : "-"}</dd>
                        {arena && <>
                          <dt>{t("Arena (W / L / D)")}<InfoTip side="top" text={t("The match between the new version and the old best version, from the new version's side: Wins / Losses / Draws.")} /></dt>
                          <dd>{arena.wins} / {arena.losses} / {arena.draws}</dd>
                          <dt>{t("Arena score")}<InfoTip side="top" text={`${t("Win = 1 point, draw = half. The new version replaces the old one only if it scores at least")} ${Math.round(threshold * 100)}%.`} /></dt>
                          <dd>{(arena.score * 100).toFixed(0)}%</dd>
                        </>}
                        <dt>{t("New version")}<InfoTip side="top" text={t("Promoted = the new version won the arena and is now the model. Kept = it did not, so the old version stays.")} /></dt>
                        <dd className={last.promoted ? "good" : "muted"}>{last.promoted ? t("Promoted") : t("Kept")}</dd>
                      </dl>
                    ) : <p className="muted">{t("Nothing yet. The first iteration has to finish.")}</p>}
                  </section>
                  <section>
                    <SectionTitle help={t("The moves the model liked most in the last position of the live game. A longer bar means it spent more of its search on that move. Moves are written as from-square then to-square, like e2e4.")}>{t("What the model sees")}</SectionTitle>
                    {board?.top?.length > 0 ? <Thoughts top={board.top} /> : <p className="muted">{t("Shown for the model you are watching, while it plays.")}</p>}
                  </section>
                </aside>
              </div>

              {/* 5: charts, each one explained */}
              {metrics.length === 0 ? (
                <div className="dock"><Empty title="No charts yet"><span>{t("Charts fill in after the first iteration finishes.")}</span></Empty></div>
              ) : (
                <div className="grid2" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))" }}>
                  <div className="dock"><section>
                    <SectionTitle help={t("Two error scores. Lower means the model is learning.")}>{t("Loss")}</SectionTitle>
                    <p className="help chart-note">{t("Both lines should slowly go down. Small bumps are normal.")}</p>
                    <LineChartBox data={metrics} lines={[
                      { key: "policy_loss", name: t("Policy loss"), color: "#e0a93b" },
                      { key: "value_loss", name: t("Value loss"), color: "#6db3d4" }]} />
                  </section></div>
                  <div className="dock"><section>
                    <SectionTitle help={t("Strength rating after each iteration.")}>{t("Elo")}</SectionTitle>
                    <p className="help chart-note">{t("Goes up each time a new version beats the old one. A flat line means no improvement yet.")}</p>
                    <LineChartBox data={metrics} lines={[{ key: "elo", name: t("Elo"), color: "#5fbf8f" }]} />
                  </section></div>
                  <div className="dock"><section>
                    <SectionTitle help={t("How fast self-play runs.")}>{t("Speed")}</SectionTitle>
                    <p className="help chart-note">{t("Self-play games finished per minute. Higher means faster training.")}</p>
                    <LineChartBox data={metrics} lines={[{ key: "games_per_min", name: t("Games per minute"), color: "#e2705f" }]} />
                  </section></div>
                </div>
              )}
            </>
          )}
        </div>
      )}
      {cap && <CapacityDialog info={cap} busy={busy} onCancel={() => { setCap(null); setCapAll(false); }} onForce={() => (capAll ? startAll(true) : start(true))} />}
    </>
  );
}

