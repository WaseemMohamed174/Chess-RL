import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play as PlayIcon, Plus, Square, Trash2 } from "lucide-react";
import { api, useSocket } from "../api";
import { useApp, useToast } from "../context";
import { useI18n } from "../i18n";
import CapacityDialog, { isCapacity } from "./CapacityDialog";
import TrainingRuns from "./TrainingRuns";
import { LineChartBox } from "./Charts";
import { Empty, Modal, SectionTitle, Stat } from "./ui";

const STATUS_LABEL = { idle: "Idle", running: "Running", paused: "Paused", stopped: "Stopped", failed: "Failed", done: "Done" };
const STATUS_CLS = { running: "ok", paused: "warn", failed: "bad", done: "ok" };
const PHASE_LABEL = { idle: "Idle", training: "Training", evaluating: "Scoring", selecting: "Selecting" };
const PHASE_HELP = {
  training: "Every agent plays, learns and tests itself for a few iterations.",
  evaluating: "Each agent plays the champion. Its score is the share of points it wins.",
  selecting: "The strongest agents stay. The others are replaced by slightly changed copies.",
};
const ROLE_LABEL = { seed: "First group", survivor: "Kept", child: "New copy" };

const pct = (v) => (v == null ? "-" : `${Math.round(v * 100)}%`);
const num = (v, d = 2) => (v == null ? "-" : Number(v).toFixed(d));
const sci = (v) => (v == null ? "-" : Number(v).toExponential(1));

/** Name of an agent by model id, looking through every generation (it may have been removed). */
function nameIndex(members) {
  const m = {};
  members.forEach((x) => { if (x.model_id != null) m[x.model_id] = x.model_name; });
  return m;
}

function CreateForm({ onCreated }) {
  const { t } = useI18n();
  const toast = useToast();
  const { models, settings } = useApp();
  const [name, setName] = useState("");
  const [base, setBase] = useState(null);
  const [size, setSize] = useState("");
  const [busy, setBusy] = useState(false);
  const lim = settings?.limits?.population?.size ?? [2, 100];
  const def = settings?.values?.population?.size ?? 10;
  const picked = models.some((m) => m.id === base) ? base : models[0]?.id ?? null;

  if (models.length === 0)
    return <p className="muted">{t("Create a model on the Models page first. The population starts as copies of it.")}</p>;

  const create = async () => {
    setBusy(true);
    try {
      const n = Number(size);
      const out = await api("/api/populations", { method: "POST", body: {
        name, base_model_id: picked, overrides: n ? { size: n } : {} } });
      toast(t("Population created"), "ok");
      setName(""); setSize("");
      onCreated(out.population?.id ?? out.id);
    } catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };

  return (
    <div className="formgrid" style={{ alignItems: "end" }}>
      <div className="field">
        <label htmlFor="pn">{t("Name")}</label>
        <input id="pn" type="text" maxLength={50} placeholder={t("e.g. Spring run")} value={name} onChange={(e) => setName(e.target.value)} />
      </div>
      <div className="field">
        <label htmlFor="pb">{t("Starting model")}</label>
        <select id="pb" value={picked ?? ""} onChange={(e) => setBase(Number(e.target.value))}>
          {models.map((m) => <option key={m.id} value={m.id}>{m.name} · {Math.round(m.elo)} Elo</option>)}
        </select>
      </div>
      <div className="field">
        <label htmlFor="ps">{t("Agents")}</label>
        <input id="ps" type="number" min={lim[0]} max={lim[1]} placeholder={String(def)} value={size} onChange={(e) => setSize(e.target.value)} />
        <span className="help">{t("How many models are in the group.")} ({lim[0]}–{lim[1]})</span>
      </div>
      <button className="btn primary" onClick={create} disabled={busy || !name.trim() || picked == null}>
        <Plus size={16} />{busy ? t("Creating...") : t("Create population")}
      </button>
    </div>
  );
}

function ChampionChart({ summary, t }) {
  const data = summary.filter((r) => r.best_score != null)
    .map((r) => ({ generation: r.generation, best: r.best_score, mean: r.mean_score }));
  return (
    <div className="dock"><section>
      <SectionTitle help={t("After every generation each agent plays the champion. The line shows the best score and the average. Scores are measured against the previous champion, so a line that stays near 50% still means steady progress.")}>
        {t("Champion over generations")}
      </SectionTitle>
      {data.length === 0 ? <p className="muted">{t("The chart appears when the first generation has been scored.")}</p> : (
        <>
          <p className="help chart-note">{t("Score = share of points won against the champion (1 = always wins, 0.5 = equal).")}</p>
          <LineChartBox data={data} xKey="generation" xLabel={t("Generation")} domain={[0, 1]} lines={[
            { key: "best", name: t("Best score"), color: "#e0a93b" },
            { key: "mean", name: t("Average score"), color: "#6db3d4" }]} />
        </>
      )}
    </section></div>
  );
}

function Agents({ members, generation, names, t }) {
  const rows = members.filter((m) => m.generation === generation);
  if (rows.length === 0) return <p className="muted">{t("No agents in this generation yet.")}</p>;
  return (
    <div className="tablewrap"><table className="poptable">
      <thead><tr>
        <th>#</th><th>{t("Agent")}</th><th>{t("Role")}</th><th>{t("Copied from")}</th>
        <th>{t("Score")}</th><th>{t("Loss")}</th><th>{t("Learning rate")}</th><th>c_puct</th>
        <th>{t("Temperature")}</th><th>{t("Noise")}</th><th>{t("Simulations")}</th>
      </tr></thead>
      <tbody>
        {rows.map((m) => {
          const h = m.hyperparams || {};
          return (
            <tr key={m.id} className={m.rank === 1 ? "top" : ""}>
              <td>{m.rank ?? "-"}</td>
              <td><b>{m.model_name}</b>{m.model_id == null && <span className="muted"> {t("(removed)")}</span>}</td>
              <td>{t(ROLE_LABEL[m.role] ?? m.role)}</td>
              <td>{m.role !== "child" || m.parent_model_id == null ? "-" : names[m.parent_model_id] ?? t("(removed model)")}</td>
              <td>{pct(m.score)}</td><td>{num(m.loss, 3)}</td>
              <td>{sci(h.train?.lr)}</td><td>{num(h.mcts?.c_puct)}</td>
              <td>{num(h.mcts?.temperature)}</td><td>{num(h.mcts?.dirichlet_eps)}</td>
              <td>{h.mcts?.simulations ?? "-"}</td>
            </tr>
          );
        })}
      </tbody>
    </table></div>
  );
}

function Lineage({ members, generation, names, t }) {
  const kids = members.filter((m) => m.generation === generation && m.role === "child" && m.parent_model_id != null);
  const groups = {};
  kids.forEach((k) => { (groups[k.parent_model_id] ||= []).push(k.model_name); });
  const ids = Object.keys(groups);
  return (
    <>
      <p className="help" style={{ marginBottom: 10 }}>{t("Each new agent of this generation starts as a copy of one kept agent, with slightly changed settings.")}</p>
      {ids.length === 0 ? <p className="muted">{t("The first generation has no parents. Its agents are copies of the starting model.")}</p> : (
        <ul className="lineage">
          {ids.map((pid) => (
            <li key={pid}>
              <b className="lin-parent">{names[pid] ?? t("(removed model)")}</b>
              <span className="lin-arrow" aria-hidden="true">→</span>
              <span className="lin-kids">{groups[pid].map((n) => <span key={n} className="chip">{n}</span>)}</span>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

export default function PopulationPanel() {
  const { t } = useI18n();
  const toast = useToast();
  const { loadModels, models } = useApp();
  const [list, setList] = useState([]);
  const [sel, setSel] = useState(null);
  const [detail, setDetail] = useState(null);
  const [gen, setGen] = useState(null);                 // generation shown in the table (null = current)
  const [generations, setGenerations] = useState("1");
  const [parallel, setParallel] = useState("0");
  const [deleteReplaced, setDeleteReplaced] = useState(false);
  const [cap, setCap] = useState(null);
  const [confirmDel, setConfirmDel] = useState(false);
  const [busy, setBusy] = useState(false);
  const selRef = useRef(null);
  selRef.current = sel;

  const loadList = useCallback(async () => {
    try {
      const ps = await api("/api/populations");
      setList(ps);
      setSel((cur) => (cur != null && ps.some((p) => p.id === cur) ? cur : ps.at(-1)?.id ?? null));
    } catch (e) { toast(e.message, "error"); }
  }, [toast]);

  const loadDetail = useCallback(async () => {
    const id = selRef.current;
    if (id == null) { setDetail(null); return; }
    try { setDetail(await api(`/api/populations/${id}`)); } catch { setDetail(null); }
  }, []);

  useEffect(() => { loadList(); }, [loadList]);
  useEffect(() => { setGen(null); loadDetail(); }, [sel, loadDetail]);

  const running = !!detail?.live?.running;
  // The coordinator pushes its state; the numbers (scores, new agents) are re-read when it changes.
  useSocket("/ws/population", (m) => {
    if (m.type === "hello" || (m.type === "population" && m.population_id === selRef.current)) loadDetail();
    if (m.type === "population") loadList();
  });
  useEffect(() => {                                   // a slow safety net while it runs
    if (!running) return undefined;
    const id = setInterval(() => { if (!document.hidden) loadDetail(); }, 5000);
    return () => clearInterval(id);
  }, [running, loadDetail]);

  const start = async (extra = {}) => {
    setBusy(true);
    try {
      await api(`/api/populations/${sel}/start`, { method: "POST", body: {
        generations: Number(generations) || 0, parallel: Number(parallel) || 0,
        delete_replaced: deleteReplaced, ...extra } });
      setCap(null);
      await loadDetail(); await loadList();
    } catch (e) {
      if (isCapacity(e)) setCap(e.detail);
      else { setCap(null); toast(e.message, "error"); }
    } finally { setBusy(false); }
  };
  const stop = async () => {
    setBusy(true);
    try { await api(`/api/populations/${sel}/stop`, { method: "POST" }); await loadDetail(); loadModels(); }
    catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };
  const remove = async () => {
    setBusy(true);
    try { await api(`/api/populations/${sel}`, { method: "DELETE" }); setConfirmDel(false); setSel(null); await loadList(); }
    catch (e) { setConfirmDel(false); toast(e.message, "error"); } finally { setBusy(false); }
  };

  const members = detail?.members ?? [];
  const names = useMemo(() => ({ ...Object.fromEntries(models.map((m) => [m.id, m.name])), ...nameIndex(members) }), [members, models]);
  const summary = detail?.summary ?? [];
  const current = detail?.generation ?? 0;
  const shownGen = gen ?? current;
  const genList = useMemo(() => [...new Set(members.map((m) => m.generation))].sort((a, b) => a - b), [members]);
  const nowMembers = members.filter((m) => m.generation === current);
  const best = summary.filter((r) => r.best_score != null).at(-1);
  const champion = members.filter((m) => m.generation === current - 1 && m.rank === 1)[0]
    ?? members.filter((m) => m.generation === current && m.rank === 1)[0];
  const status = detail?.status ?? "idle";
  const phase = detail?.live?.phase ?? detail?.phase ?? "idle";
  const failed = detail?.live?.failed_agents ?? [];

  return (
    <div className="stack" style={{ gap: 22 }}>
      <div className="dock"><section>
        <SectionTitle help={t("A population is a group of copies of one model. They all train, the strongest ones are kept, and the weakest are replaced by slightly changed copies of the strong ones. Repeat, and the group gets better over generations.")}>
          {t("New population")}
        </SectionTitle>
        <CreateForm onCreated={async (id) => { await loadList(); setSel(id); }} />
      </section></div>

      {list.length === 0 ? (
        <div className="dock"><Empty title="No populations yet"><span>{t("Create one above. It makes one copy of the starting model for every agent.")}</span></Empty></div>
      ) : (
        <>
          <div className="dock"><section className="row between" style={{ flexWrap: "wrap", gap: 14 }}>
            <div className="row" style={{ gap: 12, flexWrap: "wrap" }}>
              <select aria-label={t("Population")} style={{ width: "auto", minWidth: 200 }} value={sel ?? ""} onChange={(e) => setSel(Number(e.target.value))}>
                {list.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
              <span className={`pill ${STATUS_CLS[status] || ""}`}><i />{t(STATUS_LABEL[status] ?? status)}</span>
              {running && <span className="pill warn"><i />{t(PHASE_LABEL[phase] ?? phase)}</span>}
            </div>
            <button className="btn small danger" onClick={() => setConfirmDel(true)} disabled={busy || running}>
              <Trash2 size={14} />{t("Delete population")}
            </button>
          </section>
          {running && PHASE_HELP[phase] && <section><p className="help">{t(PHASE_HELP[phase])}</p></section>}
          {detail?.live?.error && <section><pre className="help" style={{ whiteSpace: "pre-wrap", color: "var(--bad)" }}>{detail.live.error}</pre></section>}
          {failed.length > 0 && <section><p className="help">{t("These agents failed twice and were scored as they were:")} {failed.join(", ")}</p></section>}
          <section className="stats" style={{ maxWidth: 760 }}>
            <Stat label={t("Generation")} value={current} hint={t("The group has been replaced this many times. Generation 0 is the first group.")} />
            <Stat label={t("Agents")} value={nowMembers.length} hint={t("Models in the group.")} />
            <Stat label={t("Best score")} value={best ? pct(best.best_score) : "-"} hint={t("Share of points the best agent won against the champion in its generation. 50% means equal.")} />
            <Stat label={t("Champion")} value={champion?.model_name ?? "-"} hint={t("The agent the next generation is measured against: the best one of the last generation.")} />
            <Stat label={t("Training now")} value={detail?.live?.active_runs?.length ?? 0} hint={t("Agents that are training at this moment.")} />
          </section>
          <section>
            <div className="formgrid" style={{ alignItems: "end" }}>
              <div className="field">
                <label htmlFor="pg">{t("Generations to run")}</label>
                <input id="pg" type="number" min="0" value={generations} onChange={(e) => setGenerations(e.target.value)} />
                <span className="help">{t("How many generations to finish now. 0 keeps going until you press Stop.")}</span>
              </div>
              <div className="field">
                <label htmlFor="pp">{t("Trained at the same time")}</label>
                <input id="pp" type="number" min="0" value={parallel} onChange={(e) => setParallel(e.target.value)} />
                <span className="help">{t("How many agents train together. 0 means all of them. A smaller number makes the others wait their turn, so each trains faster.")}</span>
              </div>
              <label className="switch" style={{ alignSelf: "center" }}>
                <span>{t("Delete replaced models")}<br /><span className="help">{t("Removes the models of agents that were replaced. The history of the population stays either way.")}</span></span>
                <input type="checkbox" checked={deleteReplaced} onChange={(e) => setDeleteReplaced(e.target.checked)} />
              </label>
              {running
                ? <button className="btn danger" onClick={stop} disabled={busy}><Square size={16} />{t("Stop")}</button>
                : <button className="btn primary" onClick={() => start()} disabled={busy || !sel}>
                    <PlayIcon size={16} />{current > 0 || summary.some((r) => r.best_score != null) ? t("Continue") : t("Start")}
                  </button>}
            </div>
          </section></div>

          {/* the live window of this population's agents (same window as single models, kept apart) */}
          {sel != null && <TrainingRuns key={sel} scope={sel} />}

          <ChampionChart summary={summary} t={t} />

          <div className="dock"><section>
            <SectionTitle help={t("One row per agent. Ranked by score; a tie goes to the lower loss. The last columns are the settings the agent trained with. Copies get these slightly changed.")}
              right={genList.length > 1 && (
                <select aria-label={t("Generation")} style={{ width: "auto" }} value={shownGen} onChange={(e) => setGen(Number(e.target.value))}>
                  {genList.map((g) => <option key={g} value={g}>{t("Generation")} {g}</option>)}
                </select>
              )}>
              {t("Agents")} · {t("Generation")} {shownGen}
            </SectionTitle>
            <Agents members={members} generation={shownGen} names={names} t={t} />
          </section>
          <section>
            <SectionTitle>{t("Who was copied from whom")}</SectionTitle>
            <Lineage members={members} generation={shownGen} names={names} t={t} />
          </section></div>
        </>
      )}

      {cap && (
        <CapacityDialog info={cap} busy={busy} onCancel={() => setCap(null)}
          onForce={() => start({ force: true })}
          onBatch={(n) => { setParallel(String(n)); start({ parallel: n, force: true }); }} />
      )}
      {confirmDel && (
        <Modal title={t("Delete this population?")} onClose={() => setConfirmDel(false)}>
          <p className="muted">{t("Its history is removed. The models stay on the Models page.")}</p>
          <div className="btn-row cap-actions">
            <button className="btn ghost" onClick={() => setConfirmDel(false)}>{t("Cancel")}</button>
            <button className="btn danger" onClick={remove} disabled={busy}><Trash2 size={16} />{t("Delete")}</button>
          </div>
        </Modal>
      )}
    </div>
  );
}
