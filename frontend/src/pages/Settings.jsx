import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import PageHead from "../components/PageHead";
import { InfoTip, Segmented } from "../components/ui";
import { useApp, useToast } from "../context";
import { useI18n } from "../i18n";
import { BOARD_THEMES } from "../utils";

// kind: int | float | bool. `help` says what the setting does in plain words.
const FIELDS = {
  mcts: [
    ["simulations", "Simulations per move", "int", "How many positions the model looks at before it moves. More is stronger and slower."],
    ["c_puct", "Exploration (c_puct)", "float", "Higher values make the search try more unusual moves."],
    ["dirichlet_alpha", "Noise shape (alpha)", "float", "Shape of the random noise added at the start of each search during training."],
    ["dirichlet_eps", "Noise amount", "float", "0 turns the noise off. Training needs some noise to find new ideas."],
    ["temperature", "Temperature", "float", "How adventurous move choice is in the opening. 0 always plays the best move."],
    ["temperature_moves", "Temperature moves", "int", "How many plies use the temperature before the model plays its best move."],
    ["leaves_per_batch", "Leaves per batch", "int", "Positions from one game sent to the network together. Higher can be faster on a GPU."],
    ["virtual_loss", "Virtual loss", "float", "Keeps parallel searches from choosing the same path."],
    ["reuse_tree", "Reuse search tree", "bool", "Keep the search done for a move when the game continues."],
    ["max_plies", "Max game length (plies)", "int", "Longer games are scored as a draw."],
  ],
  train: [
    ["lr", "Learning rate", "float", "How big each learning step is."],
    ["weight_decay", "Weight decay", "float", "A small pull that keeps weights from growing too large."],
    ["batch_size", "Batch size", "int", "Positions per learning step."],
    ["games_per_iteration", "Games per iteration", "int", "Self-play games before each round of learning."],
    ["epochs_per_iteration", "Epochs per iteration", "int", "Passes over the stored games each iteration."],
    ["max_steps_per_iteration", "Max steps per iteration", "int", "Upper limit on learning steps in one iteration."],
    ["buffer_size", "Memory size (positions)", "int", "How many recent positions are kept for learning."],
    ["min_buffer", "Minimum before learning", "int", "Learning starts once this many positions exist."],
    ["concurrent_games", "Games at once", "int", "Self-play games played in parallel."],
    ["arena_games", "Match games", "int", "Games in the match between the new and the current best model."],
    ["arena_simulations", "Match simulations", "int", "Simulations per move in that match."],
    ["arena_threshold", "Score to replace best", "float", "Share of points the new model needs to become the best (0.55 = 55%)."],
  ],
};

const TABS = [["network", "Network"], ["mcts", "Tree search"], ["train", "Training"], ["ui", "Interface"], ["device", "Device"]];
// What each tab is for, shown at its top.
const TAB_INTRO = {
  network: ["Network", "The size of the network inside each model."],
  mcts: ["Tree search", "How a model thinks before it moves: it tries many possible continuations and picks the most promising one. These values are used when you play, spectate and train."],
  train: ["Training", "How a model learns: it plays games against itself, studies them, then plays a match to see if it improved. These values apply to every new training run."],
  ui: ["Interface", "How the app looks and what it shows. These changes only affect your screen."],
  device: ["Device", "The computer hardware the models run on. This is information only; it is chosen when the app starts."],
};
const LANGS = [{ value: "en", label: "English" }, { value: "ar", label: "العربية" }];

function Field({ section, spec, value, saved, limits, onChange }) {
  const { t } = useI18n();
  const [key, label, kind, help] = spec;
  const changed = String(value) !== String(saved);
  const id = `${section}-${key}`;
  if (kind === "bool") {
    return (
      <div className="switch" style={{ gridColumn: "1 / -1" }}>
        <label htmlFor={id}><b>{t(label)}{changed && <i className="changed-dot" title={t("Changed, not saved yet")} />}</b><div className="help">{t(help)}</div></label>
        <input id={id} type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />
      </div>
    );
  }
  const lim = limits?.[section]?.[key];
  const bad = value === "" || Number.isNaN(Number(value)) || (kind === "int" && !Number.isInteger(Number(value)))
    || (lim && (Number(value) < lim[0] || Number(value) > lim[1]));
  return (
    <div className="field">
      <label htmlFor={id}>{t(label)}{changed && <i className="changed-dot" title={t("Changed, not saved yet")} />}<InfoTip text={t(help)} /></label>
      <input id={id} type="number" step={kind === "int" ? 1 : "any"} min={lim?.[0]} max={lim?.[1]} value={value}
             aria-invalid={bad || undefined} style={bad ? { borderColor: "var(--bad)" } : undefined}
             onChange={(e) => onChange(e.target.value)} />
      <span className="help">{lim ? `${t("Allowed")}: ${lim[0]} ${t("to")} ${lim[1].toLocaleString("en-US")}` : ""}</span>
    </div>
  );
}

export default function Settings() {
  const { t } = useI18n();
  const toast = useToast();
  const { settings, loadSettings } = useApp();
  const [tab, setTab] = useState("mcts");
  const [draft, setDraft] = useState(null);
  const [sys, setSys] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { if (settings) setDraft(structuredClone(settings.values)); }, [settings]);
  useEffect(() => { api("/api/system/status").then(setSys).catch(() => {}); }, []);

  const dirty = useMemo(() => !!draft && !!settings && JSON.stringify(draft) !== JSON.stringify(settings.values), [draft, settings]);

  const tabChanged = (k) => !!draft && !!settings && (k === "mcts" || k === "train" || k === "ui")
    && JSON.stringify(draft[k]) !== JSON.stringify(settings.values[k]);
  const set = (section, key, v) => setDraft((d) => ({ ...d, [section]: { ...d[section], [key]: v } }));

  /** Turn the draft into numbers and keep only what changed. Returns null if something is invalid. */
  const buildPatch = () => {
    const patch = {};
    for (const section of ["mcts", "train", "ui"]) {
      for (const [k, v] of Object.entries(draft[section])) {
        if (v === settings.values[section][k]) continue;
        const spec = FIELDS[section]?.find((f) => f[0] === k);
        let out = v;
        if (spec && spec[2] !== "bool") {
          out = Number(v);
          const lim = settings.limits?.[section]?.[k];
          if (v === "" || Number.isNaN(out) || (spec[2] === "int" && !Number.isInteger(out)) || (lim && (out < lim[0] || out > lim[1]))) return null;
        }
        (patch[section] ??= {})[k] = out;
      }
    }
    return patch;
  };

  const save = async () => {
    const patch = buildPatch();
    if (!patch) { toast(t("Some values are out of range. Fields marked in red need fixing."), "error"); return; }
    setBusy(true);
    try {
      await api("/api/settings", { method: "PUT", body: patch });
      await loadSettings();
      toast(t("Saved"), "ok");
    } catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };

  const reset = async () => {
    setBusy(true);
    try { await api("/api/settings", { method: "DELETE" }); await loadSettings(); toast(t("Saved"), "ok"); }
    catch (e) { toast(e.message, "error"); } finally { setBusy(false); }
  };

  if (!draft) return <PageHead title="Settings" />;
  const sel = sys?.selection;
  const hw = sys?.hardware;

  return (
    <>
      <PageHead title="Settings" subtitle="Changes apply to the next game, match or training run. A run already in progress keeps the values it started with." />
      <div className="settings-grid">
        <nav className="settings-tabs" aria-label={t("Settings sections")}>
          {TABS.map(([k, label]) => (
            <button key={k} aria-pressed={tab === k} onClick={() => setTab(k)}>{t(label)}{tabChanged(k) && <i className="changed-dot" />}</button>
          ))}
          <button onClick={reset} disabled={busy} style={{ marginTop: 14, color: "var(--muted)", fontWeight: 500 }}>{t("Reset to defaults")}</button>
        </nav>

        <div>
          <div className="tab-intro">
            <h2>{t(TAB_INTRO[tab][0])}</h2>
            <p className="muted">{t(TAB_INTRO[tab][1])}</p>
          </div>
          {(tab === "mcts" || tab === "train") && (
            <div className="formgrid">
              {FIELDS[tab].map((spec) => (
                <Field key={spec[0]} section={tab} spec={spec} value={draft[tab][spec[0]]} saved={settings.values[tab][spec[0]]} limits={settings.limits}
                       onChange={(v) => set(tab, spec[0], v)} />
              ))}
            </div>
          )}

          {tab === "ui" && (
            <div className="stack" style={{ maxWidth: 520 }}>
              <div className="field">
                <span className="label">{t("Language")}<InfoTip text={t("The language of the whole app. Arabic switches the layout to right-to-left.")} /></span>
                <Segmented label={t("Language")} value={draft.ui.language} onChange={(v) => set("ui", "language", v)} options={LANGS} />
              </div>
              <div className="field">
                <span className="label">{t("Board theme")}</span>
                <div className="btn-row">
                  {Object.entries(BOARD_THEMES).map(([k, th]) => (
                    <button key={k} className="btn" aria-pressed={draft.ui.board_theme === k}
                            style={draft.ui.board_theme === k ? { borderColor: "var(--brass)" } : undefined}
                            onClick={() => set("ui", "board_theme", k)}>
                      <span style={{ display: "inline-flex", width: 28, height: 14, borderRadius: 3, overflow: "hidden" }}>
                        <i style={{ flex: 1, background: th.light }} /><i style={{ flex: 1, background: th.dark }} />
                      </span>
                      {th.name}
                    </button>
                  ))}
                </div>
              </div>
              <div className="switch"><label htmlFor="ui-arrows"><b>{t("Show arrows")}</b><div className="help">{t("Draw the model's last move and your hints on the board.")}</div></label>
                <input id="ui-arrows" type="checkbox" checked={draft.ui.show_arrows} onChange={(e) => set("ui", "show_arrows", e.target.checked)} /></div>
              <div className="switch"><label htmlFor="ui-eval"><b>{t("Show evaluation")}</b><div className="help">{t("Show the bar beside the board with the model's judgement.")}</div></label>
                <input id="ui-eval" type="checkbox" checked={draft.ui.show_eval} onChange={(e) => set("ui", "show_eval", e.target.checked)} /></div>
            </div>
          )}

          {tab === "network" && (
            <div className="stack" style={{ maxWidth: 560 }}>
              <p className="muted">{t("Each model has its own size, chosen when you create it on the Models page. The size below is the starter model picked when the app was set up.")}</p>
              {sel ? (
                <dl className="kv dock" style={{ padding: 18 }}>
                  <dt>{t("Starter size")}</dt><dd>{sel.network.name}</dd>
                  <dt>{t("Blocks")}</dt><dd>{sel.network.blocks}</dd>
                  <dt>{t("Filters")}</dt><dd>{sel.network.filters}</dd>
                </dl>
              ) : <p className="muted">{t("Loading...")}</p>}
              <p className="help">{t("The size and device are asked each time you start")} <code>python run.py</code> ({t("use")} <code>--auto</code> {t("to skip the questions")}).</p>
            </div>
          )}

          {tab === "device" && (
            <div className="stack" style={{ maxWidth: 560 }}>
              {sel && hw ? (
                <dl className="kv dock" style={{ padding: 18 }}>
                  <dt>{t("Device")}</dt><dd>{sys.engine?.device ?? sel.device}</dd>
                  <dt>{t("Processor")}</dt><dd>{hw.cores} {t("cores")}</dd>
                  <dt>{t("Memory")}</dt><dd>{hw.ram_total} GB ({hw.ram_avail} GB {t("free")})</dd>
                  <dt>GPU</dt><dd>{hw.gpu_available ? `${hw.gpu_name} · ${hw.vram_total} GB` : t("none found")}</dd>
                  {sys.engine?.torch && <><dt>PyTorch</dt><dd>{sys.engine.torch}</dd></>}
                  {sys.engine?.fake && <><dt>{t("Engine")}</dt><dd className="brass">{t("test engine (no PyTorch)")}</dd></>}
                </dl>
              ) : <p className="muted">{t("Loading...")}</p>}
              {hw?.gpu_note && <p className="help">{hw.gpu_note}</p>}
              <p className="help">{t("The device is chosen when you start")} <code>python run.py</code>.</p>
            </div>
          )}

          {dirty ? (
            <div className="banner" style={{ marginTop: 22, position: "sticky", bottom: 16 }}>
              <span>{t("Unsaved changes")}</span>
              <div className="btn-row">
                <button className="btn ghost" onClick={() => setDraft(structuredClone(settings.values))}>{t("Discard")}</button>
                <button className="btn primary" onClick={save} disabled={busy}>{t("Save")}</button>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </>
  );
}
