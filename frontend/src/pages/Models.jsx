import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Copy, Download as DownloadIcon, Pencil, Play as PlayIcon, Plus, Trash2 } from "lucide-react";
import { api } from "../api";
import PageHead from "../components/PageHead";
import CapacityDialog, { isCapacity } from "../components/CapacityDialog";
import { Empty, InfoTip, Modal, Segmented, Stat, Tip } from "../components/ui";
import { deleteModels, downloadAllUrl, reportDelete, reportStart, startAllTraining } from "../bulk";
import { useApp, useToast } from "../context";
import { useI18n } from "../i18n";
import { countParams, fmtDate, fmtInt, fmtParams } from "../utils";

const SIZES = { tiny: [3, 64], small: [6, 128], medium: [10, 192] };
const mb = (p) => (p * 4) / 1024 ** 2;
const SIZE_HELP = {
  tiny: "Tiny: the fastest to train and the weakest. A good start for trying things out.",
  small: "Small: a balance between training speed and strength.",
  medium: "Medium: the strongest of the three, but it trains slowest and needs the most memory.",
};

function CreateModal({ onClose, onDone }) {
  const { t } = useI18n();
  const toast = useToast();
  const [name, setName] = useState("");
  const [size, setSize] = useState("tiny");
  const [blocks, setBlocks] = useState(6);
  const [filters, setFilters] = useState(128);
  const [busy, setBusy] = useState(false);
  const [b, f] = size === "custom" ? [Number(blocks) || 0, Number(filters) || 0] : SIZES[size];
  const valid = name.trim() && b >= 1 && b <= 40 && f >= 8 && f <= 512;
  const params = valid ? countParams(b, f) : null;

  const submit = async () => {
    setBusy(true);
    try {
      await api("/api/models", { method: "POST", body: { name: name.trim(), size, ...(size === "custom" ? { blocks: b, filters: f } : {}) } });
      toast(t("Saved"), "ok");
      onDone();
    } catch (e) { toast(e.message, "error"); setBusy(false); }
  };

  return (
    <Modal title={t("Create model")} onClose={onClose}>
      <div className="field">
        <label htmlFor="mn">{t("Name")}</label>
        <input id="mn" type="text" autoFocus maxLength={60} value={name} onChange={(e) => setName(e.target.value)} />
      </div>
      <div className="field">
        <span className="label">{t("Size")}<InfoTip text={t("A bigger network can learn more and play better, but it trains slower and needs more memory.")} /></span>
        <Segmented label={t("Size")} value={size} onChange={setSize} options={[
          { value: "tiny", label: t("Tiny") }, { value: "small", label: t("Small") },
          { value: "medium", label: t("Medium") }, { value: "custom", label: t("Custom") }]} />
      </div>
      {size === "custom" && (
        <div className="grid2">
          <div className="field"><label htmlFor="cb">{t("Blocks")} (1-40)<InfoTip text={t("How many repeated layers the network has. More blocks means a deeper network.")} /></label>
            <input id="cb" type="number" min="1" max="40" value={blocks} onChange={(e) => setBlocks(e.target.value)} /></div>
          <div className="field"><label htmlFor="cf">{t("Filters")} (8-512)<InfoTip text={t("How wide each layer is, meaning how many patterns it can look for at once.")} /></label>
            <input id="cf" type="number" min="8" max="512" value={filters} onChange={(e) => setFilters(e.target.value)} /></div>
        </div>
      )}
      {size !== "custom" && <p className="help">{t(SIZE_HELP[size])}</p>}
      <p className="help">
        {params ? `${b} ${t("Blocks").toLowerCase()} × ${f} ${t("Filters").toLowerCase()} · ${fmtParams(params)} ${t("Parameters").toLowerCase()} · ${mb(params).toFixed(1)} MB` : t("Enter a name and a valid size.")}
      </p>
      <div className="btn-row" style={{ justifyContent: "flex-end" }}>
        <button className="btn ghost" onClick={onClose}>{t("Cancel")}</button>
        <button className="btn primary" onClick={submit} disabled={!valid || busy}>{t("Create model")}</button>
      </div>
    </Modal>
  );
}

function NameModal({ title, initial, action, onClose, onDone }) {
  const { t } = useI18n();
  const toast = useToast();
  const [name, setName] = useState(initial);
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true);
    try { await action(name.trim()); toast(t("Saved"), "ok"); onDone(); }
    catch (e) { toast(e.message, "error"); setBusy(false); }
  };
  return (
    <Modal title={title} onClose={onClose}>
      <div className="field">
        <label htmlFor="nm">{t("Name")}</label>
        <input id="nm" type="text" autoFocus maxLength={60} value={name} onChange={(e) => setName(e.target.value)}
               onKeyDown={(e) => e.key === "Enter" && name.trim() && submit()} />
      </div>
      <div className="btn-row" style={{ justifyContent: "flex-end" }}>
        <button className="btn ghost" onClick={onClose}>{t("Cancel")}</button>
        <button className="btn primary" onClick={submit} disabled={!name.trim() || busy}>{t("Save")}</button>
      </div>
    </Modal>
  );
}

export default function Models() {
  const { t } = useI18n();
  const toast = useToast();
  const { models, loadModels } = useApp();
  const [dialog, setDialog] = useState(null);   // {kind: create|rename|clone|delete, model?}
  const [training, setTraining] = useState(new Set());
  const close = () => setDialog(null);
  const done = () => { close(); loadModels(); };
  const nameOf = (id) => models.find((m) => m.id === id)?.name;

  // Which models are being trained right now (shown as a tag in the table).
  const loadTraining = useCallback(() => api("/api/training/status")
    .then((s) => setTraining(new Set((s.runs ?? []).map((r) => r.model_id))))
    .catch(() => {}), []);
  useEffect(() => {
    loadTraining();
    const i = setInterval(loadTraining, 5000);
    return () => clearInterval(i);
  }, [loadTraining]);

  // Buttons for all models at once.
  const [bulkBusy, setBulkBusy] = useState(null);   // "start" | "delete" | null
  const [cap, setCap] = useState(null);             // capacity warning for "Start all"
  const [sure, setSure] = useState(false);          // the "I understand" box of "Delete all"
  const startable = models.filter((m) => m.file_ok && !training.has(m.id) && m.population_id == null).length;   // agents are trained on the Population tab
  const downloadable = models.some((m) => m.file_ok);

  const startAll = async (force = false) => {
    setBulkBusy("start");
    try {
      const r = await startAllTraining(force);
      setCap(null);
      reportStart(r, t, toast);
      await loadTraining(); loadModels();
    } catch (e) {
      if (isCapacity(e)) setCap(e.detail);
      else { setCap(null); toast(e.message, "error"); }
    } finally { setBulkBusy(null); }
  };

  const removeAll = async () => {
    setBulkBusy("delete");
    try {
      reportDelete(await deleteModels(models.map((m) => m.id)), t, toast);
    } catch (e) { toast(e.message, "error"); }
    finally { setBulkBusy(null); setSure(false); done(); loadTraining(); }
  };

  const summary = useMemo(() => {
    if (!models.length) return null;
    const best = models.reduce((x, y) => (y.elo > x.elo ? y : x));
    return { best, iterations: models.reduce((s, m) => s + (m.iterations || 0), 0) };
  }, [models]);

  const remove = async (m) => {
    try { await api(`/api/models/${m.id}`, { method: "DELETE" }); toast(`${m.name}: ${t("Delete")}`, "ok"); done(); }
    catch (e) { toast(e.message, "error"); close(); }
  };

  const Th = ({ children, hint, num }) => (
    <th className={num ? "num" : undefined}>
      <span className="th-in">{children}{hint && <InfoTip side="bottom" align={num ? "end" : "start"} text={t(hint)} />}</span>
    </th>
  );

  return (
    <>
      <PageHead title="Models" subtitle="Every model is a trained chess player saved as a file. Play against it, train it, copy it or download it.">
        <button className="btn primary" onClick={() => setDialog({ kind: "create" })}><Plus size={16} />{t("Create model")}</button>
      </PageHead>

      {models.length === 0 ? (
        <div className="dock"><Empty title={t("No models yet")}>
          <span>{t("A model is a chess player that learns by playing itself. Create one to start.")}</span>
          <button className="btn primary" onClick={() => setDialog({ kind: "create" })}><Plus size={16} />{t("Create model")}</button>
        </Empty></div>
      ) : (
        <div className="stack" style={{ gap: 18 }}>
          <div className="dock"><section>
            <div className="stats">
              <Stat label={t("Models")} value={models.length} />
              <Stat label={t("Strongest model")} value={summary.best.name} hint={t("The model with the highest Elo.")} />
              <Stat label={t("Highest Elo")} value={fmtInt(summary.best.elo)} hint={t("Elo is a strength rating. It starts at 1000 and goes up when a model wins matches.")} />
              <Stat label={t("Training now")} value={training.size} hint={t("Models that are learning right now. Open the Training page to follow them.")} />
              <Stat label={t("Learning cycles done")} value={summary.iterations} hint={t("All finished cycles of play, learn and test, added up over every model.")} />
            </div>
          </section></div>

          <div className="btn-row" style={{ justifyContent: "flex-end" }}>
            <Tip text={t("Train every single model that is not training yet. Population agents are trained on the Population tab.")} side="bottom" align="end">
              <button className="btn small" onClick={() => startAll()} disabled={!!bulkBusy || startable === 0}>
                <PlayIcon size={14} />{bulkBusy === "start" ? t("Starting...") : `${t("Start all")} (${startable})`}
              </button>
            </Tip>
            <Tip text={t("Download every model file in one .zip")} side="bottom" align="end">
              {downloadable
                ? <a className="btn small" href={downloadAllUrl()} download><DownloadIcon size={14} />{t("Download all")}</a>
                : <button className="btn small" disabled><DownloadIcon size={14} />{t("Download all")}</button>}
            </Tip>
            <Tip text={t("Delete every model")} side="bottom" align="end">
              <button className="btn small danger" onClick={() => { setSure(false); setDialog({ kind: "delete-all" }); }} disabled={!!bulkBusy}>
                <Trash2 size={14} />{t("Delete all")}
              </button>
            </Tip>
          </div>

          <div className="tablewrap tips">
            <table>
              <thead>
                <tr>
                  <Th>{t("Name")}</Th>
                  <Th hint="Number of blocks × filters. Blocks are repeated layers (depth) and filters are how wide each layer is.">{t("Size")}</Th>
                  <Th num hint="How many numbers the network learns. More parameters can mean a stronger but slower model.">{t("Parameters")}</Th>
                  <Th num hint="How much space the model file takes on your disk, in megabytes.">MB</Th>
                  <Th num hint="Strength rating. Higher is stronger. It changes after matches and after training.">{t("Elo")}</Th>
                  <Th num hint="How many full cycles of play, learn and test this model has finished.">{t("Iteration")}</Th>
                  <Th>{t("Created")}</Th>
                  <Th>{t("Actions")}</Th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.id}>
                    <td>
                      <Tip text={t("Open this model's architecture")} side="bottom" align="start">
                        <Link to={`/architecture?model=${m.id}`}><b>{m.name}</b></Link>
                      </Tip>
                      {m.parent_id && nameOf(m.parent_id) && <span className="muted"> · {t("copied from")} {nameOf(m.parent_id)}</span>}
                      {training.has(m.id) && <> <span className="tag live">{t("training now")}</span></>}
                      {m.population_id != null && <> <span className="tag">{t("population agent")}</span></>}
                      {!m.file_ok && <> <span className="tag bad">{t("file missing")}</span></>}
                    </td>
                    <td>{m.blocks} × {m.filters}</td>
                    <td className="num">{fmtParams(m.params)}</td>
                    <td className="num">{m.size_mb.toFixed(1)}</td>
                    <td className="num"><b>{fmtInt(m.elo)}</b></td>
                    <td className="num">{m.iterations}</td>
                    <td>{fmtDate(m.created_at)}</td>
                    <td>
                      <div className="btn-row" style={{ flexWrap: "nowrap" }}>
                        <Tip text={t("Play a game against this model")} side="bottom" align="end">
                          <Link className={`btn small icon ghost${m.file_ok ? "" : " disabled"}`} to={`/play?model=${m.id}`} aria-label={t("Play")}><PlayIcon size={15} /></Link>
                        </Tip>
                        <Tip text={t("Rename")} side="bottom" align="end"><button className="btn small icon ghost" aria-label={t("Rename")} onClick={() => setDialog({ kind: "rename", model: m })}><Pencil size={15} /></button></Tip>
                        <Tip text={t("Make a copy. The copy starts with the same knowledge.")} side="bottom" align="end"><button className="btn small icon ghost" aria-label={t("Clone")} onClick={() => setDialog({ kind: "clone", model: m })}><Copy size={15} /></button></Tip>
                        <Tip text={t("Download the model file")} side="bottom" align="end">
                          {m.file_ok
                            ? <a className="btn small icon ghost" aria-label={t("Download")} href={`/api/models/${m.id}/download`} download><DownloadIcon size={15} /></a>
                            : <button className="btn small icon ghost" disabled aria-label={t("Download")}><DownloadIcon size={15} /></button>}
                        </Tip>
                        <Tip text={t("Delete this model")} side="bottom" align="end"><button className="btn small icon ghost danger" aria-label={t("Delete")} onClick={() => setDialog({ kind: "delete", model: m })}><Trash2 size={15} /></button></Tip>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {dialog?.kind === "create" && <CreateModal onClose={close} onDone={done} />}
      {dialog?.kind === "rename" && (
        <NameModal title={t("Rename")} initial={dialog.model.name} onClose={close} onDone={done}
          action={(name) => api(`/api/models/${dialog.model.id}`, { method: "PUT", body: { name } })} />
      )}
      {dialog?.kind === "clone" && (
        <NameModal title={t("Clone")} initial={`${dialog.model.name} copy`} onClose={close} onDone={done}
          action={(name) => api(`/api/models/${dialog.model.id}/clone`, { method: "POST", body: { name } })} />
      )}
      {dialog?.kind === "delete-all" && (
        <Modal title={t("Delete all models?")} onClose={close}>
          <p className="muted">{models.length} {t("models will be deleted with their model files and training history. Games they played stay in history. This cannot be undone.")}</p>
          {training.size > 0 && <p className="muted">{t("Models that are training right now are kept. Stop their training first if you want them deleted too.")}</p>}
          <p className="help">{t("Tip: use Download all first if you want a backup.")}</p>
          <label className="switch">
            <span>{t("I understand that this cannot be undone")}</span>
            <input type="checkbox" checked={sure} onChange={(e) => setSure(e.target.checked)} />
          </label>
          <div className="btn-row" style={{ justifyContent: "flex-end" }}>
            <button className="btn ghost" onClick={close} disabled={bulkBusy === "delete"}>{t("Cancel")}</button>
            <button className="btn danger" onClick={removeAll} disabled={!sure || bulkBusy === "delete"}>
              <Trash2 size={15} />{bulkBusy === "delete" ? t("Deleting...") : t("Delete all")}
            </button>
          </div>
        </Modal>
      )}
      {cap && <CapacityDialog info={cap} busy={bulkBusy === "start"} onCancel={() => setCap(null)} onForce={() => startAll(true)} />}
      {dialog?.kind === "delete" && (
        <Modal title={`${t("Delete")} ${dialog.model.name}?`} onClose={close}>
          <p className="muted">{t("The model file and its training history are removed. Games it played stay in history. This cannot be undone.")}</p>
          <div className="btn-row" style={{ justifyContent: "flex-end" }}>
            <button className="btn ghost" onClick={close}>{t("Cancel")}</button>
            <button className="btn danger" onClick={() => remove(dialog.model)}><Trash2 size={15} />{t("Delete")}</button>
          </div>
        </Modal>
      )}
    </>
  );
}
