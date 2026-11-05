import { AlertTriangle, Layers, Play as PlayIcon } from "lucide-react";
import { Modal } from "./ui";
import { useI18n } from "../i18n";

/** True when an api() error is the gateway's "over capacity" answer (409 with the numbers). */
export const isCapacity = (e) => e?.status === 409 && e.detail?.code === "capacity";

function Meter({ label, need, avail, t }) {
  if (avail == null) return null;
  const pct = Math.min(100, Math.round((need / Math.max(avail, 0.01)) * 100));
  return (
    <div>
      <div className="cap-head">
        <b>{label}</b>
        <span><b>{need.toFixed(1)} GB</b> {t("needed")} · {avail.toFixed(1)} GB {t("free")}</span>
      </div>
      <div className={`cap-bar${need > avail ? " over" : ""}`}><i style={{ width: `${pct}%` }} /></div>
    </div>
  );
}

/**
 * The warning window shown instead of the old hard limit. `info` is the 409 body.
 *  onForce  → send the same request again with force = true
 *  onBatch  → (optional) send it with parallel = info.fits; only shown when it would help
 */
export default function CapacityDialog({ info, busy, onCancel, onForce, onBatch }) {
  const { t } = useI18n();
  const severe = info.level === "severe";
  const canBatch = onBatch && info.fits < info.runs;
  return (
    <Modal wide tone={severe ? "cap-severe" : "cap-warn"} onClose={onCancel}
           title={severe ? t("This is far more than your computer can hold") : t("This is more than your computer can comfortably hold")}>
      <div className="cap-lead">
        <AlertTriangle size={30} aria-hidden="true" />
        <div className="stack" style={{ gap: 8 }}>
          <p>{t("You asked to train")} <b>{info.runs}</b> {t("models at the same moment.")}</p>
          <p className="muted">{severe
            ? t("The memory needed is well above what is free, so the computer may freeze, swap heavily, or the training may crash.")
            : t("The memory needed is above what is free, so training may be very slow or may fail.")}</p>
        </div>
      </div>
      <div className="cap-meters">
        <Meter label={t("Memory (RAM)")} need={info.need_ram} avail={info.avail_ram} t={t} />
        <Meter label={t("Graphics memory (VRAM)")} need={info.need_vram} avail={info.avail_vram} t={t} />
      </div>
      <p className="help">
        {t("They share one processor and one graphics card, so each model trains about")} <b>{info.slower_by}×</b>{" "}
        {t("slower than it would alone. It still works, it is only slow. That is normal, not a fault.")}
      </p>
      {canBatch && (
        <p className="help">
          {t("This computer can hold about")} <b>{info.fits}</b>{" "}
          {t("at once. Running in batches trains that many together and makes the rest wait their turn.")}
        </p>
      )}
      <div className="btn-row cap-actions">
        <button className="btn ghost" onClick={onCancel} disabled={busy}>{t("Cancel")}</button>
        {canBatch && (
          <button className="btn" onClick={() => onBatch(info.fits)} disabled={busy}>
            <Layers size={16} />{t("Run in batches")} ({info.fits} {t("at a time")})
          </button>
        )}
        <button className={`btn ${severe ? "danger" : "primary"}`} onClick={onForce} disabled={busy}>
          <PlayIcon size={16} />{t("Continue anyway")}
        </button>
      </div>
    </Modal>
  );
}
