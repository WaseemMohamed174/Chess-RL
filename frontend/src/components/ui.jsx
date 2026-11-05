import { Chess } from "chess.js";
import { Info } from "lucide-react";
import { fmtSecs } from "../utils";
import { useI18n } from "../i18n";

export function Empty({ title, children }) {
  const { t } = useI18n();
  return <div className="empty"><strong>{t(title)}</strong>{children}</div>;
}

/**
 * Small "i" icon that explains something on hover or keyboard focus.
 * Use it only where a label alone is not enough.
 *  side   "bottom" | "top"      which way the bubble opens
 *  align  "start" | "end" | "center"
 */
export function InfoTip({ text, side = "bottom", align = "start", label }) {
  return (
    <span className={`tip ${side} ${align}`}>
      <button type="button" className="tip-btn" aria-label={label || text}><Info size={14} strokeWidth={2} /></button>
      <span className="tip-body" role="tooltip">{text}</span>
    </span>
  );
}

/** Wraps any element so hovering it shows a short bubble. */
export function Tip({ text, children, side = "top", align = "center", block = false }) {
  return (
    <span className={`tip wrap ${side} ${align}${block ? " block" : ""}`}>
      {children}
      <span className="tip-body" role="tooltip">{text}</span>
    </span>
  );
}

/** Section heading with an optional one-line explanation behind an info icon. */
export function SectionTitle({ children, help, right }) {
  return (
    <div className="sect-title">
      <h2>{children}</h2>
      {help && <InfoTip text={help} />}
      {right && <span className="sect-right">{right}</span>}
    </div>
  );
}

/** A number with a clear name, an optional unit, and an optional hover explanation. */
export function Stat({ label, value, unit, hint }) {
  return (
    <div className="stat">
      <b>{value}{unit && <small>{unit}</small>}</b>
      <span>{label}{hint && <InfoTip text={hint} side="top" />}</span>
    </div>
  );
}

export function Segmented({ value, onChange, options, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={value === o.value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ModelSelect({ models, value, onChange, exclude, label, id }) {
  const { t } = useI18n();
  return (
    <div className="field">
      {label && <label htmlFor={id}>{label}</label>}
      <select id={id} value={value ?? ""} onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}>
        {value == null && <option value="">{t("Select a model")}</option>}
        {models.filter((m) => m.id !== exclude).map((m) => (
          <option key={m.id} value={m.id}>{m.name} · {Math.round(m.elo)} Elo</option>
        ))}
      </select>
    </div>
  );
}

export function Modal({ title, children, onClose, wide = false, tone = "" }) {
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className={`modal${wide ? " wide" : ""}${tone ? ` ${tone}` : ""}`} role="dialog" aria-modal="true" aria-label={title}>
        <h2>{title}</h2>
        {children}
      </div>
    </div>
  );
}

/**
 * Vertical bar showing who the model thinks is winning.
 * `value` is -1 .. +1 from White's side. Raw values from a young model sit close to 0,
 * so the bar is stretched (tanh) to make small leads visible; the number is the
 * model's win chance for the side that leads (50% = equal).
 */
export function EvalBar({ value, orientation = "white", height }) {
  const v = Math.max(-1, Math.min(1, value ?? 0));
  const whiteShare = 50 + 50 * Math.tanh(2.5 * v);
  const whiteAtBottom = orientation === "white";
  const style = whiteAtBottom ? { height: `${whiteShare}%`, bottom: 0 } : { height: `${whiteShare}%`, top: 0 };
  const whiteLeads = v >= 0;
  const labelAtBottom = whiteLeads === whiteAtBottom;
  const chance = Math.round(50 + Math.abs(v) * 50);
  const who = Math.abs(v) < 0.02 ? "equal" : whiteLeads ? "White" : "Black";
  return (
    <div className="evalbar" style={{ height }} role="img" tabIndex={0}
         aria-label={`Model thinks ${who} wins with ${chance}% chance`}
         title={`The model's guess of who is winning. ${who === "equal" ? "It sees the position as equal." : `${who} has about a ${chance}% chance to win.`} The bar is stretched so small leads are easy to see.`}>
      <div className="white" style={style} />
      <div className="mid" />
      <b className={whiteLeads ? "on-white" : "on-dark"} style={labelAtBottom ? { bottom: 0 } : { top: 0 }}>
        {chance}%
      </b>
    </div>
  );
}

/**
 * Move list. Click a move to look at the board at that moment.
 *  san       list of SAN moves
 *  secs      optional list: seconds each move took
 *  selected  index (0-based ply) being looked at, or null when following the live game
 *  onSelect  (ply index) => void
 */
export function MoveList({ san, secs, selected = null, onSelect, fill = false }) {
  if (!san?.length) return <p className="muted">No moves yet.</p>;
  const rows = [];
  for (let i = 0; i < san.length; i += 2) rows.push(i);
  const cur = selected ?? san.length - 1;
  const cell = (i) => {
    if (san[i] == null) return <span />;
    const clickable = !!onSelect;
    const Tag = clickable ? "button" : "span";
    return (
      <Tag type={clickable ? "button" : undefined} className={`m ${i === cur ? "now" : ""}`}
           onClick={clickable ? () => onSelect(i) : undefined}
           aria-current={i === cur ? "true" : undefined}>
        <span>{san[i]}</span>{secs?.[i] != null && <small>{fmtSecs(secs[i])}</small>}
      </Tag>
    );
  };
  return (
    <div className={`moves${fill ? " fill" : ""}`} aria-label="Move list">
      {rows.map((i) => (
        <MoveRow key={i} n={i / 2 + 1} w={cell(i)} b={cell(i + 1)} />
      ))}
    </div>
  );
}
function MoveRow({ n, w, b }) {
  return (
    <>
      <span className="n">{n}.</span>
      {w}
      {b}
    </>
  );
}

/** One player's card: icon, name, kind (model/person), colour of pieces, and whose turn it is. */
export function PlayerCard({ name, kind, detail, color, active, turnText }) {
  return (
    <div className={`pcard${active ? " active" : ""}`}>
      <span className={`pcolor ${color}`} aria-hidden="true" title={color === "white" ? "Plays the light pieces" : "Plays the dark pieces"}>{color === "white" ? "♔" : "♚"}</span>
      <div className="pinfo">
        <b>{name}</b>
        <small>{kind}{detail ? ` · ${detail}` : ""}</small>
      </div>
      {active && <span className="pturn">{turnText}</span>}
    </div>
  );
}

/** The model's top candidate moves for one position (from its tree search). */
export function Thoughts({ fen, top, limit }) {
  if (!top?.length) return <p className="muted">Nothing yet. Ask for a hint or make a move.</p>;
  const total = top.reduce((s, t) => s + t.visits, 0) || 1;
  const toSan = (uci) => {
    if (!fen) return uci;                 // no position to name moves in: show e2e4 style
    try {
      return new Chess(fen).move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] }).san;
    } catch { return uci; }
  };
  return (
    <div>
      {(limit ? top.slice(0, limit) : top).map((t) => (
        <div className="thought" key={t.uci} title={`Prior ${(t.prior * 100).toFixed(0)}% · visits ${t.visits} · value ${t.q.toFixed(2)}`}>
          <span className="m">{toSan(t.uci)}</span>
          <div className="bar"><i style={{ width: `${(t.visits / total) * 100}%` }} /></div>
          <small>{Math.round((t.visits / total) * 100)}%</small>
        </div>
      ))}
    </div>
  );
}
