import { useEffect, useRef, useState } from "react";
import { Volume2, VolumeX } from "lucide-react";
import { useI18n } from "../i18n";
import { onSoundChange, setSoundOn, soundOn } from "../sound";

export function SoundToggle() {
  const { t } = useI18n();
  const [on, setOn] = useState(soundOn());
  useEffect(() => onSoundChange(setOn), []);
  return (
    <button className={`sound-toggle${on ? " on" : ""}`} onClick={() => setSoundOn(!on)}
      aria-pressed={on} aria-label={t("Sound")} title={t("Sound")}>
      {on ? <Volume2 size={21} strokeWidth={1.8} /> : <VolumeX size={21} strokeWidth={1.8} />}
    </button>
  );
}

const mmss = (ms) => {
  const s = Math.max(0, Math.floor(ms / 1000));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
};

/**
 * Counts seconds.
 *  gameKey     changes -> the game timer restarts
 *  moveKey     changes -> the "this move" timer restarts
 *  sessionKey  (optional) changes -> the match timer restarts; omit it to hide that timer
 *  running     false -> every timer freezes at the value it had
 */
export function Clocks({ gameKey, moveKey, sessionKey, running }) {
  const { t } = useI18n();
  const g0 = useRef(Date.now());
  const m0 = useRef(Date.now());
  const s0 = useRef(Date.now());
  const frozen = useRef(null);
  const [, tick] = useState(0);

  useEffect(() => { g0.current = m0.current = Date.now(); frozen.current = null; tick((x) => x + 1); }, [gameKey]);
  useEffect(() => { m0.current = Date.now(); }, [moveKey]);
  useEffect(() => { s0.current = Date.now(); }, [sessionKey]);
  useEffect(() => {
    if (!running) { frozen.current ??= Date.now(); tick((x) => x + 1); return undefined; }
    frozen.current = null;
    const i = setInterval(() => tick((x) => x + 1), 250);
    return () => clearInterval(i);
  }, [running]);

  const now = frozen.current ?? Date.now();
  return (
    <div className={`clocks${running ? " live" : ""}`} role="timer" aria-label={t("Game time")}>
      <div className={`clock${running ? " live" : ""}`}><span>{t("Game time")}</span><b>{mmss(now - g0.current)}</b></div>
      <div className={`clock${running ? " live" : ""}`}><span>{t("This move")}</span><b>{Math.floor((now - m0.current) / 1000)}s</b></div>
      {sessionKey !== undefined && <div className="clock"><span>{t("Match time")}</span><b>{mmss(now - s0.current)}</b></div>}
    </div>
  );
}

/**
 * Shown over the board when a game ends by checkmate.
 *  loser  "white" | "black"   the side whose king was mated
 */
export function KingDead({ loser, onClose }) {
  const { t } = useI18n();
  return (
    <div className="kingdead" role="alertdialog" aria-live="assertive" aria-label={t("Checkmate")}>
      <div className="kingdead-card">
        <span className="kingdead-icon" aria-hidden="true">{loser === "white" ? "♔" : "♚"}</span>
        <h3>{t("Checkmate")}</h3>
        <p>{t(loser === "white" ? "The white king is dead" : "The black king is dead")}</p>
        {onClose && <button className="btn" onClick={onClose} autoFocus>{t("Close")}</button>}
      </div>
    </div>
  );
}
