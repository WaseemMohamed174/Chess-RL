// Soft chess sounds made with the Web Audio API, so there are no audio files to ship.
// Browsers only allow sound after a click or tap, so the first pointer press unlocks it.

const KEY = "sound";
let ctx = null;
let master = null;
let enabled = (() => { try { return localStorage.getItem(KEY) !== "off"; } catch { return true; } })();
let lastAt = 0;
const listeners = new Set();

function audio() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
    master = ctx.createGain();
    master.gain.value = 0.38;                       // quiet by default: easy on the ears
    const soften = ctx.createBiquadFilter();        // cuts the harsh highs of every sound
    soften.type = "lowpass"; soften.frequency.value = 3200; soften.Q.value = 0.5;
    const comp = ctx.createDynamicsCompressor();    // keeps loud moments from jumping out
    comp.threshold.value = -20; comp.ratio.value = 4;
    master.connect(soften).connect(comp).connect(ctx.destination);
  }
  if (ctx.state === "suspended") ctx.resume().catch(() => {});
  return ctx;
}

/** Call from a click handler (or let the first pointer press do it). */
export function unlockSound() { audio(); }
if (typeof window !== "undefined") {
  window.addEventListener("pointerdown", unlockSound, { once: true });
  window.addEventListener("keydown", unlockSound, { once: true });
}

export const soundOn = () => enabled;
export function setSoundOn(v) {
  enabled = !!v;
  try { localStorage.setItem(KEY, enabled ? "on" : "off"); } catch { /* private mode */ }
  listeners.forEach((f) => f(enabled));
  if (enabled) { unlockSound(); play("move"); }
}
export function onSoundChange(f) { listeners.add(f); return () => listeners.delete(f); }

function noise(c, dur) {
  const n = Math.floor(c.sampleRate * dur);
  const buf = c.createBuffer(1, n, c.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, 2);
  return buf;
}

/** Each sound is a hair different, so a long game does not sound like a machine. */
const vary = () => 1 + (Math.random() - 0.5) * 0.06;

/**
 * A soft wooden "thud": a muffled low-passed puff of air plus a short, round low note.
 * No sharp click at the start, which is what made the old sound feel harsh.
 */
function thud(c, t, { vol = 0.5, pitch = 1 } = {}) {
  const p = pitch * vary();
  const src = c.createBufferSource();
  src.buffer = noise(c, 0.07);
  const lp = c.createBiquadFilter();
  lp.type = "lowpass"; lp.frequency.value = 900 * p; lp.Q.value = 0.6;
  const g = c.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.linearRampToValueAtTime(vol * 0.55, t + 0.006);     // soft attack
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.07);
  src.connect(lp).connect(g).connect(master);
  src.start(t);

  const o = c.createOscillator();
  o.type = "sine";
  o.frequency.setValueAtTime(170 * p, t);
  o.frequency.exponentialRampToValueAtTime(110 * p, t + 0.09);
  const g2 = c.createGain();
  g2.gain.setValueAtTime(0.0001, t);
  g2.gain.linearRampToValueAtTime(vol * 0.7, t + 0.008);
  g2.gain.exponentialRampToValueAtTime(0.0001, t + 0.14);
  o.connect(g2).connect(master);
  o.start(t); o.stop(t + 0.16);
}

/** A gentle bell-like note: a sine with a quiet octave above, slow fade. */
function tone(c, t, freq, dur, vol = 0.12) {
  [[1, 1], [2, 0.18]].forEach(([mult, amp]) => {
    const o = c.createOscillator();
    o.type = "sine"; o.frequency.value = freq * mult;
    const g = c.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.linearRampToValueAtTime(vol * amp, t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(master);
    o.start(t); o.stop(t + dur + 0.02);
  });
}

/** kind: "move" | "capture" | "check" | "mate" | "start" */
export function play(kind = "move") {
  if (!enabled) return;
  const now = performance.now();
  if (kind !== "mate" && now - lastAt < 60) return;   // very fast matches should not machine-gun
  lastAt = now;
  const c = audio();
  if (!c) return;
  const t = c.currentTime + 0.005;
  if (kind === "capture") { thud(c, t, { vol: 0.7, pitch: 0.85 }); thud(c, t + 0.05, { vol: 0.3, pitch: 1.1 }); }
  else if (kind === "check") { thud(c, t, { vol: 0.5 }); tone(c, t + 0.08, 659, 0.5, 0.09); }
  else if (kind === "mate") {
    thud(c, t, { vol: 0.7, pitch: 0.8 });
    [392, 330, 262, 196].forEach((f, i) => tone(c, t + 0.25 + i * 0.3, f, 0.9, 0.13));
  } else if (kind === "start") { tone(c, t, 440, 0.45, 0.08); tone(c, t + 0.12, 587, 0.55, 0.08); }
  else thud(c, t);
}

/** Pick the sound for a move from its SAN text ("Nxe5+", "Qh5#", "e4"). */
export function playSan(san) {
  if (!san) return;
  if (san.includes("#")) play("mate");
  else if (san.includes("+")) play("check");
  else if (san.includes("x")) play("capture");
  else play("move");
}
