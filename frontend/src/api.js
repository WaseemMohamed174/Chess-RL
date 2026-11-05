import { useEffect, useRef } from "react";

/** fetch wrapper: JSON in, JSON out, and a readable Error when the gateway says no. */
export async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText, raw = null;
    try {
      raw = (await res.json()).detail;
      detail = typeof raw === "string" ? raw : JSON.stringify(raw);
    } catch { /* not JSON */ }
    const err = new Error(detail);
    err.status = res.status;
    err.detail = raw;                       // the object itself, e.g. {code: "capacity", need_ram: ...}
    throw err;
  }
  if (res.status === 204) return null;
  return (res.headers.get("content-type") || "").includes("json") ? res.json() : res.text();
}

export function wsUrl(path) {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}${path}`;
}

/** Keeps one WebSocket open (and reopens it after a drop). Pass path = null to stay closed. */
export function useSocket(path, onMessage) {
  const handler = useRef(onMessage);
  handler.current = onMessage;
  useEffect(() => {
    if (!path) return undefined;
    let ws, timer, tries = 0, closed = false;
    const open = () => {
      ws = new WebSocket(wsUrl(path));
      ws.onopen = () => { tries = 0; };
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type !== "ping") handler.current(msg);
        } catch { /* ignore */ }
      };
      ws.onclose = () => {
        if (closed) return;
        timer = setTimeout(open, Math.min(1000 * 2 ** tries++, 10000));
      };
    };
    open();
    return () => { closed = true; clearTimeout(timer); ws && ws.close(); };
  }, [path]);
}
