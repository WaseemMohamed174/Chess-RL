import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { useI18n } from "./i18n";

/* ------------------------------------------------------------------ toasts */
const ToastCtx = createContext(() => {});
export const useToast = () => useContext(ToastCtx);

export function ToastProvider({ children }) {
  const [items, setItems] = useState([]);
  const id = useRef(0);
  const push = useCallback((text, kind = "info") => {
    const n = ++id.current;
    setItems((xs) => [...xs, { n, text, kind }]);
    setTimeout(() => setItems((xs) => xs.filter((x) => x.n !== n)), kind === "error" ? 7000 : 3500);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {items.map((x) => <div key={x.n} className={`toast ${x.kind}`}>{x.text}</div>)}
      </div>
    </ToastCtx.Provider>
  );
}

/* ---------------------------------------------------- settings + models cache */
const AppCtx = createContext(null);
export const useApp = () => useContext(AppCtx);

export function AppProvider({ children }) {
  const { setLang } = useI18n();
  const toast = useToast();
  const [settings, setSettings] = useState(null);   // {values, limits, defaults}
  const [models, setModels] = useState([]);

  const loadSettings = useCallback(async () => {
    const s = await api("/api/settings");
    setSettings(s);
    if (s.values.ui.language) setLang(s.values.ui.language);
    return s;
  }, [setLang]);

  const loadModels = useCallback(async () => {
    try { setModels(await api("/api/models")); } catch (e) { toast(e.message, "error"); }
  }, [toast]);

  useEffect(() => {
    loadSettings().catch((e) => toast(e.message, "error"));
    loadModels();
  }, [loadSettings, loadModels, toast]);

  const ui = settings?.values.ui ?? { board_theme: "classic", show_arrows: true, show_eval: true, language: "en" };
  return (
    <AppCtx.Provider value={{ settings, setSettings, loadSettings, models, loadModels, ui }}>
      {children}
    </AppCtx.Provider>
  );
}
