import { useEffect, useState } from "react";
import { api, useSocket } from "../api";
import { useI18n } from "../i18n";

function EnginePill() {
  const { t } = useI18n();
  const [workers, setWorkers] = useState(null);
  useEffect(() => {
    api("/api/system/status").then((s) => setWorkers(s.workers)).catch(() => setWorkers({}));
  }, []);
  useSocket("/ws/system", (m) => m.type === "workers" && setWorkers(m.workers));
  const st = workers?.play?.state;
  const [cls, text] = st === "READY" ? ["ok", "Engine ready"]
    : st === "FAILED" || st === "CRASHED" ? ["bad", "Engine offline"]
    : ["warn", "Engine starting"];
  return <span className={`pill ${cls}`}><i />{t(text)}</span>;
}

/** Title block shared by every page, with the engine status on the right. */
export default function PageHead({ title, subtitle, children }) {
  const { t } = useI18n();
  return (
    <header className="topbar">
      <div>
        <h1 className="title">{t(title)}</h1>
        {subtitle && <p className="subtitle">{t(subtitle)}</p>}
      </div>
      <div className="row">{children}<EnginePill /></div>
    </header>
  );
}
