import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Activity, Boxes, Eye, Network, SlidersHorizontal, Swords } from "lucide-react";
import { SoundToggle } from "./components/Extras";
import { useI18n } from "./i18n";
import Architecture from "./pages/Architecture";
import Models from "./pages/Models";
import Play from "./pages/Play";
import Settings from "./pages/Settings";
import Spectate from "./pages/Spectate";
import Training from "./pages/Training";

const NAV = [
  ["/play", "Play", Swords],
  ["/spectate", "Spectate", Eye],
  ["/training", "Training", Activity],
  ["/models", "Models", Boxes],
  ["/architecture", "Architecture", Network],
  ["/settings", "Settings", SlidersHorizontal],
];

export default function App() {
  const { t } = useI18n();
  const { pathname } = useLocation();
  const fit = pathname.startsWith("/play") || pathname.startsWith("/spectate");      // pages that must fit the screen with no scrolling
  return (
    <div className="shell">
      <nav className="rail" aria-label="Main">
        <div className="logo" aria-hidden="true">♞</div>
        {NAV.map(([to, label, Icon]) => (
          <NavLink key={to} to={to} className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}>
            <Icon size={21} strokeWidth={1.8} />
            {t(label)}
          </NavLink>
        ))}
        <SoundToggle />
      </nav>
      <main className={`main${fit ? " fit" : ""}`}>
        <Routes>
          <Route path="/" element={<Navigate to="/play" replace />} />
          <Route path="/play" element={<Play />} />
          <Route path="/spectate" element={<Spectate />} />
          <Route path="/training" element={<Training />} />
          <Route path="/models" element={<Models />} />
          <Route path="/architecture" element={<Architecture />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/play" replace />} />
        </Routes>
      </main>
    </div>
  );
}
