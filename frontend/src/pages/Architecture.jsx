import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import PageHead from "../components/PageHead";
import { Empty, InfoTip, ModelSelect, Stat } from "../components/ui";
import { useApp, useToast } from "../context";
import { useI18n } from "../i18n";
import { fmtInt, fmtParams } from "../utils";

const STEM = "Stem", POLICY = "Policy head", VALUE = "Value head";
const isBlock = (g) => g.startsWith("Residual block");
const GROUP_HELP = {
  [STEM]: "The first layer. It reads the board numbers and turns them into features the rest of the network can use.",
  block: "The main body of the network. Each residual block refines what the network understands about the position. More blocks means a deeper network.",
  [POLICY]: "Answers \"which move should I play?\". It gives a score to every possible move.",
  [VALUE]: "Answers \"who is winning?\". It outputs one number from -1 (Black wins) to +1 (White wins).",
};
const groupHelp = (g) => (g ? GROUP_HELP[isBlock(g) ? "block" : g] : null);
const fmtMacs = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)} G` : `${(n / 1e6).toFixed(1)} M`);

export default function Architecture() {
  const { t } = useI18n();
  const toast = useToast();
  const { models } = useApp();
  const [sp, setSp] = useSearchParams();
  const [arch, setArch] = useState(null);
  const [group, setGroup] = useState(null);      // null = all layers
  const [block, setBlock] = useState(1);

  const id = Number(sp.get("model")) || models[0]?.id || null;
  const model = models.find((m) => m.id === id);

  useEffect(() => {
    if (!id) return undefined;
    let cancelled = false;
    api(`/api/models/${id}/architecture`)
      .then((a) => { if (!cancelled) { setArch(a); setGroup(null); setBlock(1); } })
      .catch((e) => toast(e.message, "error"));
    return () => { cancelled = true; };
  }, [id, toast]);

  const stats = useMemo(() => {
    if (!arch) return null;
    const by = {};
    arch.layers.forEach((l) => { by[l.group] = (by[l.group] || 0) + l.params; });
    const tower = Object.entries(by).filter(([g]) => isBlock(g)).reduce((s, [, p]) => s + p, 0);
    return { by, tower };
  }, [arch]);

  if (models.length === 0) {
    return (<><PageHead title="Architecture" /><div className="dock"><Empty title={t("No models yet")}><span>{t("Create a model first.")}</span></Empty></div></>);
  }

  const share = (p) => (arch ? (p / arch.params) * 100 : 0);
  const rows = !arch ? [] : group ? arch.layers.filter((l) => l.group === group) : arch.layers;
  const pick = (g) => setGroup((cur) => (cur === g ? null : g));
  const towerSelected = group && isBlock(group);

  return (
    <>
      <PageHead title="Architecture" subtitle="The network is a stack of residual blocks that feeds two heads: one picks moves, the other judges the position.">
        <div style={{ minWidth: 220 }}>
          <ModelSelect id="am" models={models} value={id} onChange={(v) => setSp({ model: String(v) })} />
        </div>
      </PageHead>

      {arch && stats && (
        <div className="stack" style={{ gap: 18 }}>
          <div className="page-note">
            <span><b>{t("How to read this page")}: </b>{t("The network looks at a board position and answers two questions. The Policy head says which move to play, and the Value head says who is winning. Click any part of the diagram to see its layers in the table.")}</span>
          </div>
          <div className="dock"><section>
            <div className="stats">
              <Stat label={t("Parameters")} value={`${fmtParams(arch.params)}`} hint={t("How many numbers the network learns. More parameters can mean a stronger but slower model.")} />
              <Stat label={t("Size on disk")} value={`${arch.size_mb} MB`} hint={t("How much space the model file takes on your disk.")} />
              <Stat label={t("Blocks")} value={arch.blocks} hint={t("How many repeated layers the network has. More blocks means a deeper network.")} />
              <Stat label={t("Filters")} value={arch.filters} hint={t("How wide each layer is, meaning how many patterns it can look for at once.")} />
              <Stat label={`${t("Operations")} / ${t("position")}`} value={fmtMacs(arch.macs)} hint={t("How much computing the network does to judge one board position. Fewer is faster.")} />
            </div>
          </section></div>

          <div className="arch-grid">
            <div className="flow" aria-label={t("Network diagram")}>
              <div className="node" style={{ cursor: "default" }}>
                <span><b>{t("Input")}</b><em className="node-sub">{t("The board written as numbers")}</em></span><small>{arch.input_shape.join(" × ")}</small>
              </div>
              <div className="link" />
              <button className="node" aria-pressed={group === STEM} onClick={() => pick(STEM)}>
                <span><b>{t(STEM)}</b><em className="node-sub">{t("Reads the board")}</em></span><small>{fmtParams(stats.by[STEM])} · {share(stats.by[STEM]).toFixed(1)}%</small>
              </button>
              <div className="link" />
              <div className="towergrid" role="group" aria-label={`${arch.blocks} ${t("Residual block").toLowerCase()}`}>
                {Array.from({ length: arch.blocks }, (_, i) => i + 1).map((n) => (
                  <button key={n} aria-pressed={towerSelected && block === n} title={`${t("Residual block")} ${n}`}
                          onClick={() => { setBlock(n); setGroup(`Residual block ${n}`); }}>{n}</button>
                ))}
              </div>
              <small className="muted" style={{ marginTop: 6 }}>
                {arch.blocks} × {t("Residual block").toLowerCase()} · {fmtParams(stats.tower)} · {share(stats.tower).toFixed(1)}% <InfoTip side="top" align="center" text={t(GROUP_HELP.block)} />
              </small>
              <div className="link" />
              <div className="heads">
                <button className="node" aria-pressed={group === POLICY} onClick={() => pick(POLICY)}>
                  <b>{t(POLICY)}</b><em className="node-sub">{t("Which move to play")}</em><small>{arch.policy_size.toLocaleString("en-US")} {t("possible moves")}</small>
                  <div className="share" style={{ width: "100%", marginTop: 6 }}><i style={{ width: `${share(stats.by[POLICY])}%` }} /></div>
                  <small>{fmtParams(stats.by[POLICY])} · {share(stats.by[POLICY]).toFixed(1)}%</small>
                </button>
                <button className="node" aria-pressed={group === VALUE} onClick={() => pick(VALUE)}>
                  <b>{t(VALUE)}</b><em className="node-sub">{t("Who is winning")}</em><small>-1 … +1</small>
                  <div className="share" style={{ width: "100%", marginTop: 6 }}><i style={{ width: `${Math.max(share(stats.by[VALUE]), 0.5)}%` }} /></div>
                  <small>{fmtParams(stats.by[VALUE])} · {share(stats.by[VALUE]).toFixed(1)}%</small>
                </button>
              </div>
            </div>

            <div>
              <div className="row between" style={{ marginBottom: 10 }}>
                <h2 style={{ font: "400 22px var(--serif)" }}>{group ? t(group.replace(/ \d+$/, "")) + (group.match(/\d+$/) ? ` ${group.match(/\d+$/)[0]}` : "") : t("All layers")}</h2>
                {group && <button className="btn small ghost" onClick={() => setGroup(null)}>{t("All layers")}</button>}
              </div>
              {group ? <div className="group-note">{t(groupHelp(group))}</div>
                : <div className="group-note">{t("Every layer of the network, from input to output. Click a part of the diagram to see only its layers.")}</div>}
              <div className="tablewrap" style={{ maxHeight: 520, overflow: "auto" }}>
                <table>
                  <thead><tr>
                    <th>{t("Layers")}</th><th><span className="th-in">{t("Type")}<InfoTip text={t("The kind of operation the layer does, for example a convolution scans the board for patterns.")} /></span></th><th><span className="th-in">{t("Output shape")}<InfoTip text={t("The size of the data the layer passes on: channels × rows × columns.")} /></span></th>
                    <th className="num"><span className="th-in">{t("Parameters")}<InfoTip align="end" text={t("How many numbers this layer learns.")} /></span></th>
                  </tr></thead>
                  <tbody>
                    {rows.map((l) => (
                      <tr key={l.name}>
                        <td>{l.name}</td><td><span className="tag">{l.type}</span></td>
                        <td className="muted">{l.output_shape.join(" × ")}</td>
                        <td className="num">{l.params ? fmtInt(l.params) : "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {!model && <p className="help" style={{ marginTop: 8 }}>{t("Model not found.")}</p>}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
