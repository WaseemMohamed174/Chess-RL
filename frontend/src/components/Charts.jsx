import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const AXIS = { stroke: "#28434a" };

function Tip({ active, payload, label, xLabel = "Iteration" }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tt">
      <div className="muted">{xLabel} {label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>{p.name}: {typeof p.value === "number" ? p.value.toFixed(3) : "-"}</div>
      ))}
    </div>
  );
}

/** lines: [{key, name, color}]; data rows need an `iteration` field. */
export function LineChartBox({ data, lines, domain, xKey = "iteration", xLabel = "Iteration" }) {
  return (
    <div className="chartbox">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
          <CartesianGrid stroke="#1f363c" vertical={false} />
          <XAxis dataKey={xKey} tickLine={false} axisLine={AXIS} />
          <YAxis tickLine={false} axisLine={false} domain={domain || ["auto", "auto"]} width={52} />
          <Tooltip content={<Tip xLabel={xLabel} />} />
          {lines.length > 1 && <Legend iconType="plainline" />}
          {lines.map((l) => (
            <Line key={l.key} type="monotone" dataKey={l.key} name={l.name} stroke={l.color}
                  strokeWidth={2} dot={data.length < 40} isAnimationActive={false} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
