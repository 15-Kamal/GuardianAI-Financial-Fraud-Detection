import { useState, useEffect, useRef } from "react";

// ── SVG arc math ─────────────────────────────────────────────
const toRad = (d) => (d * Math.PI) / 180;
const polar = (cx, cy, r, deg) => ({
  x: cx + r * Math.cos(toRad(deg)),
  y: cy + r * Math.sin(toRad(deg)),
});
const svgArc = (cx, cy, r, startDeg, sweepDeg) => {
  const s = polar(cx, cy, r, startDeg);
  const e = polar(cx, cy, r, startDeg + sweepDeg);
  const large = Math.abs(sweepDeg) > 180 ? 1 : 0;
  const dir = sweepDeg > 0 ? 1 : 0;
  return `M${s.x.toFixed(1)} ${s.y.toFixed(1)} A${r} ${r} 0 ${large} ${dir} ${e.x.toFixed(1)} ${e.y.toFixed(1)}`;
};

// ── Design tokens ─────────────────────────────────────────────
const T = {
  bg:       "#07101F",
  panel:    "#0E1A2E",
  card:     "#102030",
  border:   "#1C3050",
  accent:   "#38BDF8",
  success:  "#10B981",
  danger:   "#EF4444",
  warn:     "#F59E0B",
  text:     "#F0F6FF",
  muted:    "#607090",
  sub:      "#94A3B8",
};

// ── Probability gauge ─────────────────────────────────────────
function Gauge({ probability }) {
  const cx = 120, cy = 112, r = 82;
  const startDeg = 145, total = 250;
  const p = Math.max(0, Math.min(1, probability));
  const sweep = total * p;
  const color = p < 0.4 ? T.success : p < 0.6 ? T.warn : T.danger;
  const nb1 = polar(cx, cy, 7, (startDeg + sweep) + 90);
  const nb2 = polar(cx, cy, 7, (startDeg + sweep) - 90);
  const tip  = polar(cx, cy, r - 10, startDeg + sweep);

  return (
    <svg viewBox="0 0 240 160" style={{ width: "100%", maxWidth: 340 }}>
      {/* Zone fills (faint) */}
      <path d={svgArc(cx, cy, r, startDeg, total * 0.4)} fill="none" stroke={T.success} strokeWidth="14" strokeLinecap="round" opacity="0.18" />
      <path d={svgArc(cx, cy, r, startDeg + total * 0.4, total * 0.2)} fill="none" stroke={T.warn} strokeWidth="14" strokeLinecap="round" opacity="0.18" />
      <path d={svgArc(cx, cy, r, startDeg + total * 0.6, total * 0.4)} fill="none" stroke={T.danger} strokeWidth="14" strokeLinecap="round" opacity="0.18" />
      {/* BG track */}
      <path d={svgArc(cx, cy, r, startDeg, total)} fill="none" stroke={T.border} strokeWidth="10" strokeLinecap="round" />
      {/* Active arc */}
      {p > 0.005 && (
        <path d={svgArc(cx, cy, r, startDeg, sweep)} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round" />
      )}
      {/* Ticks */}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => {
        const a = startDeg + total * t;
        const inner = polar(cx, cy, r - 16, a);
        const outer = polar(cx, cy, r + 1, a);
        return <line key={t} x1={inner.x} y1={inner.y} x2={outer.x} y2={outer.y} stroke={T.border} strokeWidth="2.5" />;
      })}
      {/* Needle */}
      <polygon points={`${tip.x},${tip.y} ${nb1.x},${nb1.y} ${nb2.x},${nb2.y}`} fill={color} opacity="0.92" />
      <circle cx={cx} cy={cy} r="7" fill={T.panel} stroke={color} strokeWidth="2" />
      {/* Score */}
      <text x={cx} y={cy + 10} textAnchor="middle" fill={color} fontSize="34" fontFamily="monospace" fontWeight="700">
        {Math.round(p * 100)}%
      </text>
      <text x={cx} y={cy + 28} textAnchor="middle" fill={T.muted} fontSize="9" fontFamily="monospace" letterSpacing="3">
        RISK SCORE
      </text>
      {/* Labels */}
      {[["0%", startDeg], ["50%", startDeg + total * 0.5], ["100%", startDeg + total]].map(([lbl, deg]) => {
        const pt = polar(cx, cy, r + 18, deg);
        return (
          <text key={lbl} x={pt.x} y={pt.y + 3} textAnchor="middle" fill={T.muted} fontSize="8" fontFamily="monospace">
            {lbl}
          </text>
        );
      })}
    </svg>
  );
}

// ── Radar scanner ─────────────────────────────────────────────
function Scanner({ angle }) {
  const cx = 110, cy = 110, r = 82;
  const end = polar(cx, cy, r, angle);
  const TRAIL = 36;

  return (
    <svg viewBox="0 0 220 220" style={{ width: 220, height: 220 }}>
      {[1, 0.67, 0.33].map((f, i) => (
        <circle key={i} cx={cx} cy={cy} r={r * f} fill="none" stroke={T.border} strokeWidth="1" />
      ))}
      <line x1={cx - r} y1={cy} x2={cx + r} y2={cy} stroke={T.border} strokeWidth="0.5" />
      <line x1={cx} y1={cy - r} x2={cx} y2={cy + r} stroke={T.border} strokeWidth="0.5" />
      <line x1={cx - r * 0.7} y1={cy - r * 0.7} x2={cx + r * 0.7} y2={cy + r * 0.7} stroke={T.border} strokeWidth="0.5" />
      <line x1={cx + r * 0.7} y1={cy - r * 0.7} x2={cx - r * 0.7} y2={cy + r * 0.7} stroke={T.border} strokeWidth="0.5" />
      {Array.from({ length: TRAIL }, (_, i) => {
        const a = angle - i * 3;
        const pt = polar(cx, cy, r, a);
        return (
          <line key={i} x1={cx} y1={cy} x2={pt.x} y2={pt.y}
            stroke={T.accent} strokeWidth="1.5" opacity={((TRAIL - i) / TRAIL) * 0.38} />
        );
      })}
      <line x1={cx} y1={cy} x2={end.x} y2={end.y} stroke={T.accent} strokeWidth="2" opacity="0.92" />
      {[[55, 0.46], [118, 0.78], [200, 0.91], [290, 0.53], [327, 0.82]].map(([a, f], idx) => {
        const diff = ((angle - a) + 360) % 360;
        if (diff > 65) return null;
        const pt = polar(cx, cy, r * f, a);
        return <circle key={idx} cx={pt.x} cy={pt.y} r="3.5" fill={T.accent} opacity={1 - diff / 65} />;
      })}
      <circle cx={cx} cy={cy} r="4.5" fill={T.accent} />
    </svg>
  );
}

// ── Risk factor row ───────────────────────────────────────────
function RiskFactor({ factor }) {
  const MAP = {
    HIGH:   { color: T.danger, icon: "▲" },
    MEDIUM: { color: T.warn,   icon: "◆" },
    LOW:    { color: T.success, icon: "●" },
  };
  const { color, icon } = MAP[factor.severity] || { color: T.muted, icon: "—" };

  return (
    <div style={{
      display: "flex", gap: 12, alignItems: "flex-start",
      background: T.panel,
      border: `1px solid ${color}22`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 6,
      padding: "10px 14px",
      marginBottom: 7,
    }}>
      <span style={{ color, fontSize: 11, marginTop: 2 }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontFamily: "monospace", fontSize: 10, fontWeight: 700, color, letterSpacing: 1, marginBottom: 3 }}>
          {factor.factor}
        </div>
        <div style={{ fontSize: 12, color: T.sub, lineHeight: 1.5 }}>
          {factor.detail}
        </div>
      </div>
      <span style={{
        fontSize: 9, fontFamily: "monospace", color,
        background: `${color}15`, padding: "2px 8px",
        borderRadius: 3, alignSelf: "center", whiteSpace: "nowrap",
      }}>
        {factor.severity}
      </span>
    </div>
  );
}

// ── Quick presets ─────────────────────────────────────────────
const PRESETS = [
  { label: "Normal purchase",  values: { tx_amount: 85,   tx_hour: 14, tx_day: 2, cust_avg_amount: 90,  term_count: 8   } },
  { label: "Large transaction",values: { tx_amount: 9800, tx_hour: 15, tx_day: 1, cust_avg_amount: 150, term_count: 6   } },
  { label: "Late-night spike", values: { tx_amount: 3200, tx_hour: 3,  tx_day: 6, cust_avg_amount: 120, term_count: 14  } },
  { label: "Terminal abuse",   values: { tx_amount: 42,   tx_hour: 11, tx_day: 4, cust_avg_amount: 50,  term_count: 135 } },
];

// ── Input styles ──────────────────────────────────────────────
const INP = {
  width: "100%", background: "#0A1525",
  border: `1px solid ${T.border}`, borderRadius: 6,
  padding: "0 11px", color: T.text,
  fontFamily: "monospace", fontSize: 14,
  height: 44, boxSizing: "border-box", outline: "none",
};
const LBL = ({ children }) => (
  <div style={{ fontSize: 9, color: T.muted, letterSpacing: 3, marginBottom: 5, fontFamily: "monospace" }}>
    {children}
  </div>
);

// ── Main app ──────────────────────────────────────────────────
export default function GuardianAI() {
  const [form, setForm] = useState({
    tx_amount: 250, tx_hour: 14, tx_day: 3, cust_avg_amount: 200, term_count: 10,
  });
  const [scanning, setScanning] = useState(false);
  const [result,   setResult]   = useState(null);
  const [error,    setError]    = useState(null);
  const [angle,    setAngle]    = useState(0);
  const animRef = useRef(null);

  const ratio    = form.cust_avg_amount > 0 ? (form.tx_amount / form.cust_avg_amount).toFixed(2) : "1.00";
  const ratioNum = parseFloat(ratio);
  const ratioCol = ratioNum > 3 ? T.danger : ratioNum > 1.5 ? T.warn : T.success;

  // Scanner animation
  useEffect(() => {
    if (scanning) {
      const loop = () => { setAngle((a) => (a + 3) % 360); animRef.current = requestAnimationFrame(loop); };
      animRef.current = requestAnimationFrame(loop);
    } else {
      cancelAnimationFrame(animRef.current);
    }
    return () => cancelAnimationFrame(animRef.current);
  }, [scanning]);

  const applyPreset = (p) => { setForm(p.values); setResult(null); setError(null); };

  const analyze = async () => {
    setScanning(true); setResult(null); setError(null);

    // Auto-detects environment:
    // - npm start (port 3001) → calls Flask on port 5000 directly
    // - Flask serving build (port 5000) → uses relative /predict
    const API = (window.location.port === "3000" || window.location.port === "3001")
      ? "http://localhost:5000/predict"
      : "/predict";

    const minDelay = new Promise((r) => setTimeout(r, 1600));
    try {
      const res = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tx_amount:       form.tx_amount,
          tx_hour:         form.tx_hour,
          tx_day:          form.tx_day,
          cust_avg_amount: form.cust_avg_amount,
          term_count:      form.term_count,
        }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const parsed = await res.json();
      await minDelay;
      setResult(parsed);
    } catch (err) {
      await minDelay;
      setError("Cannot reach Flask backend — make sure python app.py is running on port 5000.");
    } finally {
      setScanning(false);
    }
  };

  const vCol        = result?.verdict === "BLOCKED" ? T.danger : T.success;
  const statusLabel = scanning ? "ANALYZING" : result ? (result.verdict === "BLOCKED" ? "THREAT DETECTED" : "CLEARED") : "STANDBY";
  const statusCol   = scanning ? T.warn : result ? vCol : T.success;

  return (
    <div style={{
      background: T.bg, color: T.text, minHeight: "100vh", height: "100vh",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      display: "flex", flexDirection: "column", overflow: "hidden",
    }}>
      <style>{`
        @keyframes blink  { 0%,100%{opacity:1} 50%{opacity:0.2} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        input[type=range]{appearance:none;-webkit-appearance:none;height:3px;background:#1C3050;border-radius:2px;padding:0;border:none;cursor:pointer;}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#38BDF8;cursor:pointer;}
        input:focus,select:focus{border-color:#38BDF8!important;outline:none;}
        select option{background:#0E1A2E;}
        ::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:#07101F;}::-webkit-scrollbar-thumb{background:#1C3050;border-radius:2px;}
      `}</style>

      {/* ── Header ───────────────────────────────────────── */}
      <div style={{
        background: T.panel, borderBottom: `1px solid ${T.border}`,
        padding: "0 20px", height: 52, display: "flex", alignItems: "center", gap: 12, flexShrink: 0,
      }}>
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke={T.accent} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        </svg>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 700, letterSpacing: 3 }}>
            <span style={{ color: T.accent }}>GUARDIAN</span>AI
          </div>
          <div style={{ width: 1, height: 16, background: T.border }} />
          <div style={{ fontFamily: "monospace", fontSize: 10, color: T.muted, letterSpacing: 3 }}>
            FINANCIAL THREAT MONITOR v2.0
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 7, height: 7, borderRadius: "50%", background: statusCol,
            animation: scanning ? "blink 1s infinite" : "none",
          }} />
          <span style={{ fontFamily: "monospace", fontSize: 10, color: T.muted, letterSpacing: 3 }}>
            {statusLabel}
          </span>
        </div>
      </div>

      {/* ── Body ─────────────────────────────────────────── */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* ── LEFT: Form (50%) ── */}
        <div style={{
          width: "50%", minWidth: "50%", background: T.panel,
          borderRight: `1px solid ${T.border}`,
          padding: "20px 28px",
          display: "flex", flexDirection: "column",
          overflow: "hidden",
        }}>

          {/* Presets */}
          <LBL>QUICK SCENARIOS</LBL>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8, marginBottom: 16 }}>
            {PRESETS.map((p) => (
              <button key={p.label} onClick={() => applyPreset(p)}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = T.accent; e.currentTarget.style.color = T.text; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = T.border; e.currentTarget.style.color = T.sub; }}
                style={{
                  background: T.card, border: `1px solid ${T.border}`, borderRadius: 5,
                  padding: "9px 10px", cursor: "pointer", color: T.sub,
                  fontSize: 11, fontFamily: "monospace", textAlign: "center",
                  lineHeight: 1.3, transition: "all 0.15s",
                }}>
                {p.label}
              </button>
            ))}
          </div>

          <div style={{ borderTop: `1px solid ${T.border}`, marginBottom: 16 }} />
          <LBL>↳ TRANSACTION DETAILS</LBL>

          {/* Fields — grow to fill height */}
          <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 1fr", gridTemplateRows: "1fr 1fr 1fr", gap: "0 24px", alignItems: "start" }}>

            {/* Amount */}
            <div>
              <LBL>TRANSACTION AMOUNT</LBL>
              <div style={{ position: "relative" }}>
                <span style={{ position: "absolute", left: 11, top: "50%", transform: "translateY(-50%)", color: T.muted, fontFamily: "monospace", fontSize: 13 }}>$</span>
                <input type="number" value={form.tx_amount} min={0} max={99999} step={10}
                  onChange={(e) => setForm((f) => ({ ...f, tx_amount: +e.target.value }))}
                  style={{ ...INP, paddingLeft: 24 }} />
              </div>
            </div>

            {/* Hour */}
            <div>
              <LBL>TIME OF DAY — {String(form.tx_hour).padStart(2,"0")}:00</LBL>
              <div style={{ height: 44, display: "flex", flexDirection: "column", justifyContent: "center" }}>
                <input type="range" min={0} max={23} value={form.tx_hour}
                  onChange={(e) => setForm((f) => ({ ...f, tx_hour: +e.target.value }))}
                  style={{ width: "100%" }} />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: T.muted, fontFamily: "monospace", marginTop: 4 }}>
                  <span>midnight</span><span>noon</span><span>23:00</span>
                </div>
              </div>
            </div>

            {/* Day */}
            <div>
              <LBL>DAY OF WEEK</LBL>
              <select value={form.tx_day}
                onChange={(e) => setForm((f) => ({ ...f, tx_day: +e.target.value }))}
                style={{ ...INP, cursor: "pointer" }}>
                {["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].map((d, i) => (
                  <option key={i} value={i}>{d}</option>
                ))}
              </select>
            </div>

            {/* Avg spend */}
            <div>
              <LBL>CUSTOMER AVG SPEND</LBL>
              <div style={{ position: "relative" }}>
                <span style={{ position: "absolute", left: 11, top: "50%", transform: "translateY(-50%)", color: T.muted, fontFamily: "monospace", fontSize: 13 }}>$</span>
                <input type="number" value={form.cust_avg_amount} min={1} step={10}
                  onChange={(e) => setForm((f) => ({ ...f, cust_avg_amount: +e.target.value }))}
                  style={{ ...INP, paddingLeft: 24 }} />
              </div>
            </div>

            {/* Terminal count — spans both columns */}
            <div style={{ gridColumn: "1 / -1" }}>
              <LBL>TERMINAL DAILY COUNT</LBL>
              <input type="number" value={form.term_count} min={0} max={999}
                onChange={(e) => setForm((f) => ({ ...f, term_count: +e.target.value }))}
                style={{ ...INP }} />
            </div>

          </div>{/* end fields grid */}

          {/* Bottom row — ratio + button, both 44px */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, paddingTop: 16, borderTop: `1px solid ${T.border}`, marginTop: 16 }}>

            {/* Spending ratio — 44px compact */}
            <div style={{
              height: 44,
              background: `${ratioCol}0C`,
              border: `1px solid ${ratioCol}28`,
              borderRadius: 7,
              display: "flex", alignItems: "center",
              padding: "0 14px", gap: 10,
            }}>
              <div style={{ fontFamily: "monospace", fontSize: 20, fontWeight: 700, color: ratioCol, lineHeight: 1, whiteSpace: "nowrap" }}>
                {ratio}<span style={{ fontSize: 12 }}>×</span>
              </div>
              <div style={{ fontSize: 9, color: ratioCol, fontFamily: "monospace", letterSpacing: 1, lineHeight: 1.3 }}>
                {ratioNum > 3 ? "▲ HIGH" : ratioNum > 1.5 ? "◆ ELEVATED" : "● NORMAL"}
                <div style={{ fontSize: 8, opacity: 0.7, marginTop: 2 }}>SPENDING RATIO</div>
              </div>
            </div>

            {/* Analyze button — 44px */}
            <button onClick={analyze} disabled={scanning} style={{
              height: 44, borderRadius: 7,
              fontFamily: "monospace", fontSize: 13, fontWeight: 700, letterSpacing: 3,
              cursor: scanning ? "not-allowed" : "pointer",
              border: "none", width: "100%",
              background: scanning ? T.border : T.accent,
              color: scanning ? T.muted : "#04091A",
              transition: "background 0.2s, color 0.2s",
            }}>
              {scanning ? "◉  SCANNING..." : "▶  RUN ANALYSIS"}
            </button>

          </div>
        </div>

        {/* ── RIGHT: Results (50%) ── */}
        <div style={{
          width: "50%", display: "flex", alignItems: "center", justifyContent: "center",
          padding: 32, overflowY: "auto",
        }}>

          {/* Empty state */}
          {!scanning && !result && !error && (
            <div style={{ textAlign: "center" }}>
              <svg viewBox="0 0 24 24" width="52" height="52" fill="none" stroke={T.border} strokeWidth="1.2" style={{ marginBottom: 16 }}>
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
              <div style={{ fontFamily: "monospace", fontSize: 10, color: T.muted, letterSpacing: 4 }}>
                AWAITING INPUT
              </div>
              <div style={{ fontSize: 12, color: T.border, marginTop: 10, maxWidth: 250, lineHeight: 1.6 }}>
                Set the transaction parameters and click Run Analysis to scan for fraud patterns.
              </div>
            </div>
          )}

          {/* Scanner animation */}
          {scanning && (
            <div style={{ textAlign: "center", animation: "fadeUp 0.3s ease" }}>
              <Scanner angle={angle} />
              <div style={{ fontFamily: "monospace", fontSize: 9, color: T.accent, letterSpacing: 5, marginTop: 10, animation: "blink 1.8s infinite" }}>
                ANALYZING PATTERNS
              </div>
            </div>
          )}

          {/* Error */}
          {!scanning && error && (
            <div style={{
              background: `${T.danger}0F`, border: `1px solid ${T.danger}`,
              borderRadius: 8, padding: "16px 24px", color: T.danger,
              fontFamily: "monospace", fontSize: 12, textAlign: "center", maxWidth: 380,
            }}>
              ▲ {error}
            </div>
          )}

          {/* Results panel */}
          {!scanning && result && (
            <div style={{ width: "100%", maxWidth: 560, animation: "fadeUp 0.4s ease" }}>

              {/* Gauge */}
              <div style={{ display: "flex", justifyContent: "center", marginBottom: 14 }}>
                <Gauge probability={result.fraud_probability} />
              </div>

              {/* Verdict banner */}
              <div style={{
                background: `${vCol}0E`,
                border: `1px solid ${vCol}`,
                borderRadius: 8, padding: "13px 18px",
                display: "flex", alignItems: "center", gap: 14, marginBottom: 18,
              }}>
                <div style={{
                  width: 40, height: 40, borderRadius: "50%",
                  background: `${vCol}18`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 20, flexShrink: 0,
                }}>
                  {result.verdict === "BLOCKED" ? "✖" : "✔"}
                </div>
                <div>
                  <div style={{ fontFamily: "monospace", fontSize: 14, fontWeight: 700, color: vCol, letterSpacing: 2 }}>
                    {result.verdict === "BLOCKED" ? "TRANSACTION BLOCKED" : "TRANSACTION APPROVED"}
                  </div>
                  <div style={{ fontSize: 11, color: T.muted, marginTop: 3 }}>
                    Random Forest v2.0 · Confidence: {result.confidence} · Threshold: 60%
                  </div>
                </div>
              </div>

              {/* Risk factors */}
              <LBL>↳ RISK ASSESSMENT</LBL>
              <div style={{ marginBottom: 12 }}>
                {result.risk_factors?.map((rf, i) => <RiskFactor key={i} factor={rf} />)}
              </div>

              {/* Model insight */}
              <div style={{
                background: T.panel, border: `1px solid ${T.border}`,
                borderRadius: 6, padding: "11px 14px", marginBottom: 12,
              }}>
                <LBL>↳ MODEL OUTPUT</LBL>
                <div style={{ display: "flex", gap: 20 }}>
                  <div>
                    <div style={{ fontSize: 10, color: T.muted, fontFamily: "monospace" }}>FRAUD PROB</div>
                    <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "monospace", color: vCol }}>
                      {(result.fraud_probability * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: T.muted, fontFamily: "monospace" }}>LEGIT PROB</div>
                    <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "monospace", color: T.success }}>
                      {(result.legitimate_probability * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: T.muted, fontFamily: "monospace" }}>SPEND RATIO</div>
                    <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "monospace", color: T.accent }}>
                      {result.spending_ratio}×
                    </div>
                  </div>
                </div>
              </div>

              {/* Reset */}
              <button onClick={() => { setResult(null); setError(null); }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = T.accent; e.currentTarget.style.color = T.text; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = T.border; e.currentTarget.style.color = T.muted; }}
                style={{
                  width: "100%", padding: "9px 0",
                  background: "transparent", border: `1px solid ${T.border}`,
                  borderRadius: 6, color: T.muted, fontFamily: "monospace",
                  fontSize: 10, letterSpacing: 3, cursor: "pointer",
                  transition: "all 0.15s",
                }}>
                ↺ CLEAR &amp; RESET
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}