const colors = {"Flat velocity":"#8a95a7","VeloEst":"#d9b869","Diff-Synth":"#ec876c","Diff-SFProxy":"#55c8b6"};
const selected = [["Flat velocity","64"],["VeloEst","zero-shot"],["Diff-Synth","5 s"],["Diff-SFProxy","5 s"]];
let paperRows = [];

function renderBars(metric) {
  for (const dataset of ["gaps", "fl"]) {
    const target = document.getElementById(`bars-${dataset}`);
    target.textContent = "";
    for (const [method, variant] of selected) {
      const row = paperRows.find(r => r.method === method && r.variant === variant);
      if (!row) continue;
      const value = Number(row[`${dataset}_r_${metric}`]);
      const item = document.createElement("div"); item.className = "bar-row";
      const label = document.createElement("span"); label.className = "bar-label"; label.textContent = method;
      const track = document.createElement("div"); track.className = "bar-track";
      const fill = document.createElement("div"); fill.className = "bar-fill"; fill.style.width = `${value * 100}%`; fill.style.background = colors[method]; track.append(fill);
      const score = document.createElement("span"); score.className = "bar-value"; score.textContent = value.toFixed(3);
      item.append(label, track, score); target.append(item);
    }
    const axis = document.createElement("div"); axis.className = "bar-axis";
    ["0", "0.25", "0.50", "0.75", "1.0"].forEach(t => { const span = document.createElement("span"); span.textContent = t; axis.append(span); });
    target.append(axis);
  }
}

function noteColor(v) {
  const stops = [[0,[85,49,133]],[32,[153,58,138]],[64,[218,88,102]],[96,[247,160,76]],[127,[255,224,119]]];
  for (let i=1;i<stops.length;i++) if (v <= stops[i][0]) {
    const a=stops[i-1], b=stops[i], t=(v-a[0])/(b[0]-a[0]);
    return `rgb(${a[1].map((c,j)=>Math.round(c+(b[1][j]-c)*t)).join(",")})`;
  }
  return "rgb(255,224,119)";
}

function renderRoll(element, notes, duration, pitchMin, pitchMax) {
  const ns = "http://www.w3.org/2000/svg", W=360,H=204,L=29,R=9,T=11,B=23;
  const x = t => L+(W-L-R)*t/duration;
  const y = p => T+(H-T-B)*(pitchMax-p)/(pitchMax-pitchMin+1);
  const svg = document.createElementNS(ns,"svg"); svg.setAttribute("viewBox",`0 0 ${W} ${H}`); svg.setAttribute("aria-hidden","true");
  for (let t=0;t<=duration;t+=5) {
    const line=document.createElementNS(ns,"line"); line.setAttribute("x1",x(t));line.setAttribute("x2",x(t));line.setAttribute("y1",T);line.setAttribute("y2",H-B);line.setAttribute("stroke","#294057");svg.append(line);
    const label=document.createElementNS(ns,"text");label.setAttribute("x",x(t)-3);label.setAttribute("y",H-6);label.setAttribute("fill","#8096ac");label.setAttribute("font-size","10");label.textContent=`${t}s`;svg.append(label);
  }
  for (let p=Math.ceil(pitchMin/12)*12;p<=pitchMax;p+=12) {
    const line=document.createElementNS(ns,"line");line.setAttribute("x1",L);line.setAttribute("x2",W-R);line.setAttribute("y1",y(p));line.setAttribute("y2",y(p));line.setAttribute("stroke","#203449");svg.append(line);
    const label=document.createElementNS(ns,"text");label.setAttribute("x","5");label.setAttribute("y",y(p)+3);label.setAttribute("fill","#8096ac");label.setAttribute("font-size","10");label.textContent=`C${p/12-1}`;svg.append(label);
  }
  for (const n of notes) {
    const rect=document.createElementNS(ns,"rect");rect.setAttribute("x",x(n.s));rect.setAttribute("y",y(n.p)-2.5);rect.setAttribute("width",Math.max(2,x(n.e)-x(n.s)));rect.setAttribute("height","5");rect.setAttribute("rx","1");rect.setAttribute("fill",noteColor(n.v));rect.setAttribute("opacity",".94");svg.append(rect);
  }
  element.replaceChildren(svg);
}

async function init() {
  const [paper, demo] = await Promise.all([
    fetch("assets/paper_results.json").then(r=>{if(!r.ok)throw new Error("Paper data unavailable");return r.json();}),
    fetch("assets/demo_notes.json").then(r=>{if(!r.ok)throw new Error("Demo data unavailable");return r.json();})
  ]);
  paperRows = paper.evaluation; renderBars("bssl");
  document.querySelectorAll("[data-metric]").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll("[data-metric]").forEach(b => {b.classList.toggle("active",b===button);b.setAttribute("aria-pressed",String(b===button));});
    renderBars(button.dataset.metric);
  }));
  const all = Object.values(demo.notes).flat(), pitchMin=Math.min(...all.map(n=>n.p))-2, pitchMax=Math.max(...all.map(n=>n.p))+2;
  for (const method of ["flat64","diffsynth","sfproxy"]) renderRoll(document.getElementById(`roll-${method}`),demo.notes[method],demo.duration_seconds,pitchMin,pitchMax);
  document.querySelectorAll("audio").forEach(audio => audio.addEventListener("play", () => document.querySelectorAll("audio").forEach(other => {if(other!==audio)other.pause();})));
}
init().catch(error => { console.error(error); document.querySelectorAll(".loading").forEach(el=>el.textContent="Visualization unavailable; audio and MIDI downloads still work."); });
