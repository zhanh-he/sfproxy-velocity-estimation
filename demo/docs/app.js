const colors = {"Flat velocity":"#8a95a7","VeloEst":"#d9b869","Diff-Synth":"#ec876c","Diff-SFProxy":"#55c8b6"};
const selected = [["Flat velocity","64"],["VeloEst","zero-shot"],["Diff-Synth","5 s"],["Diff-SFProxy","5 s"]];
let paperRows = [];
let demoRows = null;
let selectedNoteIndex = 0;
let cases = [];
let activeCase = null;
let loadVersion = 0;
const methods = [
  {key:"reference", number:"00", title:"Ground truth", kind:"ORIGINAL RECORDING", description:"The real performance recording. For guitar, the aligned score has note times but no verified note velocities."},
  {key:"flat64", number:"01", title:"Flat 64", kind:"CONTROL", description:"The aligned notes rendered with one constant velocity."},
  {key:"diffsynth", number:"02", title:"Diff-Synth", kind:"WAVEFORM BASELINE", description:"Velocities adapted through a frozen differentiable synthesizer."},
  {key:"sfproxy", number:"03", title:"Diff-SFProxy", kind:"NOTE-WISE PROXY", description:"Velocities adapted through harmonic energy and onset flux."},
];

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

function renderRoll(element, notes, duration, pitchMin, pitchMax, neutral=false) {
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
  for (const [index,n] of notes.entries()) {
    const rect=document.createElementNS(ns,"rect");rect.setAttribute("x",x(n.s));rect.setAttribute("y",y(n.p)-2.5);rect.setAttribute("width",Math.max(2,x(n.e)-x(n.s)));rect.setAttribute("height","5");rect.setAttribute("rx","1");rect.setAttribute("fill",neutral?"#8398ab":noteColor(n.v));rect.setAttribute("opacity",".94");rect.setAttribute("data-note-index",String(index));rect.addEventListener("mouseenter",()=>selectNote(index,false));rect.addEventListener("click",()=>selectNote(index,true));svg.append(rect);
  }
  element.replaceChildren(svg);
}

function pitchName(pitch) {
  const names=["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"];
  return `${names[pitch%12]}${Math.floor(pitch/12)-1}`;
}

function selectNote(index, seekAudio) {
  if (!demoRows || !activeCase) return;
  const notes=demoRows.notes.reference;
  selectedNoteIndex=Math.max(0,Math.min(notes.length-1,index));
  const note=notes[selectedNoteIndex];
  document.getElementById("selected-note").textContent=`${pitchName(note.p)} · note ${selectedNoteIndex+1} of ${notes.length}`;
  document.getElementById("selected-time").textContent=`${note.s.toFixed(2)}–${note.e.toFixed(2)} s in this excerpt · MIDI pitch ${note.p}`;
  const values=document.getElementById("selected-velocities");
  values.replaceChildren();
  for (const method of methods) {
    const value=method.key==="reference" && !activeCase.velocityGroundTruth ? null : demoRows.notes[method.key]?.[selectedNoteIndex]?.v;
    const cell=document.createElement("div");cell.className="inspector-value";
    const name=document.createElement("span");name.textContent=method.title;
    const number=document.createElement("strong");number.textContent=value==null?"—":String(value);if(value!=null)number.style.color=noteColor(value);
    const delta=document.createElement("small");delta.textContent=value==null?(method.key==="reference"?"velocity unavailable":"awaiting prediction"):(method.key==="reference"?"reference":(activeCase.velocityGroundTruth?`|Δ| ${Math.abs(value-note.v)}`:"predicted velocity"));
    cell.append(name,number,delta);values.append(cell);
  }
  document.querySelectorAll("[data-note-index]").forEach(rect=>rect.classList.toggle("is-selected",Number(rect.dataset.noteIndex)===selectedNoteIndex));
  if (seekAudio) document.querySelectorAll("audio").forEach(audio=>{audio.currentTime=note.s;});
}

function renderCard(method, item, notes) {
  const ready=item.available.includes(method.key);
  const card=document.createElement("article");
  card.className=`demo-card ${method.key}${method.key==="sfproxy"?" featured":""}${ready?"":" pending"}`;
  const top=document.createElement("div");top.className="card-top";
  const index=document.createElement("span");index.className="card-index";index.textContent=`${method.number} / ${method.kind}`;
  const badge=document.createElement("span");badge.className="score";
  const mae=item.metrics?.[method.key] ?? notes?.stats?.[method.key]?.mae_20s;
  if (mae!=null && method.key!=="reference") {
    badge.textContent="MAE ";const strong=document.createElement("strong");strong.textContent=Number(mae).toFixed(1);badge.append(strong);
  } else badge.textContent=ready?(method.key==="reference"?"Recorded":"20 s"):(item.pending?.[method.key]||"Awaiting 5 s");
  top.append(index,badge);
  const title=document.createElement("h3");title.textContent=method.title;
  if (method.key==="sfproxy") {const mark=document.createElement("span");mark.textContent="proposed";title.append(" ",mark);}
  const description=document.createElement("p");description.textContent=method.key==="reference"?(item.velocityGroundTruth?"The original performance recording and its measured note velocities.":"The original performance recording. The aligned guitar score gives note times, but no verified velocity labels."):method.description;
  card.append(top,title,description);
  if (ready) {
    const roll=document.createElement("div");roll.className="roll";roll.setAttribute("role","img");roll.setAttribute("aria-label",`${method.title} aligned MIDI notes`);
    const methodNotes=notes?.notes?.[method.key];
    if (methodNotes?.length) {
      const all=Object.values(notes.notes).flat();
      const lo=Math.min(...all.map(n=>n.p))-2,hi=Math.max(...all.map(n=>n.p))+2;
      renderRoll(roll,methodNotes,20,lo,hi,method.key==="reference"&&!item.velocityGroundTruth);
    } else roll.textContent="Aligned MIDI view unavailable";
    const audioBox=document.createElement("div");audioBox.className="card-audio";
    const audioLabel=document.createElement("span");audioLabel.textContent=method.key==="reference"?"Original recording":"SoundFont resynthesis";
    const audio=document.createElement("audio");audio.controls=true;audio.preload="metadata";audio.src=`${item.assetBase}/${method.key}.mp3`;audio.setAttribute("aria-label",`${method.title} audio for ${item.title}`);
    audio.addEventListener("play",()=>document.querySelectorAll("audio").forEach(other=>{if(other!==audio)other.pause();}));
    audioBox.append(audioLabel,audio);
    const download=document.createElement("a");download.className="download";download.href=`${item.assetBase}/${method.key}.mid`;download.download="";download.textContent="Download aligned MIDI ↓";
    card.append(roll,audioBox,download);
  } else {
    const placeholder=document.createElement("div");placeholder.className="pending-body";
    const symbol=document.createElement("span");symbol.textContent="◌";
    const message=document.createElement("strong");message.textContent=item.pending?.[method.key]||"5 s prediction pending";
    const detail=document.createElement("small");detail.textContent=method.key==="reference"?"No recording is hosted here; see the dataset source and terms below.":method.key==="flat64"&&item.dataset==="GAPS"?"A matched render can be added after dataset publication permission is obtained.":"This slot will use the same 20-second score and time window once the selected run is recovered.";
    placeholder.append(symbol,message,detail);card.append(placeholder);
  }
  return card;
}

function renderDatasetTabs(dataset) {
  const target=document.getElementById("dataset-tabs");target.replaceChildren();
  for (const name of ["GAPS","FL","MAESTRO","SMD"]) {
    const button=document.createElement("button");button.type="button";button.className=name===dataset?"active":"";
    button.setAttribute("aria-pressed",String(name===dataset));button.textContent=name;
    button.addEventListener("click",()=>selectDataset(name));target.append(button);
  }
}

function selectDataset(dataset) {
  renderDatasetTabs(dataset);
  const select=document.getElementById("case-select");select.replaceChildren();
  for (const item of cases.filter(c=>c.dataset===dataset)) {
    const option=document.createElement("option");option.value=item.id;option.textContent=item.selectorLabel;select.append(option);
  }
  loadCase(select.value);
}

async function loadCase(id) {
  const item=cases.find(c=>c.id===id);
  if (!item)return;
  const version=++loadVersion;
  document.querySelectorAll("audio").forEach(audio=>audio.pause());
  activeCase=item;demoRows=null;
  document.getElementById("case-select").value=id;
  document.getElementById("case-meta").textContent=`${item.dataset} · ${item.instrument} · ${item.window}`;
  document.getElementById("case-title").textContent=item.title;
  document.getElementById("case-description").textContent=item.description;
  document.getElementById("case-coverage").textContent=`${item.available.length} / 4 audio comparisons ready`;
  document.getElementById("legend-note").textContent=item.velocityGroundTruth?"Select any note to compare velocities":"Guitar reference score has no velocity ground truth";
  const provenance=document.getElementById("case-provenance");provenance.textContent=item.provenance;
  if (item.sourceUrl) {const link=document.createElement("a");link.href=item.sourceUrl;link.textContent=" Source and case details ↗";link.target="_blank";link.rel="noreferrer";provenance.append(link);}
  const url=new URL(location.href);url.searchParams.set("case",id);history.replaceState({},"",url);
  let notes=null;
  if (item.notesUrl) {
    try {const response=await fetch(item.notesUrl);if(!response.ok)throw new Error(`HTTP ${response.status}`);notes=await response.json();}
    catch(error){console.error("Case note data unavailable",error);}
  }
  if (version!==loadVersion)return;
  demoRows=notes;
  const cards=document.getElementById("demo-cards");cards.replaceChildren(...methods.map(method=>renderCard(method,item,notes)));
  const inspector=document.getElementById("note-inspector");inspector.hidden=!notes?.notes?.reference?.length;
  if (!inspector.hidden)selectNote(0,false);
}

async function init() {
  const [paper, manifest] = await Promise.all([
    fetch("assets/paper_results.json").then(r=>{if(!r.ok)throw new Error("Paper data unavailable");return r.json();}),
    fetch("assets/cases.json").then(r=>{if(!r.ok)throw new Error("Case manifest unavailable");return r.json();})
  ]);
  paperRows = paper.evaluation; renderBars("bssl");
  document.querySelectorAll("[data-metric]").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll("[data-metric]").forEach(b => {b.classList.toggle("active",b===button);b.setAttribute("aria-pressed",String(b===button));});
    renderBars(button.dataset.metric);
  }));
  cases=manifest.cases;
  const requested=new URLSearchParams(location.search).get("case");
  const selectedCase=cases.find(c=>c.id===requested)||cases.find(c=>c.id==="maestro-scriabin-60")||cases[0];
  document.getElementById("case-select").addEventListener("change",event=>loadCase(event.target.value));
  renderDatasetTabs(selectedCase.dataset);
  const select=document.getElementById("case-select");
  for (const item of cases.filter(c=>c.dataset===selectedCase.dataset)) {
    const option=document.createElement("option");option.value=item.id;option.textContent=item.selectorLabel;select.append(option);
  }
  select.value=selectedCase.id;
  document.getElementById("previous-note").addEventListener("click",()=>selectNote(selectedNoteIndex-1,true));
  document.getElementById("next-note").addEventListener("click",()=>selectNote(selectedNoteIndex+1,true));
  await loadCase(selectedCase.id);
}
init().catch(error => { console.error(error); document.getElementById("case-title").textContent="Demo data unavailable"; });
