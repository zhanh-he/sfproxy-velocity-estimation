const reviewMethods = [
  {key: "reference", label: "00 · Original recording"},
  {key: "diffsynth", label: "02 · Diff-Synth"},
  {key: "sfproxy", label: "03 · Diff-SFProxy"},
];

function velocityColor(value) {
  const stops = [[0,[85,49,133]],[32,[153,58,138]],[64,[218,88,102]],[96,[247,160,76]],[127,[255,224,119]]];
  for (let i=1; i<stops.length; i++) {
    if (value <= stops[i][0]) {
      const [low, a] = stops[i-1], [high, b] = stops[i];
      const t = (value-low)/(high-low);
      return `rgb(${a.map((c,j)=>Math.round(c+(b[j]-c)*t)).join(",")})`;
    }
  }
  return "rgb(255,224,119)";
}

function makeRoll(notes, pitchMin, pitchMax, neutral) {
  const ns="http://www.w3.org/2000/svg", width=500, height=130, left=28, top=7, bottom=19;
  const svg=document.createElementNS(ns,"svg");
  svg.setAttribute("viewBox",`0 0 ${width} ${height}`);
  svg.setAttribute("aria-hidden","true");
  const x=t=>left+(width-left-6)*t/20;
  const y=p=>top+(height-top-bottom)*(pitchMax-p)/(pitchMax-pitchMin+1);
  for (let t=0; t<=20; t+=5) {
    const line=document.createElementNS(ns,"line");
    line.setAttribute("x1",x(t));line.setAttribute("x2",x(t));line.setAttribute("y1",top);line.setAttribute("y2",height-bottom);
    line.setAttribute("stroke","#294057");svg.append(line);
    const label=document.createElementNS(ns,"text");label.setAttribute("x",x(t)-5);label.setAttribute("y",height-4);
    label.setAttribute("fill","#8da1b5");label.setAttribute("font-size","10");label.textContent=`${t}s`;svg.append(label);
  }
  for (const note of notes) {
    const rect=document.createElementNS(ns,"rect");
    rect.setAttribute("x",x(note.s));rect.setAttribute("y",y(note.p)-2.1);
    rect.setAttribute("width",Math.max(1.5,x(note.e)-x(note.s)));rect.setAttribute("height","4.2");
    rect.setAttribute("fill",neutral?"#8398ab":velocityColor(note.v));rect.setAttribute("rx","1");svg.append(rect);
  }
  return svg;
}

function velocityStats(notes) {
  const values=notes.map(note=>note.v);
  return {at102: values.filter(v=>v===102).length/values.length,
    over100: values.filter(v=>v>=100).length/values.length,
    mean: values.reduce((sum,v)=>sum+v,0)/values.length};
}

function renderCase(caseInfo, data) {
  const reference=data.notes.reference;
  const all=Object.values(data.notes).flat();
  const lo=Math.min(...all.map(note=>note.p))-2, hi=Math.max(...all.map(note=>note.p))+2;
  const sf=velocityStats(data.notes.sfproxy);
  const article=document.createElement("article");article.className="fl-review-case";
  const head=document.createElement("div");head.className="fl-review-head";
  const title=document.createElement("div");
  const id=document.createElement("span");id.className="kicker";id.textContent=caseInfo.id;
  const h2=document.createElement("h2");h2.textContent=`${caseInfo.title} · ${caseInfo.window}`;
  title.append(id,h2);
  const summary=document.createElement("div");summary.className="fl-review-stats";
  for (const [label,value] of [["NOTES",String(reference.length)],["03 AT 102",`${(sf.at102*100).toFixed(1)}%`],["03 ≥100",`${(sf.over100*100).toFixed(1)}%`]]) {
    const item=document.createElement("span"), small=document.createElement("small"), strong=document.createElement("strong");
    small.textContent=label;strong.textContent=value;item.append(small,strong);summary.append(item);
  }
  head.append(title,summary);article.append(head);
  const grid=document.createElement("div");grid.className="fl-review-grid";
  for (const method of reviewMethods) {
    const cell=document.createElement("div");cell.className=`fl-review-method ${method.key}`;
    const top=document.createElement("div");top.className="fl-review-method-head";
    const name=document.createElement("strong");name.textContent=method.label;
    const score=document.createElement("span");
    const scores=caseInfo.scores?.[method.key];
    score.textContent=scores?`BSSL ${Number(scores.bssl).toFixed(3)} · BSTL ${Number(scores.bstl).toFixed(3)}`:"Scores pending";
    top.append(name,score);
    const roll=document.createElement("div");roll.className="fl-review-roll";
    roll.setAttribute("role","img");roll.setAttribute("aria-label",`${method.label} MIDI roll for ${caseInfo.id}`);
    roll.append(makeRoll(data.notes[method.key],lo,hi,method.key==="reference"));
    const audio=document.createElement("audio");audio.controls=true;audio.preload="metadata";
    audio.src=`${caseInfo.assetBase}/${method.key}.mp3`;
    audio.setAttribute("aria-label",`${method.label} audio for ${caseInfo.id}`);
    audio.addEventListener("play",()=>document.querySelectorAll("audio").forEach(other=>{if(other!==audio)other.pause();}));
    cell.append(top,roll,audio);grid.append(cell);
  }
  article.append(grid);
  const link=document.createElement("a");link.className="fl-review-link";
  link.href=`index.html?case=${encodeURIComponent(caseInfo.id)}#listen`;
  link.textContent="Open all four conditions and note inspector ↗";
  article.append(link);
  return article;
}

async function initReview() {
  const response=await fetch("assets/cases.json");
  if (!response.ok) throw new Error(`Case manifest: HTTP ${response.status}`);
  const manifest=await response.json();
  const cases=manifest.cases.filter(item=>item.dataset==="FL" && reviewMethods.every(method=>item.available.includes(method.key)));
  const list=document.getElementById("fl-review-list");list.replaceChildren();
  for (const caseInfo of cases) {
    const notesResponse=await fetch(caseInfo.notesUrl);
    if (!notesResponse.ok) throw new Error(`${caseInfo.id}: HTTP ${notesResponse.status}`);
    list.append(renderCase(caseInfo,await notesResponse.json()));
  }
}
initReview().catch(error=>{
  console.error(error);
  document.getElementById("fl-review-list").textContent="The FL excerpts could not be loaded. Please try again.";
});
