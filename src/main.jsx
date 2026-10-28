import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import './styles.css';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
let landmarkerPromise;

async function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      try {
        return await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
          runningMode: 'IMAGE', numFaces: 1,
          outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
        });
      } catch {
        return FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
          runningMode: 'IMAGE', numFaces: 1,
          outputFaceBlendshapes: false, outputFacialTransformationMatrixes: false,
        });
      }
    })().catch((err) => { landmarkerPromise = undefined; throw err; });
  }
  return landmarkerPromise;
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const dist  = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const lerp  = (a, b, t) => a + (b - a) * t;
const PHI   = 1.618033988749895;

// ── PSL Tier table (community standard 1–8) ──────────────────────────────
const PSL_TIERS = [
  { id:'sub3',     label:'Sub-3',     range:[1.0,2.9], color:'#7c5a4f', desc:'Significantly below average facial geometry.' },
  { id:'ltn',      label:'LTN',       range:[3.0,3.9], color:'#9a7052', desc:'Low-Tier Normie — below average.' },
  { id:'mtn',      label:'MTN',       range:[4.0,4.9], color:'#b08558', desc:'Mid-Tier Normie — average. The statistical midpoint.' },
  { id:'htn',      label:'HTN',       range:[5.0,5.9], color:'#c89f62', desc:'High-Tier Normie — above average, noticeably attractive.' },
  { id:'chadlite', label:'Chadlite',  range:[6.0,6.4], color:'#deba70', desc:'Strong bone structure, clearly above the norm.' },
  { id:'chad',     label:'Chad',      range:[6.5,7.4], color:'#eecf82', desc:'Exceptional facial geometry, top-tier attractiveness.' },
  { id:'truadam',  label:'True Adam', range:[7.5,8.0], color:'#f5e39e', desc:'Near-perfect facial architecture. Extremely rare — not reachable via 2D photo analysis.' },
];

function getTier(score) {
  for (const t of PSL_TIERS) if (score >= t.range[0] && score <= t.range[1]) return t;
  return score < 2 ? PSL_TIERS[0] : PSL_TIERS[PSL_TIERS.length - 1];
}

function goldenScore(ratio) {
  const best = Math.min(Math.abs(ratio - PHI), Math.abs(ratio - 1/PHI), Math.abs(ratio - 1.0));
  return clamp(100 - best * 200, 0, 100);
}

// ── Core measurement engine ───────────────────────────────────────────────
function measureFace(landmarks, W, H) {
  const p = i => ({ x: landmarks[i].x * W, y: landmarks[i].y * H });

  const leftCheek     = p(234); const rightCheek    = p(454);
  const topHead       = p(10);  const chin          = p(152);
  const leftJaw       = p(172); const rightJaw      = p(397);
  const jawLeft2      = p(136); const jawRight2     = p(365);
  const leftEyeInner  = p(133); const leftEyeOuter  = p(33);
  const rightEyeInner = p(362); const rightEyeOuter = p(263);
  const leftEyeTop    = p(159); const leftEyeBot    = p(145);
  const rightEyeTop   = p(386); const rightEyeBot   = p(374);
  const leftBrow      = p(70);
  const noseTip       = p(1);   const noseBase      = p(2);
  const noseLeft      = p(129); const noseRight     = p(358);
  const noseTop       = p(168);

  const faceWidth  = Math.max(1, dist(leftCheek, rightCheek));
  const faceHeight = Math.max(1, Math.abs(chin.y - topHead.y));
  const centerX    = (leftCheek.x + rightCheek.x) / 2;

  // 1. Symmetry
  const symPairs = [[33,263],[133,362],[159,386],[145,374],[70,300],[105,334],[61,291],[78,308],[172,397],[234,454],[129,358]];
  const symErrors = symPairs.map(([li,ri]) => {
    const l = p(li), r = p(ri);
    return (Math.abs((centerX-l.x)-(r.x-centerX))/faceWidth + Math.abs(l.y-r.y)/faceHeight) / 2;
  });
  const symmetryScore = clamp(100 - (symErrors.reduce((s,v)=>s+v,0)/symErrors.length) * 900, 0, 100);

  // 2. Facial thirds
  const browY      = leftBrow.y;
  const noseBaseY  = noseBase.y;
  const topY       = topHead.y;
  const chinY      = chin.y;
  const foreheadH  = Math.abs(browY - topY);
  const midfaceH   = Math.abs(noseBaseY - browY);
  const lowerH     = Math.abs(chinY - noseBaseY);
  const totalH     = foreheadH + midfaceH + lowerH || 1;
  const t1 = foreheadH/totalH, t2 = midfaceH/totalH, t3 = lowerH/totalH;
  const ideal1_3 = 1/3;
  const thirdsScore = clamp(100 - ((Math.abs(t1-ideal1_3)+Math.abs(t2-ideal1_3)+Math.abs(t3-ideal1_3))/3)*600, 0, 100);

  // 3. Facial fifths
  const seg1 = Math.abs(leftEyeOuter.x - leftCheek.x);
  const seg2 = dist(leftEyeOuter, leftEyeInner);
  const seg3 = dist(leftEyeInner, rightEyeInner);
  const seg4 = dist(rightEyeInner, rightEyeOuter);
  const seg5 = Math.abs(rightCheek.x - rightEyeOuter.x);
  const fTotal = [seg1,seg2,seg3,seg4,seg5].reduce((a,b)=>a+b,0) || 1;
  const fifthsErr = [seg1,seg2,seg3,seg4,seg5].map(s=>Math.abs(s/fTotal-0.20)).reduce((a,b)=>a+b,0)/5;
  const fifthsScore = clamp(100 - fifthsErr * 600, 0, 100);

  // 4. Jaw
  const jawWidth  = dist(leftJaw, rightJaw);
  const jawRatio  = jawWidth / faceWidth;
  const jawScore  = clamp(100 - Math.abs(jawRatio - 0.70) * 400, 0, 100);

  // 5. Cheekbones
  const cheekJawRatio  = faceWidth / Math.max(jawWidth, 1);
  const cheekScore     = clamp(100 - Math.abs(cheekJawRatio - 1.15) * 300, 0, 100);

  // 6. Eye area
  const avgEyeW = (dist(leftEyeOuter,leftEyeInner) + dist(rightEyeInner,rightEyeOuter)) / 2;
  const avgEyeH = (dist(leftEyeTop,leftEyeBot) + dist(rightEyeTop,rightEyeBot)) / 2;
  const eyeWidthRatio   = avgEyeW / faceWidth;
  const eyeOpenRatio    = avgEyeH / faceHeight;
  const eyeAreaScore = clamp(
    goldenScore(eyeWidthRatio/0.20)*0.5 + clamp(100-Math.abs(eyeOpenRatio-0.065)*1200,0,100)*0.5,
    0, 100
  );

  // 7. Nose
  const noseWidth  = dist(noseLeft, noseRight);
  const noseHeight = dist(noseTop, noseTip);
  const noseScore  = clamp(
    (clamp(100-Math.abs(noseWidth/faceWidth-0.20)*500,0,100) +
     clamp(100-Math.abs(noseHeight/faceHeight-0.22)*400,0,100)) / 2,
    0, 100
  );

  // 8. Golden ratio W:H
  const whr           = faceWidth / faceHeight;
  const goldenFaceScr = goldenScore(whr);

  // 9. Chin taper
  const jawTaper  = dist(jawLeft2, jawRight2) / Math.max(jawWidth, 1);
  const chinScore = clamp(100 - Math.abs(jawTaper - 0.84) * 400, 0, 100);

  // Composite
  const boneScore  = cheekScore*0.40 + jawScore*0.35 + chinScore*0.25;
  const propScore  = thirdsScore*0.35 + fifthsScore*0.35 + goldenFaceScr*0.30;
  const composite  = eyeAreaScore*0.25 + boneScore*0.30 + symmetryScore*0.20 + propScore*0.20 + noseScore*0.05;

  // ── Realistic PSL distribution calibration ────────────────────────────
  // Real PSL community distribution (approximate):
  //   Sub-3:    ~5%   of people
  //   LTN:      ~20%
  //   MTN:      ~40%  ← statistical center
  //   HTN:      ~25%
  //   Chadlite: ~7%
  //   Chad:     ~2.5%
  //   True Adam: ~0.5% (extremely rare)
  //
  // 2D geometry (no depth/3D) inflates scores vs real PSL community ratings.
  // We apply a strong downward compression + steeper sigmoid to reflect this.
  //
  // Penalty 1: high symmetry from flat/passport lighting is not real bone structure.
  // If symmetry is very high (>80) but other bone scores are modest, compress.
  const passportPenalty = symmetryScore > 78
    ? (symmetryScore - 78) * 0.18   // up to ~12 point deduction on composite
    : 0;

  // Penalty 2: if all scores are uniformly high (>70), likely ideal photo conditions
  const scores5 = [eyeAreaScore, boneScore, symmetryScore, propScore, noseScore];
  const allHigh = scores5.every(s => s > 65);
  const uniformityPenalty = allHigh ? 8 : 0;

  const adjustedComposite = clamp(composite - passportPenalty - uniformityPenalty, 0, 100);

  // Sigmoid with center shifted to 0.62 (vs 0.5 before) → most people land MTN/HTN
  // Steeper curve (k=14) → harder to reach extremes
  const norm    = adjustedComposite / 100;
  const sigmoid = 1 / (1 + Math.exp(-14 * (norm - 0.62)));

  // Map sigmoid to PSL 1.0–8.0, but cap the practical maximum at 7.2
  // (True Adam 7.5+ requires extraordinary bone structure impossible to verify in 2D)
  const rawScore = lerp(1.0, 7.2, sigmoid);

  // Final floor: minimum 1.5 (prevents extreme outliers from bad photos)
  const pslEstimate = Math.round(clamp(rawScore, 1.5, 7.2) * 10) / 10;

  return {
    pslEstimate,
    tier: getTier(pslEstimate),
    scores: {
      symmetry:      Math.round(symmetryScore),
      eyeArea:       Math.round(eyeAreaScore),
      boneStructure: Math.round(boneScore),
      facialThirds:  Math.round(thirdsScore),
      facialFifths:  Math.round(fifthsScore),
      jawDefinition: Math.round(jawScore),
      cheekbones:    Math.round(cheekScore),
      noseHarmony:   Math.round(noseScore),
      goldenRatio:   Math.round(goldenFaceScr),
    },
    measurements: {
      faceWHRatio:   Math.round(whr * 100) / 100,
      jawFaceRatio:  Math.round(jawRatio * 100) / 100,
      cheekJawRatio: Math.round(cheekJawRatio * 100) / 100,
      interEyeRatio: Math.round((seg3 / faceWidth) * 100) / 100,
      thirds: { forehead: Math.round(t1*100), midface: Math.round(t2*100), lower: Math.round(t3*100) },
    },
    landmarkCount: landmarks.length,
  };
}

// ── Landmark overlay ──────────────────────────────────────────────────────
function LandmarkOverlay({ landmarks }) {
  if (!landmarks?.length) return null;
  const outline = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10];
  const jawLine = [172,136,150,149,176,148,152,377,400,378,379,365,397];
  const pts = ids => ids.map(i=>`${landmarks[i].x},${landmarks[i].y}`).join(' ');
  return (
    <svg className="landmark-overlay" viewBox="0 0 1 1" preserveAspectRatio="none">
      <line x1="0.5" y1="0.04" x2="0.5" y2="0.97" className="axis-line"/>
      <polyline points={pts(outline)} className="face-outline"/>
      <polyline points={pts(jawLine)} className="jaw-line"/>
      {[[33,133],[362,263]].map(([a,b],i)=>(
        <line key={i} x1={landmarks[a].x} y1={landmarks[a].y} x2={landmarks[b].x} y2={landmarks[b].y} className="feature-line"/>
      ))}
      {landmarks.map((lm,i)=>i%5===0&&<circle key={i} cx={lm.x} cy={lm.y} r="0.003" className="landmark-dot"/>)}
    </svg>
  );
}

// ── Score bar ─────────────────────────────────────────────────────────────
function ScoreBar({ label, value, weight }) {
  const pct   = clamp(value, 0, 100);
  const color = pct >= 70 ? 'var(--gold)' : pct >= 45 ? 'var(--mid)' : 'var(--low)';
  return (
    <div className="score-bar">
      <div className="score-bar-head">
        <span className="score-bar-label">{label}</span>
        {weight && <span className="score-weight">{weight}</span>}
        <span className="score-val" style={{color}}>{value}</span>
      </div>
      <div className="score-track">
        <div className="score-fill" style={{width:`${pct}%`,background:color}}/>
      </div>
    </div>
  );
}

// ── Tier spectrum ─────────────────────────────────────────────────────────
function TierSpectrum({ score }) {
  const pos = clamp(((score-1)/7)*100, 1, 99);
  return (
    <div className="spectrum-wrap">
      <div className="spectrum">
        {PSL_TIERS.map(t=>(
          <div key={t.id} className="spec-seg" style={{background:t.color}} title={t.label}/>
        ))}
        <div className="spec-needle" style={{left:`${pos}%`}}/>
      </div>
      <div className="spec-labels">
        {PSL_TIERS.map(t=><span key={t.id}>{t.label}</span>)}
      </div>
    </div>
  );
}

// ── Facial thirds visual ──────────────────────────────────────────────────
function ThirdsVisual({ thirds }) {
  const { forehead, midface, lower } = thirds;
  const ideal = 33;
  return (
    <div className="thirds-vis">
      {[['Forehead', forehead], ['Midface', midface], ['Lower face', lower]].map(([lbl, val]) => {
        const ok = Math.abs(val - ideal) <= 4;
        return (
          <div key={lbl} className="third-row">
            <span className="third-lbl">{lbl}</span>
            <div className="third-track">
              <div className="third-fill" style={{width:`${val}%`, background: ok?'var(--gold)':'var(--low)'}}/>
            </div>
            <span className="third-val" style={{color: ok?'var(--gold)':'var(--low)'}}>{val}% {ok?'✓':'!'}</span>
          </div>
        );
      })}
      <p className="thirds-note">Ideal proportion: ~33% per zone</p>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────
function App() {
  const [file, setFile]             = useState(null);
  const [imageUrl, setImageUrl]     = useState('');
  const [result, setResult]         = useState(null);
  const [status, setStatus]         = useState('idle');
  const [error, setError]           = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [showLm, setShowLm]         = useState(true);
  const [tab, setTab]               = useState('breakdown');
  const [modalOpen, setModalOpen]   = useState(false);
  const fileRef  = useRef(null);
  const imageRef = useRef(null);

  useEffect(()=>()=>{ if(imageUrl) URL.revokeObjectURL(imageUrl); },[imageUrl]);

  function acceptFile(f) {
    setError(''); setResult(null);
    if (!f) return;
    if (!['image/jpeg','image/png','image/webp'].includes(f.type)) { setError('Pilih file JPG, PNG, atau WebP.'); return; }
    if (f.size > 20*1024*1024) { setError('Ukuran maksimal 20 MB.'); return; }
    setFile(f); setImageUrl(URL.createObjectURL(f)); setStatus('ready');
  }

  function reset() {
    setFile(null); setResult(null); setError(''); setStatus('idle');
    setImageUrl(p=>{ if(p) URL.revokeObjectURL(p); return ''; });
    if (fileRef.current) fileRef.current.value='';
  }

  async function analyze() {
    if (!file || !imageRef.current) return;
    setError(''); setStatus('analyzing'); setResult(null);
    try {
      const lm = await getLandmarker();
      const det = lm.detect(imageRef.current);
      const lmarks = det.faceLandmarks?.[0];
      if (!lmarks?.length) {
        setError('Wajah tidak terdeteksi. Gunakan foto frontal, cahaya merata, wajah tidak terhalang.'); setStatus('ready'); return;
      }
      const metrics = measureFace(lmarks, imageRef.current.naturalWidth, imageRef.current.naturalHeight);
      setResult({...metrics, landmarks: lmarks}); setStatus('complete');
      setTimeout(()=>document.getElementById('results')?.scrollIntoView({behavior:'smooth',block:'start'}),80);
    } catch(e) {
      console.error(e);
      setError('Model gagal dimuat. Periksa koneksi internet dan coba lagi.'); setStatus('ready');
    }
  }

  const tier = result?.tier;

  return (
    <main className="shell">

      <header className="topbar">
        <a className="brand" href="#top">
          <span className="brand-mark">P</span>
          <span className="brand-text">PSL <em>Analyzer</em></span>
        </a>
        <nav className="top-nav">
          <button className="nav-btn" onClick={()=>setModalOpen(true)}>Methodology</button>
          <span className="beta-pill">BETA</span>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-dot"/>PSL STANDARD · 7-TIER SYSTEM</p>
          <h1>Facial Geometry<br/><em>Analysis</em></h1>
          <p className="hero-sub">Measures bone structure, proportions, and symmetry using the PSL framework — Sub-3 through True Adam on the 1–8 scale.</p>
          <div className="privacy-tag"><span className="lock-ico">◎</span><span>Photo processed locally — never uploaded.</span></div>
        </div>
        <div className="tier-preview">
          <p className="eyebrow">THE 7-TIER PSL SCALE</p>
          <div className="tier-list">
            {[...PSL_TIERS].reverse().map(t=>(
              <div key={t.id} className="tier-chip" style={{borderColor:t.color}}>
                <span className="tc-label" style={{color:t.color}}>{t.label}</span>
                <span className="tc-range">{t.range[0]}–{t.range[1]}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="workspace">
        <div className="sec-head">
          <div><p className="eyebrow">01 — INPUT</p><h2>Upload a portrait</h2></div>
          <p className="sec-aside">JPG · PNG · WEBP · MAX 20MB</p>
        </div>

        <div className="upload-layout">
          <div className={`upload-card${isDragging?' dragging':''}${file?' has-file':''}`}
            onDragOver={e=>{e.preventDefault();setIsDragging(true);}}
            onDragLeave={()=>setIsDragging(false)}
            onDrop={e=>{e.preventDefault();setIsDragging(false);acceptFile(e.dataTransfer.files?.[0]);}}>
            {!file ? (
              <>
                <div className="upload-ico">＋</div>
                <h3>Select or drag a portrait</h3>
                <p>Front-facing, neutral expression, even lighting.</p>
                <button className="btn btn-primary" onClick={()=>fileRef.current?.click()}>Choose photo <span>↗</span></button>
              </>
            ) : (
              <>
                <div className="file-status"><span className="status-dot"/>PHOTO READY <span className="file-name">{file.name}</span></div>
                <div className="preview-wrap"><img ref={imageRef} src={imageUrl} alt="Portrait"/></div>
                <div className="photo-actions">
                  <button className="btn btn-quiet" onClick={reset}>Remove</button>
                  <button className="btn btn-primary" disabled={status==='analyzing'} onClick={analyze}>
                    {status==='analyzing'?'Analyzing…':'Analyze face'} <span>→</span>
                  </button>
                </div>
              </>
            )}
            <input ref={fileRef} className="sr-only" type="file" accept="image/jpeg,image/png,image/webp"
              onChange={e=>acceptFile(e.target.files?.[0])}/>
          </div>

          <aside className="guide-card">
            <p className="eyebrow">GUIDELINES</p>
            {[['01','Face forward','Head upright, look toward the camera.'],
              ['02','Even lighting','No harsh shadows or extreme angles.'],
              ['03','Unobstructed','Eyes, nose, jawline, and chin visible.']].map(([n,t,d])=>(
              <div key={n} className="guideline"><span>{n}</span><div><b>{t}</b><small>{d}</small></div></div>
            ))}
            <div className="priv-note"><span>◎</span><p><b>Privacy by design</b><small>No image upload. Runs in your browser only.</small></p></div>
          </aside>
        </div>

        {error && <div className="alert alert-error" role="alert">{error}</div>}
        {status==='analyzing' && (
          <div className="alert alert-info" role="status">
            <span className="spinner"/>Loading analysis engine and measuring 478 facial landmarks…
          </div>
        )}
      </section>

      {result && (
        <section className="results" id="results">
          <div className="sec-head res-head">
            <div><p className="eyebrow">02 — RESULTS</p><h2>Your PSL analysis</h2></div>
            <button className="btn btn-quiet" onClick={reset}>Analyze another ↗</button>
          </div>

          <div className="res-layout">
            <div className="portrait-panel">
              <div className="panel-top">
                <span>LANDMARK VIEW · {result.landmarkCount} POINTS</span>
                <button className="toggle-btn" onClick={()=>setShowLm(v=>!v)}>
                  {showLm?'Hide overlay':'Show overlay'}
                </button>
              </div>
              <div className="portrait-frame">
                <div className="img-wrap">
                  <img src={imageUrl} alt="Analyzed portrait"/>
                  {showLm && <LandmarkOverlay landmarks={result.landmarks}/>}
                </div>
              </div>
            </div>

            <div className="psl-card">
              <p className="eyebrow">PSL ESTIMATE · EXPERIMENTAL</p>
              <div className="tier-badge" style={{color:tier.color}}>{tier.label}</div>
              <div className="score-display">
                <span className="score-num">{result.pslEstimate.toFixed(1)}</span>
                <div className="score-meta"><span>/ 8.0</span><small>PSL scale</small></div>
              </div>
              <TierSpectrum score={result.pslEstimate}/>
              <p className="tier-desc">{tier.desc}</p>
              <div className="range-tag">
                <span>Tier range</span>
                <span style={{color:tier.color}}>{tier.range[0]}–{tier.range[1]}</span>
              </div>
              <p className="psl-caveat">
                2D geometry only — cannot measure bone depth, brow ridge projection, or canthal tilt. Scores are compressed vs real PSL community ratings. Chad+ tiers require extraordinary 3D bone structure unverifiable from a flat photo.
              </p>
            </div>
          </div>

          {/* ── Detailed tabs ── */}
          <div className="sec-head" style={{marginTop:'40px'}}>
            <div><p className="eyebrow">03 — BREAKDOWN</p><h2>Scoring detail</h2></div>
            <div className="tabs">
              {[['breakdown','Score breakdown'],['measurements','Measurements'],['thirds','Facial thirds']].map(([id,lbl])=>(
                <button key={id} className={`tab${tab===id?' active':''}`} onClick={()=>setTab(id)}>{lbl}</button>
              ))}
            </div>
          </div>

          {tab==='breakdown' && (
            <div className="breakdown-grid">
              <div className="bdcol">
                <p className="col-hd">BONE STRUCTURE <em>30%</em></p>
                <ScoreBar label="Cheekbone prominence" value={result.scores.cheekbones}    weight="×0.40"/>
                <ScoreBar label="Jaw definition"        value={result.scores.jawDefinition} weight="×0.35"/>
                <ScoreBar label="Chin / lower face"     value={result.scores.boneStructure} weight="×0.25"/>
              </div>
              <div className="bdcol">
                <p className="col-hd">EYE AREA <em>25%</em></p>
                <ScoreBar label="Eye size & spacing" value={result.scores.eyeArea}/>
                <p className="col-hd" style={{marginTop:'24px'}}>SYMMETRY <em>20%</em></p>
                <ScoreBar label="Bilateral symmetry" value={result.scores.symmetry}/>
              </div>
              <div className="bdcol">
                <p className="col-hd">PROPORTIONS <em>20%</em></p>
                <ScoreBar label="Facial thirds"  value={result.scores.facialThirds} weight="×0.35"/>
                <ScoreBar label="Facial fifths"  value={result.scores.facialFifths} weight="×0.35"/>
                <ScoreBar label="Golden ratio"   value={result.scores.goldenRatio}  weight="×0.30"/>
                <p className="col-hd" style={{marginTop:'24px'}}>NOSE <em>5%</em></p>
                <ScoreBar label="Nose harmony"   value={result.scores.noseHarmony}/>
              </div>
            </div>
          )}

          {tab==='measurements' && (
            <div className="measure-panel">
              {[
                ['Face W:H ratio',     result.measurements.faceWHRatio,   'Ideal ≈ 0.62–0.72 (golden proportion)'],
                ['Jaw to face ratio',  result.measurements.jawFaceRatio,   'Ideal ≈ 0.67–0.72 (strong tapered jaw)'],
                ['Cheek:jaw ratio',    result.measurements.cheekJawRatio,  'Ideal > 1.10 (prominent cheekbones)'],
                ['Inter-eye ratio',    result.measurements.interEyeRatio,  'Ideal ≈ 0.20 (facial fifths standard)'],
              ].map(([label,val,note])=>(
                <div key={label} className="measure-row">
                  <div><span className="m-label">{label}</span><span className="m-note">{note}</span></div>
                  <span className="m-val">{val}</span>
                </div>
              ))}
            </div>
          )}

          {tab==='thirds' && (
            <div className="thirds-panel">
              <p className="thirds-intro">Facial thirds assess vertical proportions — forehead, midface, and lower face should each occupy ~⅓ of the total face height. Imbalance lowers the PSL estimate.</p>
              <ThirdsVisual thirds={result.measurements.thirds}/>
            </div>
          )}

          <div className="disclaimer">
            <span className="disc-ico">ⓘ</span>
            <div>
              <b>Important context</b>
              <p>PSL is an informal community framework, not a scientific measure of attractiveness. This prototype estimates 2D geometric signals only; it cannot assess skin quality, feature masculinity/femininity, health, or aesthetic harmony beyond landmark distances. Camera angle, lighting, and lens distortion can shift results by 0.5–1.0 points. Human worth is not measurable by any geometry.</p>
            </div>
          </div>
        </section>
      )}

      <footer className="footer">
        <span>PSL FACE ANALYZER</span>
        <span>LOCAL-FIRST · NO IMAGE STORAGE</span>
        <button onClick={()=>setModalOpen(true)}>Framework notes ↗</button>
      </footer>

      {modalOpen && (
        <div className="modal-bg" role="presentation" onClick={e=>{ if(e.target===e.currentTarget) setModalOpen(false); }}>
          <section className="modal" role="dialog" aria-modal aria-labelledby="modal-title">
            <button className="modal-close" onClick={()=>setModalOpen(false)} aria-label="Close">×</button>
            <p className="eyebrow">METHODOLOGY</p>
            <h2 id="modal-title">How scoring works</h2>
            <p>The PSL scale runs 1–8 across 7 tiers: Sub-3, LTN, MTN, HTN, Chadlite, Chad, and True Adam. This analyzer maps a composite geometric index onto that scale.</p>
            <div className="method-list">
              {[
                ['Bone Structure','30%','Cheekbone prominence, jaw width, chin taper'],
                ['Eye Area',      '25%','Eye size, openness, inter-eye spacing (facial fifths)'],
                ['Symmetry',      '20%','Bilateral mirror consistency across 11 landmark pairs'],
                ['Proportions',   '20%','Facial thirds, facial fifths, golden ratio W:H'],
                ['Nose',          '5%', 'Nose width and height relative to face dimensions'],
              ].map(([cat,pct,desc])=>(
                <div key={cat} className="method-row">
                  <div><b>{cat}</b><small>{desc}</small></div>
                  <span className="m-pct">{pct}</span>
                </div>
              ))}
            </div>
            <p>MediaPipe detects 478 landmarks client-side. Your photo is never uploaded. This is an experimental prototype — not an official PSL evaluation.</p>
            <a href="https://thefacereport.com/guides/looksmaxxing-chart" target="_blank" rel="noreferrer">PSL chart reference ↗</a>
            <button className="btn btn-primary" style={{marginTop:'20px',width:'100%'}} onClick={()=>setModalOpen(false)}>Got it</button>
          </section>
        </div>
      )}
    </main>
  );
}

createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);
