import {useEffect,useRef,useState} from 'react';
import {frameIndex,type PhysicsRun} from './types';
export default function PhysicsPanel({run,time,onRun}:{run?:PhysicsRun;time:number;onRun:(run:PhysicsRun)=>void}){
  const [drive,setDrive]=useState(1),[turn,setTurn]=useState(0),[seconds,setSeconds]=useState(2);
  const [silenced,setSilenced]=useState(false),[feedback,setFeedback]=useState(true);
  const [busy,setBusy]=useState(false),[message,setMessage]=useState(''),[error,setError]=useState(''),[available,setAvailable]=useState(false);
  const [installed,setInstalled]=useState(false);
  const mounted=useRef(true),callback=useRef(onRun);callback.current=onRun;
  useEffect(()=>{mounted.current=true;fetch('/api/physics/status').then(r=>r.json()).then(s=>{if(mounted.current){setAvailable(s.available);setInstalled(s.installed===true);if(s.installed===false)setMessage(s.phase||"Run npm run setup:physics locally to enable physics.");if(s.state==='running'){setBusy(true);setMessage(s.phase);}}}).catch(()=>{});return()=>{mounted.current=false;};},[]);
  async function load(){
    try{const response=await fetch('/api/physics/result');if(!response.ok)throw new Error('No completed physics run is available');const result=await response.json();if(result.rig!=='neuromechfly-2.1.0'||result.datasetId!=='male-cns-v1-full')throw new Error('Unsupported physics run');if(mounted.current){callback.current(result);setAvailable(true);}}
    catch(e){if(mounted.current)setError((e as Error).message);}
  }
  useEffect(()=>{
    if(!busy)return;
    let cancelled=false;
    const timer=setInterval(async()=>{try{const r=await fetch('/api/physics/status');const s=await r.json();if(cancelled)return;setMessage(`${s.phase}${s.progress!==undefined?` · ${Math.round(s.progress*100)}%`:''}`);if(s.state==='complete'){setBusy(false);void load();}else if(s.state==='failed'){setBusy(false);setError(s.error);}}catch(e){if(!cancelled){setBusy(false);setError((e as Error).message);}}},1000);
    return()=>{cancelled=true;clearInterval(timer);};
  },[busy]);
  async function start(){
    setError('');setMessage('Starting local physics…');setBusy(true);
    try{const r=await fetch('/api/physics/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({duration:seconds,drive,turn,silenced,feedback})});const s=await r.json();if(!r.ok)throw new Error(s.error);}
    catch(e){setBusy(false);setError((e as Error).message);}
  }
  const frame=run?.frames[frameIndex(run,time)];
  return <section className="motor-panel physics-panel" aria-label="Physical fly simulation">
    <div className="motor-heading"><div><span className="eyebrow">PHYSICAL FLY · LOCAL MUJOCO</span><h3>Neurons → legs → ground → feedback.</h3></div><button onClick={start} disabled={busy||!installed}>{busy?'Computing…':'Run physics'}</button></div>
    <p className="motor-provenance">Actual NeuroMechFly body and contact physics. MaleCNS rate dynamics and population-to-leg mapping are experimental engineering assumptions. This is not a validated biological walking model.</p>
    <div className="motor-controls">
      <label>Descending stimulus <output>{drive.toFixed(2)}</output><input aria-label="Descending stimulus" type="range" min="0" max="2" step=".1" value={drive} onChange={e=>setDrive(+e.target.value)} disabled={busy}/></label>
      <label>Left/right bias <output>{turn.toFixed(2)}</output><input aria-label="Left/right bias" type="range" min="-1" max="1" step=".1" value={turn} onChange={e=>setTurn(+e.target.value)} disabled={busy}/></label>
      <label>Duration <select aria-label="Physics duration" value={seconds} onChange={e=>setSeconds(+e.target.value)} disabled={busy}><option value="1">1 second</option><option value="2">2 seconds</option><option value="5">5 seconds</option></select></label>
    </div>
    <div className="physics-options"><label><input type="checkbox" checked={feedback} onChange={e=>setFeedback(e.target.checked)} disabled={busy}/> Contact feedback to neural model</label><label><input type="checkbox" checked={silenced} onChange={e=>setSilenced(e.target.checked)} disabled={busy}/> Silence neural model (control)</label>{busy?<button onClick={()=>fetch('/api/physics/cancel',{method:'POST'})}>Cancel run</button>:available&&<button onClick={load}>Load latest run</button>}</div>
    <p role="status" className="motor-provenance">{message||'Runs at 0.1 ms physics steps. Compute first, then replay or scrub below.'}</p>
    {error&&<p role="alert" className="motor-error">{error}</p>}
    {run&&<><div className="motor-telemetry" data-testid="physics-metrics"><output>{run.metrics.displacementMm.toFixed(2)} mm displacement</output><output>{frame?.contacts??0} contacts</output><output>drive {frame?.drive.map(v=>v.toFixed(2)).join(' / ')}</output><output>{run.metrics.wallSeconds.toFixed(1)} s computation</output></div><p className="motor-provenance">{run.metrics.neurons.toLocaleString()} neurons simulated · {run.metrics.effectiveSignedEdges.toLocaleString()} nonzero signed connections · {run.metrics.recordedNeurons} cells recorded. {run.parameters.silenced?'Neural model silenced.':'Neural output enabled.'} {run.parameters.feedback?'Contact feedback enabled.':'Contact feedback disabled.'}</p><details><summary>Model assumptions and provenance</summary>{Object.entries(run.provenance).map(([key,value])=><p className="motor-provenance" key={key}><strong>{key}: </strong>{value}</p>)}</details></>}
  </section>;
}
