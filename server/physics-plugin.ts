import type { Plugin, ViteDevServer, PreviewServer } from 'vite';
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync, createReadStream } from 'node:fs';
import { resolve } from 'node:path';

/** Fixed local simulator entry point. Imported datasets can never choose executable/arguments. */
export function physicsPlugin(): Plugin {
  let child: ChildProcess | undefined;
  let state: {state:string; phase:string; progress?:number; error?:string} = {state:'idle',phase:'Ready'};
  const root=process.cwd(), result=resolve(root,'data/physics/latest.json');
  function install(server: ViteDevServer | PreviewServer) {
    server.httpServer?.on('close',()=>child?.kill());
    server.middlewares.use('/api/physics',async(req,res,next)=>{
      const url=(req.url??'').split('?')[0];
      if(!['/status','/run','/result','/cancel'].includes(url)){next();return;}
      const host=req.headers.host??'';
      const origin=req.headers.origin;
      if(!/^(127\.0\.0\.1|localhost|\[::1\]):\d+$/.test(host) || (origin && origin!==`http://${host}`)){
        res.writeHead(403);res.end('Local same-origin requests only');return;
      }
      const json=(code:number,value:unknown)=>{res.writeHead(code,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(value));};
      if(url==='/status'&&req.method==='GET'){json(200,{...state,available:existsSync(result),installed:existsSync(resolve(root,'.venv-physics/bin/python'))});return;}
      if(url==='/result'&&req.method==='GET'){
        if(child||!existsSync(result)){json(409,{error:'No completed run available'});return;}
        res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'no-store'});createReadStream(result).pipe(res);return;
      }
      if(url==='/cancel'&&req.method==='POST'){child?.kill();json(200,{ok:true});return;}
      if(url!=='/run'||req.method!=='POST'){json(405,{error:'Method not allowed'});return;}
      if(child){json(409,{error:'A simulation is already running'});return;}
      try{
        let body='';for await(const chunk of req){body+=chunk;if(body.length>2048)throw new Error('Request too large');}
        const p=JSON.parse(body);
        if(!Number.isFinite(p.duration)||p.duration<0.5||p.duration>5||!Number.isFinite(p.drive)||p.drive<0||p.drive>2||!Number.isFinite(p.turn)||p.turn< -1||p.turn>1||typeof p.silenced!=='boolean'||typeof p.feedback!=='boolean')throw new Error('Invalid physics parameters');
        for(const key of ['odour','odourControl','obstacles','avoidance'])if(p[key]!==undefined&&typeof p[key]!=='boolean')throw new Error('Invalid odour option');
        if(p.vision!==undefined&&typeof p.vision!=='boolean')throw new Error('Invalid vision option');
        if(p.visionControl!==undefined&&typeof p.visionControl!=='boolean')throw new Error('Invalid vision control option');
        if(p.targetY!==undefined&&![4,-4].includes(p.targetY))throw new Error('Invalid target position');
        if(p.layout!==undefined&&!['training','offset','wide'].includes(p.layout))throw new Error('Invalid obstacle layout');
        const python=resolve(root,'.venv-physics/bin/python');
        if(!existsSync(python))throw new Error('Physics runtime missing. Run npm run setup:physics.');
        state={state:'running',phase:'Starting simulator',progress:0};
        child=spawn(python,[resolve(root,'scripts/physics/run.py'),'--duration',String(p.duration),'--drive',String(p.drive),'--turn',String(p.turn),...(p.silenced?['--silenced']:[]),...(!p.feedback?['--no-feedback']:[]),...(p.vision?['--vision']:[]),...((p.vision||p.odour)?['--target-y',String(p.targetY??4)]:[]),...(p.odour?['--odour']:[]),...(p.odourControl===false?['--no-odour-control']:[]),...(p.obstacles?['--obstacles','--layout',p.layout??'training']:[]),...(p.avoidance===false?['--no-avoidance']:[]),...(p.visionControl===false?['--no-vision-control']:[])],{cwd:root,stdio:['ignore','pipe','pipe']});
        let pending='',stderr='';
        child.stdout?.on('data',chunk=>{pending+=chunk;let i;while((i=pending.indexOf('\n'))>=0){const line=pending.slice(0,i);pending=pending.slice(i+1);try{const v=JSON.parse(line);state={state:'running',phase:v.phase,progress:v.progress};}catch{}}});
        child.stderr?.on('data',chunk=>{stderr=(stderr+chunk).slice(-3000);});
        const timeout=setTimeout(()=>child?.kill(),180_000);
        child.on('error',error=>{state={state:'failed',phase:'Could not start simulator',error:error.message};});
        child.on('close',(code)=>{clearTimeout(timeout);child=undefined;state=code===0?{state:'complete',phase:'Run complete',progress:1}:{state:'failed',phase:'Run stopped',error:stderr||'Simulation cancelled or timed out'};});
        json(202,state);
      }catch(e){json(400,{error:(e as Error).message});}
    });
  }
  return {name:'flyscope-physics',configureServer(server){install(server);},configurePreviewServer(server){install(server);}};
}
