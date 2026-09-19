"""Hosted inference of the same source-masked checkpoint used locally."""
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json, sys, os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/language'))
from runtime import ChatCircuit

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        def emit(event):
            self.wfile.write((json.dumps(event,allow_nan=False)+'\n').encode())
            self.wfile.flush()
        streaming=False
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=12000:raise ValueError('Request must contain at most 12,000 bytes')
            data=json.loads(self.rfile.read(size))
            message=data.get('message')
            history=data.get('history',[])
            if not isinstance(message,str) or not message.strip() or len(message)>1000:raise ValueError('Enter 1–1,000 characters')
            if not isinstance(history,list) or len(history)>4 or any(not isinstance(t,dict) or t.get('role') not in ['user','assistant'] or not isinstance(t.get('content'),str) or len(t['content'])>1000 for t in history):raise ValueError('Invalid conversation history')
            model=ChatCircuit()
            streaming=data.get('stream') is True
            self.send_response(200)
            self.send_header('Content-Type','application/x-ndjson' if streaming else 'application/json')
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            result=model.generate(message,history,max_tokens=40,ablated=data.get('ablated') is True,inspect=True,emit=emit if streaming else None,pace=.05 if data.get('visibleSteps') is True else 0)
            emit({'type':'result','result':result} if streaming else result)
        except (BrokenPipeError,ConnectionResetError):
            return
        except Exception as exc:
            if streaming:
                emit({'type':'error','error':'Inference stopped'})
            else:
                self.send_response(400)
                self.send_header('Content-Type','application/json')
                self.end_headers()
                emit({'error':str(exc) if isinstance(exc,ValueError) else 'Inference unavailable'})
