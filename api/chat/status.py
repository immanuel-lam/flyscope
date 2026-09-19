from http.server import BaseHTTPRequestHandler
from pathlib import Path
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        root=Path(__file__).resolve().parents[2]
        path=root/'models/malecns-chat/manifest.json'
        manifest=json.loads(path.read_text())
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(json.dumps({'ready':(path.parent/'runtime.npz').exists(),'modelId':manifest['modelId'],'neurons':len(manifest['neurons']),'edges':manifest['edges']}).encode())
