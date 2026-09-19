"""Exercise the actual hosted Python handler with local HTTP requests."""
import importlib.util,json,threading,unittest,urllib.request
from http.server import HTTPServer
from pathlib import Path

class HostedChatChecks(unittest.TestCase):
    def test_stream_and_reject_invalid_input(self):
        path=Path(__file__).resolve().parents[1]/'api/chat/generate.py'
        spec=importlib.util.spec_from_file_location('hosted_generate',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        server=HTTPServer(('127.0.0.1',0),module.handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}/'
            request=urllib.request.Request(url,data=json.dumps({'message':'Hi','stream':True}).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request) as response:
                events=[json.loads(line) for line in response]
            self.assertEqual(events[0]['type'],'start')
            self.assertEqual(events[-1]['type'],'result')
            states=[e for e in events if e['type']=='state']
            self.assertGreater(len(states),1)
            self.assertEqual(len(states[0]['values']),512)
            tokens=[e for e in events if e['type']=='token']
            self.assertEqual(tokens[-1]['text'],events[-1]['result']['text'])
            self.assertGreater(tokens[-1]['elapsedSeconds'],0)
            self.assertEqual(tokens[-1]['tokenCount'],len(events[-1]['result']['tokenIds']))
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(url,data=b'{"message":""}'))
            self.assertEqual(error.exception.code,400)
        finally:
            server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
