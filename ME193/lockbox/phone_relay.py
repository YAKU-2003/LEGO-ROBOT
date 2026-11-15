"""
Phone page for the LunchLock box, served from the LAPTOP.

  python phone_relay.py        then on your phone (same Wi-Fi): http://<laptop IP>:8003

The board (unoq-lunchlock-app) already sends its live camera view and status to this laptop
over MQTT. This script passes them on to your phone and turns the phone's button into the
safe's button. Needed because App Lab doesn't let the phone reach the board's own port 7000.

First run: Windows asks whether Python may accept connections - allow it on PRIVATE networks.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from latch_link import LatchLink

PORT = 8003

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LunchLock</title>
<style>
body{margin:0;background:#0b0d12;color:#e9ecf2;font:16px system-ui,sans-serif;text-align:center}
h1{font-size:20px;margin:14px 0 4px} #st{color:#8b93a5;margin-bottom:10px;min-height:22px}
img{width:100%;max-width:640px;display:block;margin:0 auto;background:#000;min-height:120px}
button{margin:18px auto;display:block;width:80%;max-width:360px;padding:22px;font-size:22px;font-weight:700;
color:#fff;border:0;border-radius:999px;background:linear-gradient(135deg,#ff6b6b,#e0344f)}
button:active{transform:translateY(2px)} p{color:#8b93a5;font-size:13px;padding:0 16px}
#ev{color:#c9cfdb;font-size:13px;line-height:1.6}
</style></head><body>
<h1>&#129386; LunchLock</h1><div id="st">connecting&hellip;</div>
<img src="/stream.mjpg" alt="LunchLock camera">
<button onclick="fetch('/button',{method:'POST'})">Press button</button>
<p>Press, then show your 3 numbers in the box &mdash; hold each until its box fills.</p>
<div id="ev"></div>
<script>
async function poll(){try{const s=await (await fetch('/status')).json();
const b=s.board; document.getElementById('st').textContent = b ?
 b.state+' \\u00b7 sees '+b.k+' ('+Math.round(b.conf*100)+'%) \\u00b7 '+b.entered+'/3 \\u00b7 '+b.fps+' fps' : 'waiting for the box\\u2026';
document.getElementById('ev').innerHTML=s.events.map(e=>e[0]+' &nbsp;'+e[1]).reverse().join('<br>');}
catch(e){document.getElementById('st').textContent='laptop offline';} setTimeout(poll,500);} poll();
</script></body></html>"""

link = LatchLink(print_events=False)
watchers = {"last": 0.0}


def keepalive():
    """Ask the board for preview frames while a phone is watching."""
    while True:
        if time.time() - watchers["last"] < 5:
            link.keep_preview_on()
        time.sleep(2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/status":
            board = None
            if link.board_status and link.board_online:
                try:
                    board = json.loads(link.board_status)
                except ValueError:
                    pass
            return self._send(json.dumps({"board": board, "events": link.events[-5:]}).encode(),
                              "application/json")
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while True:
                    watchers["last"] = time.time()
                    jpg = link.last_preview
                    if jpg is not None and jpg is not last and time.time() - link.t_preview < 3:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        last = jpg
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                return
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path == "/button":
            link.press_button()
            return self._send(b'{"ok":true}', "application/json")
        self._send(b"not found", "text/plain", 404)


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))            # no data sent; just picks the Wi-Fi interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "<laptop IP>"


def main():
    threading.Thread(target=keepalive, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"LunchLock phone page: http://{lan_ip()}:{PORT}   (phone on the same Wi-Fi)")
    print("Ctrl+C to stop")
    server.serve_forever()


if __name__ == "__main__":
    main()
