#!/usr/bin/env python3
"""Read-only capability-gated review sharing. Run through uv; never serves a directory."""
import argparse
import hmac
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import socket
from socketserver import ThreadingMixIn
import stat
import subprocess
import sys
import threading
import time
import io
import webbrowser

from json_safe import unique_object

MAX_PAGE = 128 * 1024 * 1024
MAX_HEADERS = 8192
CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; object-src 'none'"
BOOTSTRAP = b'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NYM private review</title><p id="status">Opening private review...</p><script>
(async()=>{const key='nym-review-auth:'+location.host;let token=new URLSearchParams(location.hash.slice(1)).get('token');history.replaceState(null,'',location.pathname);
try{if(token)sessionStorage.setItem(key,token);else token=sessionStorage.getItem(key)}catch{}
if(!token){document.getElementById('status').textContent='Open the private link printed by NYM. This page contains no review data.';return}
try{const r=await fetch('/review',{headers:{Authorization:'Bearer '+token},credentials:'omit',cache:'no-store',redirect:'error',referrerPolicy:'no-referrer'});if(!r.ok)throw Error();const html=await r.text();document.open();document.write(html);document.close()}
catch{document.getElementById('status').textContent='Review unavailable. Reopen the current private link from NYM.'}})();
</script>'''


def snapshot(path, expected_sha256=None):
    """Extract data, discard supplied scripts, and rerender the trusted repository template."""
    if path.suffix.lower() != '.html' or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('review must be a regular HTML file')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_PAGE:
            raise ValueError('review file limit exceeded')
        raw = source.read(MAX_PAGE + 1)
    if len(raw) > MAX_PAGE:
        raise ValueError('review file limit exceeded')
    if expected_sha256 is not None and not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256):
        raise ValueError('review changed before snapshot; no server started')
    blocks = re.findall(r'<script type="application/json" id="data">(.*?)</script>', raw.decode('utf-8'), re.S)
    if len(blocks) != 1:
        raise ValueError('expected a NYM review data block')
    data = json.loads(blocks[0], object_pairs_hook=unique_object)
    if (not isinstance(data, dict) or set(data) != {'discovery', 'review'}
            or data['discovery'].get('schema') != 'nym.terms.discovery.v1'
            or data['review'].get('schema') != 'nym.terms.review.v1'):
        raise ValueError('expected NYM vocabulary review data')
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    template = (Path(__file__).resolve().parents[1] / 'src/terms_review.html').read_text()
    page = template.replace('__NYM_DATA__', encoded).encode('utf-8')
    if len(page) > MAX_PAGE:
        raise ValueError('rendered review file limit exceeded')
    return page


def allowed_bind(value):
    address = ipaddress.ip_address(value)
    if not (address.is_loopback or address in ipaddress.ip_network('100.64.0.0/10')
            or address in ipaddress.ip_network('fd7a:115c:a1e0::/48')):
        raise ValueError('only loopback or tailnet bindings are permitted')
    return str(address)


def tailnet_address():
    result = subprocess.run(['tailscale', 'ip', '-4'], capture_output=True, timeout=5)
    value = result.stdout.decode('ascii').strip()
    if result.returncode or len(value) > 64 or ipaddress.ip_address(value) not in ipaddress.ip_network('100.64.0.0/10'):
        raise ValueError('tailnet IPv4 address unavailable')
    return allowed_bind(value)


def header_snapshot(connection):
    """Bound total header bytes and absolute time; stdlib parses the resulting request."""
    deadline = time.monotonic() + 2
    raw = bytearray()
    while len(raw) < MAX_HEADERS:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('request header timeout')
        connection.settimeout(remaining)
        chunk = connection.recv(min(1024, MAX_HEADERS - len(raw)))
        if not chunk:
            raise ValueError('incomplete headers')
        raw.extend(chunk)
        end = raw.find(b'\r\n\r\n')
        if end >= 0:
            if end + 4 != len(raw) or any(byte == 0 for byte in raw):
                raise ValueError('body/pipelining or invalid header bytes')
            return bytes(raw)
    raise ValueError('request header limit exceeded')


class Handler(BaseHTTPRequestHandler):
    server_version = 'NYM'
    sys_version = ''

    def log_message(self, *_):
        pass  # Never log URLs, capabilities, peers, source identities or content.

    def handle(self):
        self.close_connection = True
        try:
            headers = header_snapshot(self.connection)
            self.connection.settimeout(2)
            original = self.rfile
            self.rfile = io.BytesIO(headers)
            original.close()
            self.handle_one_request()
        except (OSError, ValueError):
            pass  # Failed connection, no content, no logs; listener remains available.

    def handle_expect_100(self):
        self.send_error(417)
        return False

    def send_error(self, code, message=None, explain=None):
        self.reply(code, b'Request rejected', 'text/plain')

    def reply(self, code, body, content_type='text/html; charset=utf-8'):
        self.send_response(code)
        for key, value in [('Content-Type', content_type), ('Content-Length', str(len(body))),
                           ('Cache-Control', 'no-store'), ('Referrer-Policy', 'no-referrer'),
                           ('X-Content-Type-Options', 'nosniff'), ('X-Frame-Options', 'DENY'),
                           ('Content-Security-Policy', CSP), ('Connection', 'close')]:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def valid_headers(self):
        auth = self.headers.get_all('Authorization', [])
        length = self.headers.get_all('Content-Length', [])
        return (len(auth) <= 1 and len(length) <= 1 and length in [[], ['0']]
                and not self.headers.get_all('Transfer-Encoding') and not self.headers.get_all('Expect'))

    def do_GET(self):
        if self.request_version not in ['HTTP/1.0', 'HTTP/1.1'] or not self.valid_headers():
            self.reply(400, b'Request rejected', 'text/plain')
        elif self.path == '/':
            self.reply(200, BOOTSTRAP)
        elif self.path == '/review':
            supplied = self.headers.get('Authorization', '').encode('utf-8')
            if hmac.compare_digest(supplied, b'Bearer ' + self.server.token.encode('ascii')):
                self.reply(200, self.server.page)
            else:
                self.reply(403, b'Private review requires the current capability', 'text/plain')
        else:
            self.reply(404, b'Not found', 'text/plain')

    def do_POST(self):
        self.reply(405, b'Read-only review server', 'text/plain')


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 8

    def __init__(self, address, page, token):
        self.slots = threading.BoundedSemaphore(8)
        self.page = page
        self.token = token
        if ':' in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, *_):
        pass  # No raw traceback/request/source/capability diagnostics.


def main(argv=None):
    parser = argparse.ArgumentParser(description='Share only a read-only NYM review, not a directory. Private capability link; no public funnel.')
    parser.add_argument('html', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--tailnet', action='store_true')
    group.add_argument('--bind', default=None)
    parser.add_argument('--port', type=int, default=8521)
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--json', action='store_true', help='Print a private ready URL as JSON (contains capability)')
    parser.add_argument('--yaml', action='store_true', help='Print private ready URL as YAML (contains capability)')
    parser.add_argument('--expected-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.port <= 65535:
            raise ValueError('invalid port')
        address = tailnet_address() if args.tailnet else allowed_bind(args.bind or '127.0.0.1')
        page = snapshot(args.html, args.expected_sha256)
        token = secrets.token_urlsafe(32)
        with Server((address, args.port), page, token) as server:
            host = '[' + address + ']' if ':' in address else address
            url = f'http://{host}:{server.server_address[1]}/#token={token}'
            if args.yaml:
                print('status: serving\nurl: ' + json.dumps(url) + '\nread_only: true', flush=True)
            elif args.json:
                print(json.dumps({'status': 'serving', 'url': url, 'read_only': True}), flush=True)
            else:
                print('Private review URL (capability; share only with trusted devices):\n' + url + '\nRead-only snapshot. Ctrl-C stops sharing; existing artifacts remain unchanged.', flush=True)
            if args.open:
                try:
                    webbrowser.open(url)
                except Exception:
                    print('Could not open browser; use the printed private link.', file=sys.stderr)
            try:
                server.serve_forever(poll_interval=0.1)
            except KeyboardInterrupt:
                pass
        return 0
    except Exception:
        print('error: private review server could not start; no directory or public listener exposed', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
