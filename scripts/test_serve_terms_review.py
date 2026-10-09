#!/usr/bin/env python3
"""Synthetic protocol/auth/snapshot proofs. Never reads real reviews or sessions."""
import http.client
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import serve_terms_review as serve


class ReviewServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.html = self.root / 'review.html'
        self.marker = 'SYNTHETIC_PRIVATE_CONTEXT'
        data = {'discovery': {'schema': 'nym.terms.discovery.v1', 'candidates': [], 'marker': self.marker},
                'review': {'schema': 'nym.terms.review.v1', 'decisions': []}}
        self.html.write_text('<script>window.untrusted=true</script><script type="application/json" id="data">' + json.dumps(data) + '</script>')
        self.original = self.html.read_bytes()
        self.token = 'synthetic-capability-only-not-a-real-token'
        self.server = serve.Server(('127.0.0.1', 0), serve.snapshot(self.html), self.token)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)
        self.temp.cleanup()

    def request(self, path, token=None, method='GET'):
        client = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=4)
        headers = {'Authorization': 'Bearer ' + token} if token else {}
        try:
            client.request(method, path, headers=headers)
            reply = client.getresponse()
            return reply.status, dict(reply.getheaders()), reply.read()
        finally:
            client.close()

    def raw(self, head):
        with socket.create_connection(('127.0.0.1', self.server.server_port), timeout=4) as connection:
            connection.sendall(head)
            chunks = []
            while True:
                try: chunk = connection.recv(8192)
                except ConnectionResetError: break
                if not chunk: break
                chunks.append(chunk)
            return b''.join(chunks)

    def test_bootstrap_is_public_but_no_private_data_auth_required(self):
        code, headers, body = self.request('/')
        self.assertEqual(code, 200);self.assertNotIn(self.marker.encode(), body)
        self.assertIn(b'history.replaceState', body)
        for token in [None, 'incorrect', self.token + 'x']:
            code, headers, body = self.request('/review', token)
            self.assertEqual(code, 403);self.assertNotIn(self.marker.encode(), body)
        code, headers, body = self.request('/review', self.token)
        self.assertEqual(code, 200);self.assertIn(self.marker.encode(), body)
        self.assertNotIn(b'window.untrusted=true', body)
        self.assertEqual(self.html.read_bytes(), self.original)
        self.assertEqual({k: headers[k] for k in ['Cache-Control', 'Referrer-Policy', 'X-Content-Type-Options', 'X-Frame-Options', 'Content-Security-Policy']},
                         {'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY', 'Content-Security-Policy': serve.CSP})

    def test_directory_deck_query_and_mutating_routes_are_never_exposed(self):
        (self.root / 'discovery.json').write_text(self.marker)
        for path in ['/discovery.json', '/config.toml', '/../review.html', '/%2e%2e/review.html', '/?token=' + self.token, '/review?token=' + self.token]:
            code, _, body = self.request(path, self.token)
            self.assertEqual(code, 404);self.assertNotIn(self.marker.encode(), body)
        self.assertEqual(self.request('/review', self.token, method='POST')[0], 405)

    def test_duplicate_auth_body_transfer_and_oversized_headers_fail_closed(self):
        auth = b'Authorization: Bearer ' + self.token.encode() + b'\r\n'
        for extra in [auth, b'Content-Length: 1\r\n', b'Transfer-Encoding: chunked\r\n', b'Expect: 100-continue\r\n']:
            reply = self.raw(b'GET /review HTTP/1.1\r\nHost: test\r\n' + auth + extra + b'\r\n')
            self.assertNotIn(self.marker.encode(), reply)
        for raw in [b'GET /review HTTP/1.1\r\nX: ' + b'A' * serve.MAX_HEADERS,
                    b'GET /review HTTP/1.1\r\nHost: test\x00\r\n\r\n',
                    b'GET /review HTTP/1.1\r\n' + auth + b'\r\nBODY']:
            self.assertNotIn(self.marker.encode(), self.raw(raw))
        self.assertEqual(self.request('/review', self.token)[0], 200)

    def test_snapshot_is_immutable_and_only_review_data_is_executed(self):
        self.html.write_text('changed after snapshot')
        self.assertIn(self.marker.encode(), self.request('/review', self.token)[2])
        self.assertNotIn(b'window.untrusted', self.server.page)

    def test_native_expected_hash_refuses_verification_snapshot_races(self):
        expected=hashlib.sha256(self.original).hexdigest()
        self.assertEqual(serve.snapshot(self.html,expected),self.server.page)
        self.html.write_bytes(self.original+b'\n ')
        with self.assertRaises(ValueError): serve.snapshot(self.html,expected)

    def test_source_directory_symlink_bad_data_and_size_fail(self):
        for path in [self.root, self.root / 'missing.html']:
            with self.assertRaises(Exception): serve.snapshot(path)
        if hasattr(os, 'symlink'):
            link = self.root / 'link.html';link.symlink_to(self.html)
            with self.assertRaises(ValueError): serve.snapshot(link)
        for body in ['not a review', '<script type="application/json" id="data">{"discovery":{},"discovery":{},"review":{}}</script>']:
            self.html.write_text(body)
            with self.assertRaises(ValueError): serve.snapshot(self.html)
        with self.html.open('wb') as target: target.truncate(serve.MAX_PAGE + 1)
        with self.assertRaises(ValueError): serve.snapshot(self.html)

    def test_bind_policy_and_unavailable_tailnet_never_fall_back(self):
        for address in ['127.0.0.1', '::1', '100.64.0.12', 'fd7a:115c:a1e0::1234']:
            self.assertEqual(serve.allowed_bind(address), address)
        for address in ['0.0.0.0', '::', '192.168.1.1', '8.8.8.8', '100.128.0.1']:
            with self.assertRaises(ValueError): serve.allowed_bind(address)
        class Result:
            returncode = 0
            stdout = b'127.0.0.1\n'
        with patch('serve_terms_review.subprocess.run', return_value=Result()):
            with self.assertRaises(ValueError): serve.tailnet_address()
        with self.assertRaises(OSError): serve.Server(self.server.server_address, b'page', 'token')

    def test_slow_request_deadline_and_concurrency_are_bounded(self):
        connections = [socket.create_connection(('127.0.0.1', self.server.server_port), timeout=4) for _ in range(8)]
        try:
            for c in connections: c.sendall(b'GET /')
            slots_full = False
            for _ in range(50):
                if not self.server.slots.acquire(blocking=False):
                    slots_full = True
                    break
                self.server.slots.release()
                time.sleep(.01)
            self.assertTrue(slots_full)
            with socket.create_connection(('127.0.0.1', self.server.server_port), timeout=.5) as extra:
                try: self.assertEqual(extra.recv(1), b'')
                except ConnectionResetError: pass
            deadline = time.monotonic() + 3
            for c in connections:
                c.settimeout(max(.1, deadline - time.monotonic()))
                self.assertEqual(c.recv(1), b'')
            self.assertEqual(self.request('/review', self.token)[0], 200)
        finally:
            for c in connections: c.close()


if __name__ == '__main__':
    unittest.main()
