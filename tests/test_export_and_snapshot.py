# pylint: disable=protected-access,invalid-name  # fakes mirror PyMISP / http.server names
from __future__ import annotations

import gzip
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scripts import export_events, snapshot_taxonomy


class MISPHandler(BaseHTTPRequestHandler):
    events = [
        {"Event": {"id": "1", "uuid": "u1", "info": "one"}},
        {"Event": {"id": "2", "uuid": "u2", "info": "two"}},
        {"Event": {"id": "3", "uuid": "u3", "info": "three"}},
    ]
    authorization_headers: list[str] = []

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        self.authorization_headers.append(self.headers.get("Authorization", ""))
        page = int(request["page"])
        limit = int(request["limit"])
        start = (page - 1) * limit
        body = json.dumps({"response": self.events[start : start + limit]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def test_streaming_export_against_local_misp_double(tmp_path: Path, monkeypatch, capsys):
    MISPHandler.authorization_headers = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), MISPHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = tmp_path / ".env"
        env.write_text(
            f"MISP_BASE_URL=http://127.0.0.1:{server.server_port}\n"
            "MISP_API_KEY=super-secret-test-key\n"
            "MISP_VERIFY_SSL=true\n",
            encoding="utf-8",
        )
        out = tmp_path / "events.jsonl.gz"
        monkeypatch.delenv("MISP_BASE_URL", raising=False)
        monkeypatch.delenv("MISP_API_KEY", raising=False)
        if hasattr(export_events._THREAD_LOCAL, "session"):
            del export_events._THREAD_LOCAL.session
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "export_events",
                "--env",
                str(env),
                "--out",
                str(out),
                "--page-size",
                "2",
                "--workers",
                "1",
                "--min-free-gb",
                "0",
            ],
        )
        assert export_events.main() == 0
        with gzip.open(out, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        assert [row["Event"]["id"] for row in rows] == ["1", "2", "3"]
        assert MISPHandler.authorization_headers == [
            "super-secret-test-key",
            "super-secret-test-key",
        ]
        assert oct(out.stat().st_mode & 0o777) == "0o600"
        captured = capsys.readouterr()
        assert "super-secret-test-key" not in captured.out + captured.err
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_export_detects_existing_complete_output(tmp_path: Path, monkeypatch, capsys):
    output = tmp_path / "events.jsonl.gz"
    output.write_bytes(b"existing")
    monkeypatch.setattr(sys, "argv", ["export_events", "--out", str(output)])
    assert export_events.main() == 0
    assert "use --overwrite" in capsys.readouterr().out


class FakeMISP:
    root_url = "https://misp.invalid"

    def taxonomies(self, pythonify=False):
        return [{"Taxonomy": {"id": "1", "namespace": "rsit"}}]

    def get_taxonomy(self, _id, pythonify=False):
        return {
            "Taxonomy": {
                "id": "1",
                "namespace": "rsit",
                "description": "RSIT",
                "enabled": True,
            },
            "entries": [
                {
                    "tag": 'rsit:fraud="phishing"',
                    "expanded": "Fraud: Phishing",
                }
            ],
        }

    def galaxies(self, withCluster=False, pythonify=False):
        return [{"Galaxy": {"id": "2", "type": "mitre-attack-pattern"}}]

    def get_galaxy(self, _id, withCluster=True, pythonify=False):
        return {
            "Galaxy": {
                "id": "2",
                "uuid": "g2",
                "name": "ATT&CK",
                "type": "mitre-attack-pattern",
                "enabled": True,
            },
            "GalaxyCluster": [
                {
                    "id": "3",
                    "uuid": "c3",
                    "type": "mitre-attack-pattern",
                    "value": "Phishing - T1566",
                    "tag_name": 'misp-galaxy:mitre-attack-pattern="Phishing - T1566"',
                    "GalaxyElement": [{"key": "external_id", "value": "T1566"}],
                }
            ],
        }

    def tags(self, pythonify=False):
        return {
            "Tag": [
                {"Tag": {"name": 'stone:incident-classification="phishing"'}},
                {"Tag": {"name": "tlp:red"}},
            ]
        }


def test_taxonomy_snapshot_with_fake_read_only_client(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(snapshot_taxonomy, "create_misp_client", lambda _env: FakeMISP())
    out = tmp_path / "taxonomy"
    monkeypatch.setattr(
        sys,
        "argv",
        ["snapshot_taxonomy", "--env", "unused.env", "--out-dir", str(out)],
    )
    assert snapshot_taxonomy.main() == 0
    latest = out / "snapshot.json"
    assert latest.is_symlink()
    value = json.loads(latest.read_text(encoding="utf-8"))
    assert value["summary"]["taxonomy_entry_count"] == 1
    assert value["summary"]["galaxy_cluster_count"] == 1
    assert value["custom_tags"] == ['stone:incident-classification="phishing"']
    target = latest.resolve()
    assert oct(target.stat().st_mode & 0o777) == "0o600"
