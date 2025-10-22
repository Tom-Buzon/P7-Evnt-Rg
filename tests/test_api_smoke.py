import subprocess, time, httpx

def test_api_smoke():
    # Simple smoke test: start server, hit /docs, then stop.
    proc = subprocess.Popen(["poetry","run","uvicorn","src.api.app:app","--port","8010"])
    try:
        time.sleep(2)
        r = httpx.get("http://127.0.0.1:8010/docs", timeout=10)
        assert r.status_code == 200
    finally:
        proc.terminate()
