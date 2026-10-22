"""
P1-00 - Benchmark 3 phuong phap Discovery (thuc nghiem lap lai, do thoi gian + RAM).
VA chay that (full mode) de thu thap metadata 5.000 bai cho crawler.

Thiet ke:
  * Moi (phuong phap, lan lap) chay trong 1 TIEN TRINH RIENG => RAM dinh khong bi lan giua cac phuong phap.
  * Lap R lan (mac dinh 3), xen ke de giam sai lech do mang troi.
  * Che do "full": quet Kaggle lay ngay version -> goi API lay so version chuan -> ghi data/raw/discovery.jsonl

Cai dat:  pip install requests feedparser sickle pandas psutil
Chay benchmark: python discovery.py --mode benchmark --contact "email@example.com" ...
Chay that:      python discovery.py --mode full --id_start 2506.00788 --id_end 2506.05787 --contact "email" --kaggle_json ...
(moi co --xxx-yyy deu nhan them dang --xxx_yyy; ID la ID day du YYMM.NNNNN, dai co the di qua nhieu thang)
"""
import argparse, hashlib, json, math, os, platform, re, shlex, statistics as st, subprocess, sys, time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

DELAY = 3.0
OAI_URL = "https://oaipmh.arxiv.org/oai"
API_URL = "http://export.arxiv.org/api/query"
LIBS = ["pandas", "sickle", "psutil", "requests", "feedparser", "kaggle"]

# gia tri t hai phia 95% theo bac tu do (df = n - 1); df > 30 dung 1.96
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
       10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
       18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
       26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}

# ------------------------------------------------------------------ do RAM
def _psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None

def rss_mb():
    ps = _psutil()
    return round(ps.Process().memory_info().rss / 1e6, 1) if ps else None

def peak_ram_mb():
    """RAM dinh cua tien trinh (MB). Windows: psutil peak_wset; Linux/macOS: ru_maxrss."""
    ps = _psutil()
    if sys.platform.startswith("win") and ps:
        mi = ps.Process().memory_info()
        return round((getattr(mi, "peak_wset", None) or mi.rss) / 1e6, 1)
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round((v / 1e6) if sys.platform == "darwin" else (v / 1024), 1)
    except ImportError:
        return rss_mb()

ID_RE = re.compile(r"^(\d{4})\.(\d{5})$")

def parse_id(s):
    """ID day du dang YYMM.NNNNN -> (YYMM, so thu tu trong thang)."""
    m = ID_RE.match(s)
    if not m:
        sys.exit(f"ID khong hop le: {s!r} (can ID day du dang YYMM.NNNNN, VD: 2506.00788)")
    return m.group(1), int(m.group(2))

def range_size(id_start, id_end):
    """So ID trong [id_start, id_end] neu cung 1 thang. Dai di qua nhieu thang -> None
    (khong biet moi thang co bao nhieu bai chi tu 2 dau mut; so ID that lay tu snapshot Kaggle)."""
    (p1, n1), (p2, n2) = parse_id(id_start), parse_id(id_end)
    if (p1, n1) > (p2, n2):
        sys.exit(f"--id_start ({id_start}) phai <= --id_end ({id_end}).")
    return n2 - n1 + 1 if p1 == p2 else None

def make_ids(id_start, n):
    prefix, start = parse_id(id_start)
    return [f"{prefix}.{i:05d}" for i in range(start, start + n)]

# ------------------------------------------------------------------ cac worker
def worker_api(a, ids):
    import requests, feedparser
    base = rss_mb()
    ua = f"DS-Project-HCMUS-arXiv-crawler/0.1 (student project; contact: {a.contact})"
    out, req_s, nbytes = {}, [], 0
    t0 = time.perf_counter()
    for i in range(0, len(ids), a.batch):
        chunk = ids[i:i + a.batch]
        t1 = time.perf_counter()
        r = requests.get(API_URL, params={"id_list": ",".join(chunk), "max_results": len(chunk)},
                         headers={"User-Agent": ua}, timeout=60)
        r.raise_for_status()
        nbytes += len(r.content)
        entries = feedparser.parse(r.text).entries
        req_s.append(time.perf_counter() - t1)
        for e in entries:
            m = re.search(r"arxiv\.org/abs/(.+?)v(\d+)$", e.id)
            if m:
                out[m.group(1)] = int(m.group(2))
        if i + a.batch < len(ids):
            time.sleep(DELAY)
    return {"seconds": time.perf_counter() - t0, "requests": len(req_s), "found": len(out),
            "req_seconds": req_s, "bytes": nbytes, "versions": out, "baseline_mb": base}

def worker_oai(a, ids):
    import xml.etree.ElementTree as ET
    from sickle import Sickle
    base = rss_mb()
    ua = f"DS-Project-HCMUS-arXiv-crawler/0.1 (student project; contact: {a.contact})"
    ns = "{http://arxiv.org/OAI/arXivRaw/}"
    sk = Sickle(OAI_URL, headers={"User-Agent": ua}, max_retries=3)
    out, req_s, errors = {}, [], 0
    t0 = time.perf_counter()
    for k, aid in enumerate(ids):
        t1 = time.perf_counter()
        try:
            rec = sk.GetRecord(identifier=f"oai:arXiv.org:{aid}", metadataPrefix="arXivRaw")
            root = ET.fromstring(rec.raw)
            out[aid] = len(list(root.iter(f"{ns}version")))
        except Exception:
            errors += 1
        req_s.append(time.perf_counter() - t1)
        if k < len(ids) - 1:
            time.sleep(DELAY)
    return {"seconds": time.perf_counter() - t0, "requests": len(req_s), "found": len(out),
            "errors": errors, "req_seconds": req_s, "versions": out, "baseline_mb": base}

def worker_kaggle(a, ids):
    import pandas as pd
    base = rss_mb()
    wanted = set(ids)
    found, sample = 0, {}
    t0 = time.perf_counter()
    with pd.read_json(a.kaggle_json, lines=True, chunksize=a.chunksize, dtype=False) as reader:
        for ch in reader:
            ch_str = ch["id"].astype(str)
            mask = (ch_str >= a.id_start) & (ch_str <= a.id_end)
            ch_filtered = ch[mask]
            found += len(ch_filtered)
            for _, row in ch_filtered.iterrows():
                if row["id"] in wanted:
                    sample[row["id"]] = len(row["versions"])
    return {"seconds": time.perf_counter() - t0, "found_full_range": found,
            "found_in_sample": len(sample), "versions": sample,
            "file_gb": round(os.path.getsize(a.kaggle_json) / 1e9, 2), "baseline_mb": base}

WORKERS = {"api": worker_api, "oai": worker_oai, "kaggle": worker_kaggle}

def run_worker(a):
    n = a.oai_n if a.worker == "oai" else a.n
    ids = make_ids(a.id_start, n)
    res = WORKERS[a.worker](a, ids)
    res["peak_mb"] = peak_ram_mb()
    print("RESULT " + json.dumps(res))

# ------------------------------------------------------------------ thong ke
def fmt(x, d=2): return "n/a" if x is None else f"{x:.{d}f}"
def tcrit(df): return T95.get(df, 1.96)
def stats(v):
    v = [x for x in v if x is not None]
    if not v: return None
    n = len(v)
    sd = st.stdev(v) if n > 1 else 0.0
    ci = tcrit(n - 1) * sd / math.sqrt(n) if n > 1 else None
    return {"mean": st.mean(v), "median": st.median(v), "std": sd, "min": min(v), "max": max(v), "n": n, "ci95": ci}

def est_block(total_req, q, extra=None):
    d = {"requests": total_req, "n_samples": q["n"], "mean": total_req * (q["mean"] + DELAY),
         "low": total_req * (q["min"] + DELAY), "high": total_req * (q["max"] + DELAY)}
    if q["ci95"] is not None:
        d["ci_low"] = total_req * (max(q["mean"] - q["ci95"], 0.0) + DELAY)
        d["ci_high"] = total_req * (q["mean"] + q["ci95"] + DELAY)
    if extra: d.update(extra)
    return d

# ------------------------------------------------------------------ thong tin tai lap
def lib_versions():
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for n in LIBS:
        try: out[n] = version(n)
        except PackageNotFoundError: out[n] = None
    return out

def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)), timeout=5)
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except Exception: return None

def sha256_file(path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def masked_argv():
    out, skip = [], False
    for tok in sys.argv[1:]:
        if skip:
            out.append("<email>")
            skip = False
        elif tok == "--contact":
            out.append(tok)
            skip = True
        else: out.append(tok)
    return out

def env_info(a):
    info = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "libs": lib_versions(), "git_commit": git_commit(), "argv": masked_argv(), "delay_s": DELAY}
    if a.kaggle_json and os.path.exists(a.kaggle_json):
        fs = os.stat(a.kaggle_json)
        info["kaggle_file"] = {
            "name": os.path.basename(a.kaggle_json), "size_bytes": fs.st_size,
            "mtime_utc": datetime.fromtimestamp(fs.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_version": a.kaggle_version, "snapshot_date": a.kaggle_snapshot_date, "sha256": None}
    return info

# ------------------------------------------------------------------ MODE FULL
def iso_utc(s):
    """Doi ngay RFC 2822 cua Kaggle ('Sun, 1 Jun 2025 12:00:00 GMT') sang ISO 8601 UTC giong API
    ('2025-06-01T12:00:00Z') de v1_date va version_dates cung dinh dang. Loi -> giu nguyen chuoi goc."""
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return s

def api_get(params, ua, retries=4):
    """GET arXiv API. Loi mang / HTTP != 200 -> thu lai voi backoff. Tra ve (response, so lan thu)."""
    import requests
    last = None
    for k in range(retries):
        try:
            r = requests.get(API_URL, params=params, headers={"User-Agent": ua}, timeout=60)
            if r.status_code == 200:
                return r, k + 1
            last = f"HTTP {r.status_code}"
        except requests.RequestException as ex:
            last = repr(ex)
        if k < retries - 1:
            print(f"   ! loi request ({last}), thu lai {k + 1}/{retries - 1} ...", flush=True)
            time.sleep(DELAY * (2 ** (k + 1)))
    raise RuntimeError(f"API that bai sau {retries} lan thu: {last}")

def api_fetch(ids, a, ua, api, req_log, label):
    """Goi API theo batch, ghi vao `api` va log tung request (de kiem tra duoc)."""
    import feedparser
    for i in range(0, len(ids), a.batch):
        chunk = ids[i:i + a.batch]
        t1 = time.perf_counter()
        r, attempts = api_get({"id_list": ",".join(chunk), "max_results": len(chunk)}, ua)
        entries = feedparser.parse(r.text).entries
        for e in entries:
            m = re.search(r"arxiv\.org/abs/(.+?)v(\d+)$", e.id)
            if m:
                api[m.group(1)] = {
                    "n_versions": int(m.group(2)),
                    "v1_date": e.get("published", ""),
                    "latest_version_date": e.get("updated", ""),
                    "title": " ".join(e.get("title", "").split()),
                    "authors": ", ".join(x.name for x in e.get("authors", []) if "name" in x),
                }
        req_log.append({"pass": label, "ids": len(chunk), "attempts": attempts, "entries": len(entries),
                        "seconds": round(time.perf_counter() - t1, 3)})
        print(f"   [{label}] {min(i + a.batch, len(ids))}/{len(ids)} ID ({req_log[-1]['seconds']}s)", flush=True)
        if i + a.batch < len(ids):
            time.sleep(DELAY)

def run_full(a):
    import pandas as pd
    print(f"--- CHAY DISCOVERY TOAN DAI ({a.id_start} -> {a.id_end}) ---")
    exp = range_size(a.id_start, a.id_end)          # None neu dai di qua nhieu thang
    ua = f"DS-Project-HCMUS-arXiv-crawler/0.1 (student project; contact: {a.contact})"
    env = env_info(a)
    warnings = []
    if _psutil() is None:
        warnings.append("Chua cai psutil: khong do duoc RAM (pip install psutil). Tren Windows RAM dinh can psutil.")
    ram = {"baseline_mb": rss_mb()}                  # RAM nen sau khi import pandas
    t_all = time.perf_counter()

    # ---- 1) quet Kaggle MOT LAN: ngay tung version
    print("1. Quet Kaggle de lay ngay tung version...", flush=True)
    t = time.perf_counter()
    kaggle = {}                                      # id -> {"v1": iso, "v2": iso, ...}
    with pd.read_json(a.kaggle_json, lines=True, chunksize=a.chunksize, dtype=False,
                      convert_dates=False) as reader:
        for ch in reader:
            ids_s = ch["id"].astype(str)
            sub = ch[(ids_s >= a.id_start) & (ids_s <= a.id_end)]   # ID cung do rong YYMM.NNNNN => so sanh chuoi dung
            for aid, vs in zip(sub["id"].astype(str), sub["versions"]):
                kaggle[aid] = {v["version"]: iso_utc(v["created"]) for v in vs} if isinstance(vs, list) else {}
    t_kaggle = time.perf_counter() - t
    ram["peak_after_kaggle_scan_mb"] = peak_ram_mb()
    kgb = os.path.getsize(a.kaggle_json) / 1e6
    print(f"   Tim thay {len(kaggle)} bai trong snapshot ({t_kaggle:.1f}s, doc {kgb / t_kaggle:.0f} MB/s, "
          f"RAM dinh {ram['peak_after_kaggle_scan_mb']} MB).", flush=True)

    # ---- tap ID can hoi API
    if exp is not None:                              # cung thang: sinh du dai, khong phu thuoc snapshot
        ids = make_ids(a.id_start, exp)
        missing_kaggle = [i for i in ids if i not in kaggle]
    else:                                            # nhieu thang: do dai moi thang chua biet => lay tu snapshot
        ids = sorted(kaggle)
        missing_kaggle = []
        warnings.append("Dai di qua nhieu thang: tap ID lay tu snapshot Kaggle (khong sinh duoc tu 2 dau mut).")
    if len(ids) != a.total_ids:
        warnings.append(f"Co {len(ids)} ID trong dai nhung --total-ids = {a.total_ids}.")
    if missing_kaggle:
        warnings.append(f"{len(missing_kaggle)} ID khong co trong snapshot Kaggle (van hoi API).")

    # ---- 2) goi API toan dai (batch)
    n_req_expected = math.ceil(len(ids) / a.batch)
    print(f"2. Goi arXiv API: {len(ids)} ID, batch {a.batch} => {n_req_expected} request...", flush=True)
    t = time.perf_counter()
    api, req_log = {}, []
    api_fetch(ids, a, ua, api, req_log, "lan 1")
    missing = [i for i in ids if i not in api]
    if missing:
        print(f"   {len(missing)} ID chua co ket qua, hoi lai 1 lan...", flush=True)
        time.sleep(DELAY)
        api_fetch(missing, a, ua, api, req_log, "hoi lai")
    missing_api = [i for i in ids if i not in api]
    if missing_api:
        warnings.append(f"{len(missing_api)} ID khong co tren API (bo qua, khong ghi vao jsonl): {missing_api[:10]}...")
    t_api = time.perf_counter() - t

    # ---- 3) doi chieu API vs Kaggle (API la chuan) + ghi jsonl
    print(f"3. Ghi ket qua ra {a.out} ...", flush=True)
    t = time.perf_counter()
    mismatch, v1_mismatch, written = [], [], 0
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for aid in ids:
            ai = api.get(aid)
            if not ai:
                continue
            vd = kaggle.get(aid, {})
            if aid in kaggle and len(vd) != ai["n_versions"]:
                mismatch.append({"id": aid, "api": ai["n_versions"], "kaggle": len(vd)})
            if "v1" in vd and vd["v1"] != ai["v1_date"]:
                v1_mismatch.append(aid)
            f.write(json.dumps({"arxiv_id": aid, "n_versions": ai["n_versions"], "v1_date": ai["v1_date"],
                                "latest_version_date": ai["latest_version_date"], "version_dates": vd,
                                "title": ai["title"], "authors": ai["authors"]}, ensure_ascii=False) + "\n")
            written += 1
    os.replace(tmp, a.out)
    t_write = time.perf_counter() - t
    t_total = time.perf_counter() - t_all            # wall time end-to-end (chua gom thoi gian TAI file Kaggle)
    ram["peak_final_mb"] = peak_ram_mb()             # RAM dinh toan tien trinh (max qua ca 3 giai doan)
    ram["chunksize"] = a.chunksize
    ram["limit_mb"] = a.ram_limit_mb
    if ram["peak_final_mb"] and ram["peak_final_mb"] > a.ram_limit_mb:
        warnings.append(f"RAM dinh {ram['peak_final_mb']} MB vuot nguong {a.ram_limit_mb:.0f} MB, can giam --chunksize.")
    speed = {"kaggle_read_mb_per_s": round(kgb / t_kaggle, 1), "kaggle_rows_in_range_per_s": round(len(kaggle) / t_kaggle, 1),
             "api_ids_per_s": round(len(api) / t_api, 2) if t_api else None,
             "api_mean_request_s": round(sum(r["seconds"] for r in req_log) / len(req_log), 3) if req_log else None,
             "delay_between_requests_s": DELAY}

    # ---- bao cao (kiem tra duoc): ghi ra file de dua vao docs/discovery_report.md
    dl = a.download_seconds
    cfg = vars(a).copy()
    cfg["contact"] = "<email>"
    if a.hash_kaggle and "kaggle_file" in env:
        env["kaggle_file"]["sha256"] = sha256_file(a.kaggle_json)
    result = {
        "mode": "full", "config": cfg, "environment": env,
        "timing_s": {"kaggle_scan": round(t_kaggle, 2), "api_total": round(t_api, 2), "write": round(t_write, 2),
                     "total_wall_end_to_end": round(t_total, 2),
                     "kaggle_download_given": dl,
                     "total_incl_download": round(t_total + dl, 2) if dl else None},
        "ram_mb": ram, "speed": speed,
        "api_requests": {"expected_batches": n_req_expected, "actual_requests": len(req_log),
                         "http_attempts": sum(r["attempts"] for r in req_log),
                         "seconds_stats": stats([r["seconds"] for r in req_log]), "log": req_log},
        "counts": {"ids_in_range": len(ids), "in_kaggle": len(kaggle), "in_api": len(api), "written": written,
                   "missing_in_kaggle": len(missing_kaggle), "missing_in_api": len(missing_api)},
        "check": {"n_versions_mismatch_api_vs_kaggle": mismatch, "v1_date_mismatch_api_vs_kaggle": v1_mismatch},
        "missing_in_kaggle": missing_kaggle, "missing_in_api": missing_api,
        "output": {"path": a.out, "sha256": sha256_file(a.out)}, "warnings": warnings,
    }
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    print("--- HOAN THANH ---")
    print(f"Ghi {written}/{len(ids)} bai ra {a.out}")
    print(f"Thoi gian: Kaggle {t_kaggle:.1f}s | API {t_api:.1f}s ({len(req_log)} request) | ghi {t_write:.1f}s")
    print(f"TONG wall time end-to-end: {t_total:.1f} s (~{t_total / 60:.2f} phut), chua tinh thoi gian tai file Kaggle"
          + (f"; gom tai: {t_total + dl:.0f} s" if dl else ""))
    print(f"RAM: nen {ram['baseline_mb']} MB -> dinh {ram['peak_final_mb']} MB (chunksize {a.chunksize}, nguong {a.ram_limit_mb:.0f} MB)")
    print(f"Toc do: Kaggle {speed['kaggle_read_mb_per_s']} MB/s | API {speed['api_ids_per_s']} ID/s "
          f"(TB {speed['api_mean_request_s']}s/request + nghi {DELAY}s)")
    print(f"Doi chieu n_versions API vs Kaggle: {len(mismatch)} ID lech (lay API lam chuan) {[m['id'] for m in mismatch[:10]]}")
    print(f"Doi chieu ngay v1 API vs Kaggle: {len(v1_mismatch)} ID lech")
    for w in warnings:
        print("CANH BAO:", w)
    print(f"Bao cao do (kiem tra duoc): {a.report}")

# ------------------------------------------------------------------ MODE BENCHMARK
def orchestrate(a):
    methods = [m.strip() for m in a.methods.split(",")]
    passthrough = ["--id-start", a.id_start, "--n", str(a.n), "--oai-n", str(a.oai_n),
                   "--id-end", a.id_end, "--batch", str(a.batch), "--contact", a.contact,
                   "--chunksize", str(a.chunksize)]
    if a.kaggle_json:
        passthrough += ["--kaggle-json", a.kaggle_json]
    env = env_info(a)
    warnings = []
    exp_range = range_size(a.id_start, a.id_end)
    if exp_range is None:      # dai di qua nhieu thang: khong tinh duoc tu 2 dau mut
        exp_range = a.total_ids
        warnings.append(f"Dai [{a.id_start}, {a.id_end}] di qua nhieu thang: lay so ID ky vong = --total-ids ({a.total_ids}), "
                        "chua tu kiem duoc. Che do full se dem so ID that tu snapshot.")
    elif exp_range != a.total_ids:
        warnings.append(f"Dai ID [{a.id_start}, {a.id_end}] co {exp_range} ID nhung --total-ids = {a.total_ids}.")
    if a.repeats < 3:
        warnings.append(f"Chi lap {a.repeats} lan: std va khoang tin cay khong dang tin (nen >= 3).")

    runs = {m: [] for m in methods}
    failed = {m: 0 for m in methods}
    for r in range(a.repeats):
        for m in methods:
            print(f"[lan {r + 1}/{a.repeats}] {m} ...", flush=True)
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", m] + passthrough,
                               capture_output=True, text=True)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            if p.returncode != 0 or line is None:
                failed[m] += 1
                print(f"  LOI ({m}):", (p.stderr or p.stdout)[-400:])
                continue
            runs[m].append(json.loads(line[7:]))
            print(f"  xong: {runs[m][-1]['seconds']:.2f}s, peak RAM {runs[m][-1]['peak_mb']} MB", flush=True)
            if m in ("api", "oai"):
                time.sleep(DELAY)

    if a.hash_kaggle and "kaggle_file" in env:
        print("Tinh SHA-256 file Kaggle ...", flush=True)
        env["kaggle_file"]["sha256"] = sha256_file(a.kaggle_json)

    exp_sample = {"api": a.n, "oai": a.oai_n, "kaggle": a.n}
    S = {}
    for m, rs in runs.items():
        if not rs:
            continue
        S[m] = {"seconds": stats([x["seconds"] for x in rs]),
                "peak_mb": stats([x["peak_mb"] for x in rs]),
                "baseline_mb": stats([x["baseline_mb"] for x in rs]),
                "ok_runs": len(rs), "failed_runs": failed[m],
                "found": [x["found_in_sample"] if m == "kaggle" else x["found"] for x in rs],
                "errors": sum(x.get("errors", 0) for x in rs)}
        if m == "kaggle":
            S[m]["found_range"] = [x["found_full_range"] for x in rs]
        if m in ("api", "oai"):
            S[m]["req_seconds"] = stats([s for x in rs for s in x["req_seconds"]])
            S[m]["requests_per_run"] = rs[0]["requests"]

    est = {}
    if "api" in S:
        est["api"] = est_block(math.ceil(a.total_ids / a.batch), S["api"]["req_seconds"])
    if "oai" in S:
        est["oai"] = est_block(a.total_ids, S["oai"]["req_seconds"])
    if "kaggle" in S:
        s = S["kaggle"]["seconds"]
        dl = a.download_seconds or 0.0
        est["kaggle"] = {"requests": 0, "n_samples": s["n"], "mean": s["mean"], "low": s["min"], "high": s["max"],
                         "download": dl, "end_to_end": s["mean"] + dl}
        if s["ci95"] is not None:
            est["kaggle"]["ci_low"] = max(s["mean"] - s["ci95"], 0.0)
            est["kaggle"]["ci_high"] = s["mean"] + s["ci95"]

    votes = {}
    for m, rs in runs.items():
        for x in rs:
            for i, v in x["versions"].items():
                votes.setdefault(i, {}).setdefault(m, set()).add(v)
    common = {i: d for i, d in votes.items() if len(d) >= 2}
    mismatch = [i for i, d in common.items() if len({v for s in d.values() for v in s}) > 1]
    unstable = [i for i, d in votes.items() if any(len(s) > 1 for s in d.values())] 

    cfg = vars(a).copy()
    cfg["contact"] = "<email>"
    result = {"config": cfg, "environment": env, "runs": runs, "summary": S, "estimate": est,
              "compared_ids": len(common), "mismatch_ids": mismatch, "unstable_ids": unstable,
              "warnings": warnings}
    names = {"api": "arXiv API", "oai": "OAI-PMH GetRecord", "kaggle": "Kaggle (quet ca dai)"}
    L = []
    L.append(f"### Ket qua do ({a.repeats} lan lap, mau {a.n} ID API/Kaggle, {a.oai_n} ID OAI)\n")
    L.append("| Phuong phap | Thoi gian mau, s (mean ± std) | Median | [min, max] | RAM dinh, MB (mean ± std) | "
             "RAM nen, MB | So request | Tim thay/mau | Lan OK |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        if m not in S:
            continue
        t, r, b = S[m]["seconds"], S[m]["peak_mb"], S[m]["baseline_mb"]
        nreq = S[m].get("requests_per_run", 0)
        fo = S[m]["found"]
        found = f"{fo[0]}/{exp_sample[m]}" if len(set(fo)) == 1 else f"{min(fo)}-{max(fo)}/{exp_sample[m]}"
        if S[m]["errors"]:
            found += f" ({S[m]['errors']} loi)"
        L.append(f"| {names[m]} | {fmt(t['mean'])} ± {fmt(t['std'])} | {fmt(t['median'])} | "
                 f"[{fmt(t['min'])}, {fmt(t['max'])}] | "
                 f"{fmt(r['mean'] if r else None, 1)} ± {fmt(r['std'] if r else None, 1)} | "
                 f"{fmt(b['mean'] if b else None, 1)} | {nreq} | {found} | "
                 f"{S[m]['ok_runs']}/{S[m]['ok_runs'] + S[m]['failed_runs']} |")

    L.append(f"\n### Uoc luong cho {a.total_ids} ID\n")
    L.append("| Phuong phap | So request | Trung binh | KTC 95% cua trung binh | [min, max] (cuc doan) | So mau |")
    L.append("|---|---|---|---|---|---|")
    for m in methods:
        if m not in est:
            continue
        e = est[m]
        extra = f" (+ tai {e['download']:.0f}s = {e['end_to_end']:.0f}s end-to-end)" if m == "kaggle" else ""
        ci = f"{e['ci_low']:.0f} - {e['ci_high']:.0f} s" if "ci_low" in e else "n/a"
        unit = "lan do" if m == "kaggle" else "request"
        L.append(f"| {names[m]} | {e['requests']} | {e['mean']:.0f} s ({e['mean']/60:.1f} phut){extra} | "
                 f"{ci} | {e['low']:.0f} - {e['high']:.0f} s | {e['n_samples']} {unit} |")
    low_n = [names[m] for m in ("api", "oai") if m in est and est[m]["n_samples"] < 10]
    if low_n:
        warnings.append("Qua it mau request de uoc luong dang tin: " + ", ".join(low_n) +
                        " (< 10 request). Khoang [min, max] va KTC rat rong; can them --n hoac --repeats.")

    L.append(f"\n### Doi chieu so version\n\nSo ID duoc doi chieu giua >= 2 nguon: {len(common)}; so ID lech: "
             f"{len(mismatch)} {mismatch if mismatch else ''}\n"
             f"\nSo ID doi version giua cac lan lap cua cung 1 nguon: {len(unstable)} {unstable if unstable else ''}")

    L.append("\n### Kiem tra quy quyet dinh\n")
    if {"api", "kaggle", "oai"} <= set(est):
        combo = est["api"]["mean"] + est["kaggle"]["mean"]
        L.append(f"- Phuong an ket hop (API toan dai + quet Kaggle), chua tinh tai: {combo:.0f} s ({combo/60:.1f} phut); "
                 f"tinh ca tai lan dau: {combo + est['kaggle']['download']:.0f} s.")
        L.append(f"- OAI GetRecord toan dai: {est['oai']['mean']:.0f} s ({est['oai']['mean']/3600:.1f} gio) "
                 f"=> gap {est['oai']['mean']/combo:.1f} lan (chua tinh tai), "
                 f"{est['oai']['mean']/(combo + est['kaggle']['download']):.1f} lan (tinh ca tai).")
        be = 1 - combo / est["oai"]["mean"]
        be_dl = 1 - (combo + est["kaggle"]["download"]) / est["oai"]["mean"]
        L.append(f"- Diem hoa von ve thoi gian cua phuong an ket hop so voi OAI don le: ti le lech ≈ {be*100:.0f}% "
                 f"(chua tinh tai), ≈ {be_dl*100:.0f}% (tinh ca tai).")
    if "kaggle" in S:
        fr = S["kaggle"]["found_range"]
        ok = all(x == exp_range for x in fr)
        L.append(f"- Kaggle tim thay {fr} bai trong dai (ky vong {exp_range}) => {'DAT' if ok else 'KHONG DAT, snapshot thieu bai'}.")
        if S["kaggle"]["peak_mb"]:
            pk = S["kaggle"]["peak_mb"]["max"]
            ok = pk <= a.ram_limit_mb
            L.append(f"- RAM dinh Kaggle (max cac lan) = {pk:.0f} MB; nguong rang buoc = {a.ram_limit_mb:.0f} MB "
                     f"(~50% RAM Colab CPU) => {'DAT (khong phai rang buoc)' if ok else 'VUOT (la rang buoc, can giam chunksize)'}.")
    for m in ("api", "oai"):
        if m in S:
            full = all(x == exp_sample[m] for x in S[m]["found"])
            L.append(f"- {names[m]} tim thay du {exp_sample[m]} ID moi lan => {'DAT' if full else 'KHONG DAT, thieu bai hoac loi request'}.")
    L.append(f"- Nhat quan version giua cac nguon: {'DAT' if not mismatch else 'KHONG DAT, xem danh sach lech'}.")

    L.append("\n### Thong tin tai lap\n")
    L.append(f"- Chay luc (UTC): {env['run_utc']}; Python {env['python']}; {env['platform']}; git: {env['git_commit'] or 'n/a'}")
    L.append("- Thu vien: " + ", ".join(f"{k} {v or 'chua cai'}" for k, v in env["libs"].items()))
    L.append(f"- Lenh: python discovery.py {' '.join(shlex.quote(x) for x in env['argv'])}")
    kf = env.get("kaggle_file")
    if kf:
        L.append(f"- File Kaggle: {kf['name']}, {kf['size_bytes']/1e9:.2f} GB, mtime (UTC) {kf['mtime_utc']}, "
                 f"dataset version: {kf['dataset_version'] or 'CHUA KHAI BAO (--kaggle-version)'}, "
                 f"ngay snapshot: {kf['snapshot_date'] or 'CHUA KHAI BAO (--kaggle-snapshot-date)'}, "
                 f"SHA-256: {kf['sha256'] or 'chua tinh (--hash-kaggle)'}")
    if warnings:
        L.append("\n### Canh bao\n")
        L += [f"- {w}" for w in warnings]
    with open("discovery_benchmark_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    md = "\n".join(L)
    with open("discovery_results.md", "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print("\n" + md)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    def add(*names, **kw):
        """Moi --xxx-yyy nhan them --xxx_yyy (README dung gach duoi)."""
        al = []
        for n in names:
            al.append(n)
            if "-" in n[2:]:
                al.append("--" + n[2:].replace("-", "_"))
        ap.add_argument(*al, **kw)
    add("--mode", choices=["benchmark", "full"], default="benchmark", help="Che do chay")
    add("--id-start", default="2506.00788", help="ID day du bat dau (VD: 2506.00788)")
    add("--id-end", default="2506.05787", help="ID day du ket thuc (bao gom); co the khac thang voi --id-start")
    add("--n", type=int, default=50, help="so ID mau cho API trong che do benchmark")
    add("--oai-n", type=int, default=20, help="so ID mau cho OAI GetRecord")
    add("--batch", type=int, default=50)
    add("--total-ids", type=int, default=5000)
    add("--repeats", type=int, default=3)
    add("--methods", default="api,oai,kaggle")
    add("--contact", required=True)
    add("--kaggle-json", default=None)
    add("--kaggle-version", default=None)
    add("--kaggle-snapshot-date", default=None)
    add("--hash-kaggle", action="store_true")
    add("--download-seconds", type=float, default=None)
    add("--chunksize", type=int, default=200_000)
    add("--ram-limit-mb", type=float, default=6000)
    add("--out", default="data/raw/discovery.jsonl", help="(mode full) file jsonl ban giao cho P2-01")
    add("--report", default="discovery_full_result.json", help="(mode full) file bao cao do thoi gian/doi chieu")
    add("--worker", default=None, help="(noi bo)")
    a = ap.parse_args()
    range_size(a.id_start, a.id_end)   # kiem tra dinh dang + thu tu ID som
    
    if a.worker:
        run_worker(a)
    elif a.mode == "full":
        if not a.kaggle_json:
            sys.exit("Che do full can cung cap --kaggle-json.")
        run_full(a)
    else:
        if "kaggle" in a.methods and not a.kaggle_json:
            sys.exit("Thieu --kaggle-json (hoac bo 'kaggle' khoi --methods).")
        if a.kaggle_json and not os.path.exists(a.kaggle_json):
            sys.exit(f"Khong thay file: {a.kaggle_json}")
        if _psutil() is None and sys.platform.startswith("win"):
            sys.exit("Windows can psutil de do RAM dinh: pip install psutil")
        orchestrate(a)