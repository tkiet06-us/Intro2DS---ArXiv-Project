"""
P1-00 - Benchmark 3 phuong phap Discovery (thuc nghiem lap lai, do thoi gian + RAM).
VA chay that (full mode) de thu thap metadata 5.000 bai cho crawler.

Thiet ke:
  * Moi (phuong phap, lan lap) chay trong 1 TIEN TRINH RIENG => RAM dinh khong bi lan giua cac phuong phap.
  * Lap R lan (mac dinh 3), xen ke de giam sai lech do mang troi.
  * Che do "full": quet Kaggle lay ngay version -> goi API lay so version chuan -> ghi data/raw/discovery.jsonl

Cai dat:  pip install requests feedparser sickle pandas psutil
Chay benchmark: python discovery.py --mode benchmark --contact "email@example.com" ...
Chay that:      python discovery.py --mode full --ranges ranges.txt --total-ids 25000 --contact "email" --kaggle_json ...
  (hoac 1 dai: --id_start 2506.00788 --id_end 2506.05787)
ranges.txt: moi dong 1 dai "ID_dau ID_cuoi" (dong trong hoac sau dau # bi bo qua). Cac dai khong duoc chong len nhau.
Che do full co CHECKPOINT/RESUME: sau moi batch API ghi noi tiep vao --out; chay lai CUNG LENH se bo qua ID da co.
  Muon chay lai tu dau: them --fresh (hoac doi --out).
Va ngay version (SAU khi chay full, khong can Kaggle/API):
  python discovery.py --mode patch --out data/raw/discovery_25k.jsonl --contact "email"
  Bai thieu dung 1 version cuoi -> dien tu latest_version_date (0 request); bai thieu >= 2 version -> OAI GetRecord.
  Chay lai nhieu lan duoc (da va thi bo qua). Ket qua ghi vao --patch-report (mac dinh discovery_patch_result.json).
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

def load_ranges(a):
    """Doc danh sach dai ID: tu file --ranges (moi dong 'start end') hoac tu --id-start/--id-end.
    Kiem tra dinh dang, thu tu, chong lan. Tra ve list[(start, end)] da sap xep."""
    if a.ranges_file:
        if not os.path.exists(a.ranges_file):
            sys.exit(f"Khong thay file dai ID: {a.ranges_file}")
        ranges = []
        with open(a.ranges_file, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.split("#")[0].strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) != 2:
                    sys.exit(f"{a.ranges_file}, dong {ln}: can dung 2 ID ('start end'), gap: {line!r}")
                ranges.append((parts[0], parts[1]))
        if not ranges:
            sys.exit(f"{a.ranges_file} khong co dai nao.")
    else:
        ranges = [(a.id_start, a.id_end)]
    for s, e in ranges:
        range_size(s, e)                  # kiem tra dinh dang + start <= end
    ranges.sort()
    for (s1, e1), (s2, e2) in zip(ranges, ranges[1:]):
        if s2 <= e1:
            sys.exit(f"Hai dai chong len nhau: [{s1}, {e1}] va [{s2}, {e2}].")
    return ranges

def in_ranges(ids_s, ranges):
    """Mask pandas: ID nam trong BAT KY dai nao (ID cung do rong YYMM.NNNNN => so sanh chuoi dung)."""
    mask = None
    for s, e in ranges:
        m = (ids_s >= s) & (ids_s <= e)
        mask = m if mask is None else (mask | m)
    return mask

def build_ids(ranges, kaggle):
    """Tap ID can hoi API. Dai trong 1 thang: sinh du ID (khong phu thuoc snapshot).
    Dai qua nhieu thang: lay ID tu snapshot Kaggle. Tra ve (ids da sap xep, ID thieu trong Kaggle, cac dai lay tu snapshot)."""
    ids, missing_kaggle, from_snapshot = [], [], []
    for s, e in ranges:
        n = range_size(s, e)
        if n is not None:
            part = make_ids(s, n)
            missing_kaggle += [i for i in part if i not in kaggle]
        else:
            part = [i for i in kaggle if s <= i <= e]
            from_snapshot.append((s, e))
        ids += part
    return sorted(set(ids)), missing_kaggle, from_snapshot

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

def api_fetch(ids, a, ua, api, req_log, label, sink=None):
    """Goi API theo batch, ghi vao `api` va log tung request (de kiem tra duoc).
    `sink(batch_dict)` (neu co) duoc goi SAU MOI BATCH de ghi checkpoint ra dia."""
    import feedparser
    for i in range(0, len(ids), a.batch):
        new = {}
        chunk = ids[i:i + a.batch]
        t1 = time.perf_counter()
        r, attempts = api_get({"id_list": ",".join(chunk), "max_results": len(chunk)}, ua)
        entries = feedparser.parse(r.text).entries
        for e in entries:
            m = re.search(r"arxiv\.org/abs/(.+?)v(\d+)$", e.id)
            if m:
                new[m.group(1)] = api[m.group(1)] = {
                    "n_versions": int(m.group(2)),
                    "v1_date": e.get("published", ""),
                    "latest_version_date": e.get("updated", ""),
                    "title": " ".join(e.get("title", "").split()),
                    "authors": [x.name for x in e.get("authors", []) if "name" in x],
                    "venue": " ".join(e["arxiv_journal_ref"].split()) if e.get("arxiv_journal_ref") else None,
                }
        if sink and new:
            sink(new)
        req_log.append({"pass": label, "ids": len(chunk), "attempts": attempts, "entries": len(entries),
                        "seconds": round(time.perf_counter() - t1, 3)})
        print(f"   [{label}] {min(i + a.batch, len(ids))}/{len(ids)} ID ({req_log[-1]['seconds']}s)", flush=True)
        if i + a.batch < len(ids):
            time.sleep(DELAY)

def load_jsonl(path):
    """Doc jsonl -> {arxiv_id: row}. Dong hong (vd dong cuoi bi cat khi crash) bi bo qua."""
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows[r["arxiv_id"]] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return rows

def write_jsonl(path, rows):
    """Ghi atomically (tmp -> replace)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def run_full(a):
    import pandas as pd
    ranges = a.ranges
    print(f"--- CHAY DISCOVERY TOAN DAI ({len(ranges)} dai) ---")
    for s_, e_ in ranges:
        print(f"   {s_} -> {e_}")
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
            sub = ch[in_ranges(ids_s, ranges)]       # loc theo TAT CA cac dai trong 1 lan quet
            for aid, vs in zip(sub["id"].astype(str), sub["versions"]):
                kaggle[aid] = {v["version"]: iso_utc(v["created"]) for v in vs} if isinstance(vs, list) else {}
    t_kaggle = time.perf_counter() - t
    ram["peak_after_kaggle_scan_mb"] = peak_ram_mb()
    kgb = os.path.getsize(a.kaggle_json) / 1e6
    print(f"   Tim thay {len(kaggle)} bai trong snapshot ({t_kaggle:.1f}s, doc {kgb / t_kaggle:.0f} MB/s, "
          f"RAM dinh {ram['peak_after_kaggle_scan_mb']} MB).", flush=True)

    # ---- tap ID can hoi API
    ids, missing_kaggle, from_snapshot = build_ids(ranges, kaggle)
    if from_snapshot:
        warnings.append(f"{len(from_snapshot)} dai di qua nhieu thang: tap ID lay tu snapshot Kaggle "
                        f"(khong sinh duoc tu 2 dau mut): {from_snapshot}")
    if len(ids) != a.total_ids:
        warnings.append(f"Co {len(ids)} ID trong cac dai nhung --total-ids = {a.total_ids}.")
    if missing_kaggle:
        warnings.append(f"{len(missing_kaggle)} ID khong co trong snapshot Kaggle (van hoi API).")

    # ---- CHECKPOINT: doc ket qua cu (neu co), bo qua ID da xong
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if a.fresh and os.path.exists(a.out):
        os.remove(a.out)
    prev = load_jsonl(a.out)
    for r in prev.values():
        if not isinstance(r.get("authors"), list) or "venue" not in r:
            sys.exit(f"{a.out} la file cu (authors khong phai list / thieu venue). "
                     "Dung --fresh hoac doi --out de chay lai tu dau.")
    resumed = [i for i in ids if i in prev]
    outside = len(prev) - len(resumed)
    if outside:
        warnings.append(f"{a.out} co {outside} ID nam ngoai cac dai hien tai: bi loai khoi file ket qua.")
    todo = [i for i in ids if i not in prev]
    # ghi lai file sach (bo dong hong/ngoai dai) truoc khi noi tiep
    write_jsonl(a.out, [prev[i] for i in resumed])
    if resumed:
        print(f"   RESUME: da co {len(resumed)}/{len(ids)} ID trong {a.out}, con {len(todo)} ID can hoi API.", flush=True)

    def make_row(aid, ai):
        vd = kaggle.get(aid, {})
        return {"arxiv_id": aid, "n_versions": ai["n_versions"], "v1_date": ai["v1_date"],
                "latest_version_date": ai["latest_version_date"], "version_dates": vd,
                "title": ai["title"], "authors": ai["authors"], "venue": ai["venue"]}

    # ---- 2) goi API (batch), ghi noi tiep vao jsonl sau MOI batch
    n_req_expected = math.ceil(len(todo) / a.batch)
    print(f"2. Goi arXiv API: {len(todo)} ID, batch {a.batch} => {n_req_expected} request...", flush=True)
    t = time.perf_counter()
    api, req_log = {}, []
    fout = open(a.out, "a", encoding="utf-8")

    def sink(batch):
        for aid, ai in batch.items():
            fout.write(json.dumps(make_row(aid, ai), ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())

    try:
        if todo:
            api_fetch(todo, a, ua, api, req_log, "lan 1", sink)
            missing = [i for i in todo if i not in api]
            if missing:
                print(f"   {len(missing)} ID chua co ket qua, hoi lai 1 lan...", flush=True)
                time.sleep(DELAY)
                api_fetch(missing, a, ua, api, req_log, "hoi lai", sink)
    except RuntimeError as ex:
        fout.close()
        sys.exit(f"\nDUNG GIUA CHUNG: {ex}\nTien do da luu trong {a.out}. Chay lai CUNG LENH de tiep tuc.")
    finally:
        fout.close()
    t_api = time.perf_counter() - t

    # ---- 3) gop + doi chieu API vs Kaggle (API la chuan) + ghi file cuoi (sap xep, khong trung)
    print(f"3. Gop va ghi ket qua cuoi ra {a.out} ...", flush=True)
    t = time.perf_counter()
    rows = load_jsonl(a.out)
    missing_api = [i for i in ids if i not in rows]
    if missing_api:
        warnings.append(f"{len(missing_api)} ID khong co tren API (bo qua, khong ghi vao jsonl): {missing_api[:10]}...")
    mismatch, v1_mismatch = [], []
    final = []
    for aid in ids:
        r = rows.get(aid)
        if not r:
            continue
        vd = r["version_dates"]
        if aid in kaggle and len(vd) != r["n_versions"]:
            mismatch.append({"id": aid, "api": r["n_versions"], "kaggle": len(vd)})
        if "v1" in vd and vd["v1"] != r["v1_date"]:
            v1_mismatch.append(aid)
        final.append(r)
    write_jsonl(a.out, final)
    written = len(final)
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
        "mode": "full", "config": cfg, "ranges": ranges, "environment": env,
        "timing_s": {"kaggle_scan": round(t_kaggle, 2), "api_total": round(t_api, 2), "write": round(t_write, 2),
                     "total_wall_end_to_end": round(t_total, 2),
                     "kaggle_download_given": dl,
                     "total_incl_download": round(t_total + dl, 2) if dl else None},
        "ram_mb": ram, "speed": speed,
        "api_requests": {"expected_batches": n_req_expected, "actual_requests": len(req_log),
                         "http_attempts": sum(r["attempts"] for r in req_log),
                         "seconds_stats": stats([r["seconds"] for r in req_log]), "log": req_log},
        "counts": {"ids_in_range": len(ids), "in_kaggle": len(kaggle), "resumed_from_checkpoint": len(resumed),
                   "fetched_this_run": len(api), "written": written,
                   "missing_in_kaggle": len(missing_kaggle), "missing_in_api": len(missing_api)},
        "check": {"n_versions_mismatch_api_vs_kaggle": mismatch, "v1_date_mismatch_api_vs_kaggle": v1_mismatch},
        "missing_in_kaggle": missing_kaggle, "missing_in_api": missing_api,
        "output": {"path": a.out, "sha256": sha256_file(a.out)}, "warnings": warnings,
    }
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    print("--- HOAN THANH ---")
    print(f"Ghi {written}/{len(ids)} bai ra {a.out} (resume: {len(resumed)}, hoi API lan nay: {len(api)})")
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

# ------------------------------------------------------------------ MODE PATCH
def oai_version_dates(sk, aid, retries=3):
    """OAI GetRecord (arXivRaw) -> {'v1': ISO8601 UTC, 'v2': ...} theo dung thu tu version.
    Loi mang/parse -> thu lai toi da `retries` lan, het luot thi nem Exception."""
    import xml.etree.ElementTree as ET
    ns = "{http://arxiv.org/OAI/arXivRaw/}"
    last = None
    for k in range(retries):
        try:
            rec = sk.GetRecord(identifier=f"oai:arXiv.org:{aid}", metadataPrefix="arXivRaw")
            root = ET.fromstring(rec.raw)
            out = {}
            for v in root.iter(f"{ns}version"):
                d = v.find(f"{ns}date")
                if d is None or not d.text:
                    raise ValueError("version khong co <date>")
                out[v.get("version")] = iso_utc(d.text.strip())
            if not out:
                raise ValueError("khong co <version>")
            return out
        except Exception as ex:
            last = ex
            time.sleep(DELAY * (k + 1))
    raise RuntimeError(f"OAI GetRecord {aid} that bai sau {retries} lan: {last}")

def run_patch(a):
    """Va version_dates tren file jsonl CO SAN (khong quet Kaggle, khong goi lai API):
      - thieu dung 1 version (version cuoi) -> v{n} = latest_version_date (API la chuan, 0 request)
      - thieu >= 2 version -> OAI GetRecord lay ngay moi version
    Ghi lai --out (atomic). Chay lai an toan: dong da du version bi bo qua."""
    if not os.path.exists(a.out):
        sys.exit(f"Khong thay file: {a.out}")
    rows = load_jsonl(a.out)
    if not rows:
        sys.exit(f"{a.out} rong.")
    print(f"--- VA version_dates: {a.out} ({len(rows)} bai) ---")
    t0 = time.perf_counter()
    sha_before = sha256_file(a.out)
    free, via_oai, anomalies = [], [], []
    for aid, r in rows.items():
        n, vd = r["n_versions"], r["version_dates"]
        if len(vd) == n:
            continue
        if len(vd) == n - 1 and list(vd) == [f"v{i}" for i in range(1, n)]:
            free.append(aid)
        elif len(vd) < n - 1:
            via_oai.append(aid)
        else:
            anomalies.append(aid)                       # vd nhieu hon n_versions / khoa bat thuong: khong tu dong va
    print(f"   Can va: {len(free)} bai (dien tu latest_version_date), {len(via_oai)} bai (OAI GetRecord); "
          f"bat thuong: {len(anomalies)}", flush=True)
    if not (free or via_oai or anomalies):
        print("   Khong co bai nao can va (file da du version_dates). Khong sua file va khong ghi bao cao.")
        return

    for aid in free:
        r = rows[aid]
        r["version_dates"][f"v{r['n_versions']}"] = r["latest_version_date"]

    failed, oai_conflict, oai_req_s = [], [], []
    if via_oai:
        from sickle import Sickle
        ua = f"DS-Project-HCMUS-arXiv-crawler/0.1 (student project; contact: {a.contact})"
        sk = Sickle(OAI_URL, headers={"User-Agent": ua}, max_retries=3)
        for k, aid in enumerate(via_oai):
            r = rows[aid]
            n = r["n_versions"]
            t1 = time.perf_counter()
            try:
                got = oai_version_dates(sk, aid)
            except RuntimeError as ex:
                print("   LOI:", ex, flush=True)
                failed.append(aid)
                got = None
            oai_req_s.append(time.perf_counter() - t1)
            if got is not None:
                if list(got) != [f"v{i}" for i in range(1, n + 1)]:
                    print(f"   {aid}: OAI tra {list(got)} khong khop n_versions={n} -> bo qua", flush=True)
                    failed.append(aid)
                else:
                    # ngay Kaggle da co phai trung OAI; neu khong trung thi ghi lai, lay OAI
                    if any(got[v] != d for v, d in r["version_dates"].items()):
                        oai_conflict.append(aid)
                    if got[f"v{n}"] != r["latest_version_date"]:
                        oai_conflict.append(aid)
                    r["version_dates"] = got
                    print(f"   [{k + 1}/{len(via_oai)}] {aid}: da va {n} version", flush=True)
            if k < len(via_oai) - 1:
                time.sleep(DELAY)

    write_jsonl(a.out, list(rows.values()))
    sha_after = sha256_file(a.out)
    still = [aid for aid, r in rows.items() if len(r["version_dates"]) != r["n_versions"]]
    last_ok = sum(1 for r in rows.values()
                  if r["version_dates"].get(f"v{r['n_versions']}") == r["latest_version_date"])
    v1_ok = sum(1 for r in rows.values() if r["version_dates"].get("v1") == r["v1_date"])
    t_total = time.perf_counter() - t0
    result = {
        "mode": "patch", "environment": {k: v for k, v in env_info(a).items() if k != "kaggle_file"},
        "input_out": a.out, "rows": len(rows),
        "patched": {"from_latest_version_date": len(free), "via_oai": len(via_oai) - len(
            [x for x in failed if x in via_oai]), "ids_from_latest_version_date": free,
            "ids_via_oai": [x for x in via_oai if x not in failed]},
        "oai_requests": {"n": len(oai_req_s), "seconds": round(sum(oai_req_s), 2),
                         "seconds_stats": stats(oai_req_s) if len(oai_req_s) > 1 else None},
        "failed": failed, "anomalies": anomalies, "oai_vs_kaggle_or_api_conflict": sorted(set(oai_conflict)),
        "check_after": {"rows_with_n_versions_dates": len(rows) - len(still), "rows_still_missing": still,
                        "last_version_date_equals_latest_version_date": last_ok,
                        "v1_equals_v1_date": v1_ok},
        "timing_s": round(t_total, 2),
        "output": {"path": a.out, "sha256_before": sha_before, "sha256_after": sha_after},
    }
    with open(a.patch_report, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("--- HOAN THANH VA ---")
    print(f"Da va: {len(free)} bai tu latest_version_date, {len(via_oai) - len(failed)} bai qua OAI "
          f"({sum(oai_req_s):.1f}s). Con thieu: {len(still)}.")
    print(f"Du version: {len(rows) - len(still)}/{len(rows)} | v_n == latest_version_date: {last_ok}/{len(rows)} "
          f"| v1 == v1_date: {v1_ok}/{len(rows)}")
    if oai_conflict:
        print("CANH BAO: OAI khac Kaggle/API o:", sorted(set(oai_conflict)))
    print(f"SHA-256 truoc: {sha_before}\nSHA-256 sau:  {sha_after}")
    print(f"Bao cao va: {a.patch_report}")
    if failed or anomalies or still:
        sys.exit("Con bai chua va duoc (xem bao cao); chay lai cung lenh de thu lai.")

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
    add("--mode", choices=["benchmark", "full", "patch"], default="benchmark", help="Che do chay")
    add("--ranges", dest="ranges_file", default=None,
        help="(mode full) file nhieu dai ID, moi dong 'ID_dau ID_cuoi'; neu co thi bo qua --id-start/--id-end")
    add("--fresh", action="store_true", help="(mode full) bo checkpoint cu trong --out, chay lai tu dau")
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
    add("--patch-report", default="discovery_patch_result.json", help="(mode patch) file bao cao va version_dates")
    add("--worker", default=None, help="(noi bo)")
    a = ap.parse_args()
    if a.ranges_file and not a.worker and a.mode != "full":
        sys.exit("--ranges chi dung cho --mode full (benchmark dung --id-start/--id-end).")
    a.ranges = load_ranges(a)           # kiem tra dinh dang, thu tu, chong lan som

    if a.worker:
        run_worker(a)
    elif a.mode == "patch":
        run_patch(a)
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