"""
P1-00 - Benchmark 3 phuong phap Discovery (thuc nghiem lap lai, do thoi gian + RAM).

Thiet ke:
  * Moi (phuong phap, lan lap) chay trong 1 TIEN TRINH RIENG => RAM dinh khong bi lan giua cac phuong phap.
  * Lap R lan (mac dinh 3), xen ke (api, oai, kaggle, api, oai, kaggle, ...) de giam sai lech do mang troi.
  * Bao cao mean, median, std, min, max; uoc luong 5.000 ID kem khoang tin cay 95% cua trung binh
    va khoang [min, max] (kich ban cuc doan).
  * Doi chieu so version giua 3 nguon tren cac ID chung; kiem tra du lieu day du (so bai tim thay).
  * Tu ghi thong tin tai lap: phien ban Python/thu vien, he dieu hanh, commit git, file Kaggle
    (kich thuoc, mtime, version/ngay snapshot do ban khai bao, SHA-256 neu bat --hash-kaggle).

Cai dat:  pip install requests feedparser sickle pandas psutil
Chay:     python discovery.py --contact "email@example.com" \\
              --kaggle-json arxiv-metadata-oai-snapshot.json --download-seconds 4980 --repeats 3 \\
              --kaggle-version "<Version N tren trang dataset>" --kaggle-snapshot-date 2026-MM-DD \\
              --hash-kaggle
Chi 1 phuong phap: them  --methods kaggle   (hoac api / oai / api,kaggle)
Ket qua:  discovery_benchmark_result.json (so lieu tho) + discovery_results.md (bang dan vao bao cao)

Luu y do:
  * Thoi gian API/OAI GOM sleep 3 s giua cac request; thoi gian rieng tung request nam trong req_seconds.
  * Thoi gian Kaggle chi gom doc + loc, KHONG gom tai file (nhap tay bang --download-seconds).
  * RAM dinh: Windows = psutil peak_wset; Linux/macOS = ru_maxrss. RAM nen = RSS sau khi import thu vien
    cua phuong phap, truoc khi lam viec. Hash file Kaggle chay SAU khi do xong de khong lam nong cache.
"""
import argparse, hashlib, json, math, os, platform, re, shlex, statistics as st, subprocess, sys, time
from datetime import datetime, timezone

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


def make_ids(prefix, start, n):
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
    pre = a.prefix + "."
    found, sample = 0, {}
    t0 = time.perf_counter()
    with pd.read_json(a.kaggle_json, lines=True, chunksize=a.chunksize, dtype=False) as reader:
        for ch in reader:
            ch = ch[ch["id"].astype(str).str.startswith(pre)]
            if ch.empty:
                continue
            num = ch["id"].str.split(".").str[1].astype(int)
            ch = ch[(num >= a.start) & (num <= a.end)]
            found += len(ch)
            for _, row in ch.iterrows():
                if row["id"] in wanted:
                    sample[row["id"]] = len(row["versions"])
    return {"seconds": time.perf_counter() - t0, "found_full_range": found,
            "found_in_sample": len(sample), "versions": sample,
            "file_gb": round(os.path.getsize(a.kaggle_json) / 1e9, 2), "baseline_mb": base}


WORKERS = {"api": worker_api, "oai": worker_oai, "kaggle": worker_kaggle}


def run_worker(a):
    n = a.oai_n if a.worker == "oai" else a.n
    ids = make_ids(a.prefix, a.start, n)
    res = WORKERS[a.worker](a, ids)
    res["peak_mb"] = peak_ram_mb()
    print("RESULT " + json.dumps(res))


# ------------------------------------------------------------------ thong ke
def fmt(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def tcrit(df):
    return T95.get(df, 1.96)


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    n = len(v)
    sd = st.stdev(v) if n > 1 else 0.0
    ci = tcrit(n - 1) * sd / math.sqrt(n) if n > 1 else None   # nua do rong KTC 95% cua trung binh
    return {"mean": st.mean(v), "median": st.median(v), "std": sd, "min": min(v), "max": max(v),
            "n": n, "ci95": ci}


def est_block(total_req, q, extra=None):
    """Uoc luong tong thoi gian = so request * (thoi gian 1 request + sleep)."""
    d = {"requests": total_req, "n_samples": q["n"],
         "mean": total_req * (q["mean"] + DELAY),
         "low": total_req * (q["min"] + DELAY), "high": total_req * (q["max"] + DELAY)}
    if q["ci95"] is not None:
        d["ci_low"] = total_req * (max(q["mean"] - q["ci95"], 0.0) + DELAY)
        d["ci_high"] = total_req * (q["mean"] + q["ci95"] + DELAY)
    if extra:
        d.update(extra)
    return d


# ------------------------------------------------------------------ thong tin tai lap
def lib_versions():
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for n in LIBS:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)), timeout=5)
        return (r.stdout.strip() or None) if r.returncode == 0 else None
    except Exception:
        return None


def sha256_file(path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
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
        else:
            out.append(tok)
    return out


def env_info(a):
    info = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "libs": lib_versions(), "git_commit": git_commit(),
            "argv": masked_argv(), "delay_s": DELAY}
    if a.kaggle_json and os.path.exists(a.kaggle_json):
        fs = os.stat(a.kaggle_json)
        info["kaggle_file"] = {
            "name": os.path.basename(a.kaggle_json), "size_bytes": fs.st_size,
            "mtime_utc": datetime.fromtimestamp(fs.st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_version": a.kaggle_version, "snapshot_date": a.kaggle_snapshot_date, "sha256": None}
    return info


# ------------------------------------------------------------------ dieu phoi
def orchestrate(a):
    methods = [m.strip() for m in a.methods.split(",")]
    passthrough = ["--prefix", a.prefix, "--start", str(a.start), "--n", str(a.n), "--oai-n", str(a.oai_n),
                   "--end", str(a.end), "--batch", str(a.batch), "--contact", a.contact,
                   "--chunksize", str(a.chunksize)]
    if a.kaggle_json:
        passthrough += ["--kaggle-json", a.kaggle_json]
    env = env_info(a)
    warnings = []
    exp_range = a.end - a.start + 1
    if exp_range != a.total_ids:
        warnings.append(f"Dai ID [{a.start}, {a.end}] co {exp_range} ID nhung --total-ids = {a.total_ids}.")
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

    # hash chay SAU khi do de khong lam nong cache cua lan doc dau
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

    # --- uoc luong cho total_ids
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

    # --- doi chieu so version giua cac nguon (tren ID chung)
    votes = {}
    for m, rs in runs.items():
        for x in rs:
            for i, v in x["versions"].items():
                votes.setdefault(i, {}).setdefault(m, set()).add(v)
    common = {i: d for i, d in votes.items() if len(d) >= 2}
    mismatch = [i for i, d in common.items() if len({v for s in d.values() for v in s}) > 1]
    unstable = [i for i, d in votes.items() if any(len(s) > 1 for s in d.values())]  # cung nguon, khac lan lap

    # --- ghi file
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

    L.append("\n### Kiem tra quy tac quyet dinh\n")
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
    ap.add_argument("--prefix", default="2506")
    ap.add_argument("--start", type=int, default=788)
    ap.add_argument("--end", type=int, default=5787, help="can tren dai ID (cho Kaggle)")
    ap.add_argument("--n", type=int, default=50, help="so ID mau cho API (1 request) va de doi chieu")
    ap.add_argument("--oai-n", type=int, default=20, help="so ID mau cho OAI GetRecord (moi lan lap)")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--total-ids", type=int, default=5000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--methods", default="api,oai,kaggle")
    ap.add_argument("--contact", required=True)
    ap.add_argument("--kaggle-json", default=None)
    ap.add_argument("--kaggle-version", default=None, help="phien ban dataset tren Kaggle (ghi vao ket qua)")
    ap.add_argument("--kaggle-snapshot-date", default=None, help="ngay cap nhat cua phien ban do (YYYY-MM-DD)")
    ap.add_argument("--hash-kaggle", action="store_true", help="tinh SHA-256 file Kaggle sau khi do xong")
    ap.add_argument("--download-seconds", type=float, default=None)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--ram-limit-mb", type=float, default=6000)
    ap.add_argument("--worker", default=None, help="(noi bo) chay 1 phuong phap trong tien trinh rieng")
    a = ap.parse_args()
    if a.worker:
        run_worker(a)
    else:
        if "kaggle" in a.methods and not a.kaggle_json:
            sys.exit("Thieu --kaggle-json (hoac bo 'kaggle' khoi --methods).")
        if a.kaggle_json and not os.path.exists(a.kaggle_json):
            sys.exit(f"Khong thay file: {a.kaggle_json}")
        if _psutil() is None and sys.platform.startswith("win"):
            sys.exit("Windows can psutil de do RAM dinh: pip install psutil")
        orchestrate(a)