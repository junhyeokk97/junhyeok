#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commit .py files to GitHub by folder rotation and date plan (Sundays excluded).

Folders (rotation): python -> python_import -> keras -> keras2 -> AE -> torch -> ml -> llm
Period: default 2025-05-15 ~ today (local), Sundays excluded
Uses existing .py files without modifying them
Backdates commits using GIT_AUTHOR_DATE/GIT_COMMITTER_DATE
Creates commit_schedule_assigned.csv for transparency

Usage (CMD from repo root, e.g., C:\study25):
    python commit_auto.py --dry-run
    python commit_auto.py --commit --random-time "09:00-21:30" --push
"""
import argparse, csv, os, subprocess, sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

ROTATION_DIRS = ["python", "python_import", "keras", "keras2", "AE", "torch", "ml", "llm"]
EXCLUDE_DIR_NAMES = {".git", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints"}
KST = timezone(timedelta(hours=9))

def list_py_files(root: Path, subdir: str):
    base = root / subdir
    if not base.exists():
        return []
    out = []
    for p in base.rglob("*.py"):
        if any(part in EXCLUDE_DIR_NAMES for part in p.parts):
            continue
        out.append(p)
    out.sort(key=lambda x: (len(str(x)), str(x).lower()))
    return out

def build_dates(start: date, end: date, exclude_sunday: bool = True):
    d = start
    out = []
    while d <= end:
        if not (exclude_sunday and d.weekday() == 6):
            out.append(d)
        d += timedelta(days=1)
    return out

def parse_time_window(s: str):
    a, b = s.split("-")
    h1, m1 = map(int, a.split(":"))
    h2, m2 = map(int, b.split(":"))
    return (h1, m1), (h2, m2)

def pick_random_time(d: date, start_hm, end_hm):
    (h1, m1), (h2, m2) = start_hm, end_hm
    start_minutes = h1*60 + m1
    end_minutes = h2*60 + m2
    if end_minutes <= start_minutes:
        end_minutes = start_minutes + 1
    import random
    tmin = __import__("random").randint(start_minutes, end_minutes-1)
    hh, mm = divmod(tmin, 60)
    return datetime(d.year, d.month, d.day, hh, mm, 0, tzinfo=KST)

def git(*args, env=None):
    print("git", " ".join(args))
    res = subprocess.run(["git", *args], env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        sys.exit(res.returncode)
    return res.stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-05-15")
    parser.add_argument("--end", default=None)
    parser.add_argument("--no-exclude-sunday", action="store_true")
    parser.add_argument("--random-time", dest="random_time", default=None, help='e.g. "09:00-21:30"')
    parser.add_argument("--hour", type=int, default=13)
    parser.add_argument("--minute", type=int, default=42)
    parser.add_argument("--second", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--commit", action="store_true", default=False)
    parser.add_argument("--push", action="store_true", default=False)
    args = parser.parse_args()

    repo = Path.cwd()
    if not (repo / ".git").exists():
        print("[!] Not a git repository (missing .git). Run this script at repo root.")
        sys.exit(1)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end = datetime.now().date()

    dates = build_dates(start, end, exclude_sunday=not args.no_exclude_sunday)
    if not dates:
        print("[!] No dates generated. Check start/end or Sunday exclusion.")
        sys.exit(1)

    per_dir_files = {d: list_py_files(repo, d) for d in ROTATION_DIRS}
    per_dir_counts = {d: len(per_dir_files[d]) for d in ROTATION_DIRS}

    total_files = sum(per_dir_counts.values())
    print("\n[File counts]")
    for d in ROTATION_DIRS:
        print(f"  {d:14s}: {per_dir_counts[d]}")
    print(f"  Total: {total_files}")
    print(f"[Dates] {len(dates)} (period: {start} ~ {end}, exclude Sunday: {not args.no_exclude_sunday})")

    assigned = []
    file_indices = {d: 0 for d in ROTATION_DIRS}
    for i, day in enumerate(dates, 1):
        target_dir = ROTATION_DIRS[(i-1) % len(ROTATION_DIRS)]
        files = per_dir_files[target_dir]
        if not files:
            assigned.append((day, target_dir, None))
            continue
        idx = file_indices[target_dir] % len(files)
        fpath = files[idx]
        file_indices[target_dir] += 1
        assigned.append((day, target_dir, fpath))

    csv_path = repo / "commit_schedule_assigned.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = __import__("csv").writer(f)
        w.writerow(["date", "weekday_ko", "target_dir", "file_relpath"])
        for day, tdir, fpath in assigned:
            weekday_ko = ["월","화","수","목","금","토","일"][day.weekday()]
            rel = str(fpath.relative_to(repo)) if fpath else ""
            w.writerow([day.isoformat(), weekday_ko, tdir, rel])

    print(f"\n[Saved plan] {csv_path}")

    show_n = min(10, len(assigned))
    print(f"\n[Preview] top {show_n}:")
    for row in assigned[:show_n]:
        day, tdir, fpath = row
        print(f"  {day} ({tdir}) -> {fpath.name if fpath else '<<no .py in folder>>'}")

    if not args.commit:
        print("\n[dry-run] No commits performed. Use --commit to actually commit.")
        return

    try:
        uname = git("config", "user.name").strip()
        uemail = git("config", "user.email").strip()
        if not uname or not uemail:
            print("[!] git user.name/user.email are not set. Configure and retry.")
            sys.exit(1)
        print(f"[git user] {uname} <{uemail}>")
    except SystemExit:
        raise
    except Exception:
        print("[!] Failed to read git config. Is git installed?")
        sys.exit(1)

    for day, tdir, fpath in assigned:
        if fpath is None:
            print(f"[SKIP] {day} {tdir}: no .py files")
            continue

        if args.random_time:
            start_hm, end_hm = parse_time_window(args.random_time)
            cdt = pick_random_time(day, start_hm, end_hm)
        else:
            cdt = datetime(day.year, day.month, day.day, args.hour, args.minute, args.second, tzinfo=KST)
        ts = cdt.isoformat()

        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = ts
        env["GIT_COMMITTER_DATE"] = ts

        rel = str(fpath.relative_to(repo))
        git("add", rel, env=env)
        msg = f"[auto] {day.isoformat()} : add {tdir} / {fpath.name}"
        git("commit", "-m", msg, env=env)

    if args.push:
        try:
            git("push", "-u", "origin", "main")
        except SystemExit:
            print("[!] push failed. Check remote/branch settings.")
            sys.exit(1)

    print("\n[Done] Commits created as per schedule.")
    if not args.push:
        print("  (Optional) push:  git push -u origin main")

if __name__ == "__main__":
    main()
