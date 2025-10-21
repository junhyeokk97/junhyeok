#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Commit .py files by strict folder blocks (folders in fixed order), while shuffling
file order inside each folder. Sundays are excluded by default.

Folder order (fixed, never interleaved):
  python -> python_import -> keras -> keras2 -> AE -> torch -> ml -> llm

NEW:
  --per-day N   Do N commits per calendar day (default 1). e.g., 5 for 4~5/day plan.

Usage (run in repo root, e.g., C:\study25):
  # Preview (Sundays excluded; 2025-05-15 ~ today; shuffle inside folders = ON; 5 per day)
  python commit_auto_block.py --dry-run --per-day 5

  # Real commit at fixed time 13:42 (times per day spread around that minute)
  python commit_auto_block.py --commit --push --per-day 5

  # Random commit time window per day (unique times inside the window)
  python commit_auto_block.py --commit --push --per-day 5 --random-time "09:00-21:30"
"""
import argparse, csv, os, subprocess, sys, math, random
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

ROTATION_DIRS = ["python", "python_import", "keras", "keras2", "AE", "torch", "ml", "llm"]
EXCLUDE_DIR_NAMES = {".git", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints"}
KST = timezone(timedelta(hours=9))

def list_py_files_sorted(root: Path, subdir: str):
    base = root / subdir
    if not base.exists():
        return []
    out = []
    for p in base.rglob("*.py"):
        if any(part in EXCLUDE_DIR_NAMES for part in p.parts):
            continue
        out.append(p)
    out.sort(key=lambda x: str(x).lower())  # deterministic before shuffle
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

def minutes_between(start_hm, end_hm):
    (h1, m1), (h2, m2) = start_hm, end_hm
    start_minutes = h1*60 + m1
    end_minutes = h2*60 + m2
    if end_minutes <= start_minutes:
        end_minutes = start_minutes + 1
    return start_minutes, end_minutes

def pick_unique_times_for_day(d: date, k: int, random_window=None, fixed_hms=(13,42,0)):
    """Return k sorted datetimes (tz=KST) within a day, unique."""
    if random_window:
        start_hm, end_hm = parse_time_window(random_window)
        s, e = minutes_between(start_hm, end_hm)
        span = max(1, e - s)
        k = min(k, span)  # cannot exceed available minutes
        mins = random.sample(range(s, e), k)
        mins.sort()
        times = [datetime(d.year, d.month, d.day, m//60, m%60, 0, tzinfo=KST) for m in mins]
    else:
        h, m, sec = fixed_hms
        # spread +0,+1,+2,... minutes to avoid identical timestamps in same day
        times = [datetime(d.year, d.month, d.day, h, (m+i)%60, sec, tzinfo=KST) for i in range(k)]
    return times

def git(*args, env=None):
    print("git", " ".join(args))
    res = subprocess.run(
        ["git", *args],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        sys.exit(res.returncode)
    return res.stdout

def assign_block(per_dir_files, dates, per_day: int):
    """
    Strict folder blocks:
      - Finish all files in 'python', then 'python_import', ..., then 'llm'.
      - Build 'slots' = each date repeated per_day times.
      - Map files sequentially to slots; leftover slots -> skip.
    """
    slots = []
    for d in dates:
        for i in range(per_day):
            slots.append(d)
    assigned = []
    di = 0  # directory index
    fi = 0  # file index inside directory
    si = 0  # slot index
    while si < len(slots) and di < len(ROTATION_DIRS):
        curr = ROTATION_DIRS[di]
        files = per_dir_files[curr]
        if fi >= len(files):
            di += 1
            fi = 0
            continue
        assigned.append((slots[si], curr, files[fi]))
        fi += 1
        si += 1
    while si < len(slots):
        assigned.append((slots[si], None, None))
        si += 1
    return assigned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-05-15")
    parser.add_argument("--end", default=None)
    parser.add_argument("--no-exclude-sunday", action="store_true")
    parser.add_argument("--random-time", dest="random_time", default=None, help='e.g. "09:00-21:30"')
    parser.add_argument("--hour", type=int, default=13)
    parser.add_argument("--minute", type=int, default=42)
    parser.add_argument("--second", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true", help="Do not shuffle inside folders (default is shuffle ON)")
    parser.add_argument("--per-day", type=int, default=1, help="Commits per day (1..10)")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--commit", action="store_true", default=False)
    parser.add_argument("--push", action="store_true", default=False)
    args = parser.parse_args()

    if args.per_day < 1 or args.per_day > 10:
        print("[!] --per-day must be 1..10")
        sys.exit(1)

    repo = Path.cwd()
    if not (repo / ".git").exists():
        print("[!] Not a git repository. Run this at repo root.")
        sys.exit(1)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else datetime.now().date()
    dates = build_dates(start, end, exclude_sunday=not args.no_exclude_sunday)
    if not dates:
        print("[!] No dates generated. Check start/end or Sunday exclusion.")
        sys.exit(1)

    per_dir_files = {d: list_py_files_sorted(repo, d) for d in ROTATION_DIRS}
    if not args.no_shuffle:
        for d in ROTATION_DIRS:
            random.shuffle(per_dir_files[d])

    print("\n[Counts per folder]")
    total = 0
    for d in ROTATION_DIRS:
        c = len(per_dir_files[d])
        total += c
        print(f"  {d:14s}: {c}")
    print(f"  Total .py: {total}")
    print(f"[Dates available] {len(dates)} (period {start} ~ {end}, Sunday excluded: {not args.no_exclude_sunday})")
    print(f"[Assign mode] block, per-day={args.per_day}")

    assigned = assign_block(per_dir_files, dates, args.per_day)

    csv_path = repo / "commit_schedule_assigned.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "weekday_ko", "folder", "file_relpath"])
        for day, folder, fpath in assigned:
            weekday_ko = ["월","화","수","목","금","토","일"][day.weekday()]
            rel = str(fpath.relative_to(repo)) if fpath else ""
            w.writerow([day.isoformat(), weekday_ko, folder or "", rel])

    print(f"\n[Saved plan] {csv_path}")

    print("\n[Preview] top 12")
    for day, folder, fpath in assigned[:12]:
        name = fpath.name if fpath else "<<skip>>"
        print(f"  {day} | {folder or '-'} | {name}")

    if not args.commit:
        print("\n[dry-run] No commits performed. Use --commit to actually commit.")
        return

    try:
        uname = git("config", "user.name").strip()
        uemail = git("config", "user.email").strip()
        if not uname or not uemail:
            print("[!] git user.name/user.email not set.")
            sys.exit(1)
        print(f"[git user] {uname} <{uemail}>")
    except SystemExit:
        raise
    except Exception:
        print("[!] Failed to read git config.")
        sys.exit(1)

    # Precompute daily times
    times_by_day = {}
    for d in dates:
        times_by_day[d] = pick_unique_times_for_day(
            d,
            k=args.per_day,
            random_window=args.random_time,
            fixed_hms=(args.hour, args.minute, args.second),
        )

    # commit loop
    per_day_counter = {d: 0 for d in dates}
    for day, folder, fpath in assigned:
        if not fpath:
            print(f"[SKIP] {day}: no file assigned")
            continue

        idx = per_day_counter[day]
        if idx >= len(times_by_day[day]):
            # safety: if slots exhausted, push to +1 minute
            base = times_by_day[day][-1]
            cdt = base + timedelta(minutes=1)
        else:
            cdt = times_by_day[day][idx]
        per_day_counter[day] += 1

        ts = cdt.isoformat()
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = ts
        env["GIT_COMMITTER_DATE"] = ts

        rel = str(fpath.relative_to(repo))
        git("add", rel, env=env)
        msg = f"[auto] {day.isoformat()} : add {folder} / {fpath.name}"
        git("commit", "-m", msg, env=env)

    if args.push:
        try:
            git("push", "-u", "origin", "newmain")
        except SystemExit:
            print("[!] push failed. Check remote/branch settings.")
            sys.exit(1)

    print("\n[Done] Commits created as per plan.")
    if not args.push:
        print("  (Optional) push: git push -u origin newmain")

if __name__ == "__main__":
    main()
