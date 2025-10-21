import os, sys, json, subprocess, shutil, random
from datetime import datetime, date, time, timedelta

# ---------- 유틸 ----------
def daterange(start: date, end: date, exclude_weekdays=None):
    cur = start
    while cur <= end:
        if not exclude_weekdays or cur.weekday() not in exclude_weekdays:
            yield cur
        cur += timedelta(days=1)

def run(cmd, cwd, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    # UTF-8로 강제 디코드, 오류는 무시
    res = subprocess.run(
        cmd, cwd=cwd,
        capture_output=True, text=True,
        encoding="utf-8", errors="ignore",
        env=env
    )
    if res.returncode != 0:
        err = (res.stderr or "").strip()
        raise RuntimeError(f"[git] error: {' '.join(cmd)}\n{err}")
    return (res.stdout or "").strip()

def gather_files(folder_abs):
    """폴더에서 .py 파일만 수집, 캐시/숨김/가상환경/노트북 체크포인트 등 제외"""
    files = []
    EXCLUDE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", "env"}
    for root, dirs, fns in os.walk(folder_abs):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            if fn.startswith("."):
                continue
            files.append(os.path.join(root, fn))
    files.sort()
    return files

def parse_hms(s: str) -> time:
    return datetime.strptime(s, "%H:%M:%S").time()

def random_time_between(t_start: time, t_end: time) -> time:
    start_sec = t_start.hour*3600 + t_start.minute*60 + t_start.second
    end_sec = t_end.hour*3600 + t_end.minute*60 + t_end.second
    if end_sec <= start_sec:
        end_sec = start_sec + 3600
    r = random.randint(start_sec, end_sec - 1)
    return time(hour=r//3600, minute=(r%3600)//60, second=r%60)

# ---------- 분배 로직 ----------
def assign_range_even(files, start_date, end_date, exclude_weekdays=None):
    days = list(daterange(start_date, end_date, exclude_weekdays))
    if not days:
        return []
    plan, idx = [], 0
    for f in files:
        plan.append((f, days[idx % len(days)]))
        idx += 1
    return plan

def plan_for_schedule(schedule, repo_path, exclude_weekdays):
    folder_abs = os.path.join(repo_path, schedule["folder"])
    if not os.path.isdir(folder_abs):
        raise FileNotFoundError(f"폴더 없음: {folder_abs}")

    files = gather_files(folder_abs)
    if schedule.get("take_first_k_files"):
        files = files[: int(schedule["take_first_k_files"])]

    start_date = datetime.strptime(schedule["start_date"], "%Y-%m-%d").date()
    end_date   = datetime.strptime(schedule["end_date"],   "%Y-%m-%d").date()

    mode = schedule.get("mode", "range_even")
    if mode != "range_even":
        raise ValueError("현재 스크립트는 range_even 모드만 지원합니다.")

    return assign_range_even(files, start_date, end_date, exclude_weekdays)

# ---------- 커밋 ----------
def commit_one(repo_path, file_abs, commit_dt: date, commit_time: time):
    rel = os.path.relpath(file_abs, repo_path)

    # Git이 UTF-8로 출력/로그 처리하도록 환경변수 보강
    extra_env = {
        "GIT_COMMITTER_DATE": f"{commit_dt.isoformat()}T{commit_time.strftime('%H:%M:%S')}+09:00",
        "GIT_AUTHOR_DATE":    f"{commit_dt.isoformat()}T{commit_time.strftime('%H:%M:%S')}+09:00",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "GIT_PAGER": "cat",
        "PYTHONIOENCODING": "utf-8",
    }

    run(["git", "add", rel], cwd=repo_path, extra_env=extra_env)

    commit_time_str = extra_env["GIT_AUTHOR_DATE"]
    msg = f"Add {rel} ({commit_dt})"

    res = subprocess.run(
        ["git", "commit", "-m", msg, f"--date={commit_time_str}"],
        cwd=repo_path,
        capture_output=True, text=True,
        encoding="utf-8", errors="ignore",
        env={**os.environ, **extra_env}
    )
    if res.returncode != 0:
        # 이미 커밋할 변경이 없거나 기타 사유
        return False, (res.stderr or "").strip()
    return True, (res.stdout or "").strip()

# ---------- 메인 ----------
def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    repo_path  = cfg["repo_path"]
    dry_run    = bool(cfg.get("dry_run", True))
    push_after = bool(cfg.get("push_after", False))
    remote     = cfg.get("remote", "origin")
    branch     = cfg.get("branch", "main")

    rt_cfg = cfg.get("random_time", {"enabled": True, "start": "08:30:00", "end": "22:30:00"})
    rt_enabled = bool(rt_cfg.get("enabled", True))
    rt_start = parse_hms(rt_cfg.get("start", "08:30:00"))
    rt_end   = parse_hms(rt_cfg.get("end",   "22:30:00"))

    # 일요일 제외 (weekday: 월=0 … 일=6)
    exclude_weekdays = cfg.get("exclude_weekdays_py", [6])

    if not shutil.which("git"):
        print("git 이 설치되어 있지 않습니다. 먼저 Git을 설치하세요.")
        sys.exit(1)

    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"{repo_path} 는 git 저장소가 아닙니다. 먼저 `git init`과 원격 연결을 완료하세요.")
        sys.exit(1)

    print("=== 계획 생성 ===")
    grand_plan = []
    for sch in cfg["schedules"]:
        plan = plan_for_schedule(sch, repo_path, exclude_weekdays)
        fixed_time = sch.get("time_of_day")  # 없으면 랜덤
        print(f"- {sch['name']} | {len(plan)} files | days(excl Sun): yes")
        for f, d in plan:
            grand_plan.append((f, d, fixed_time))
    grand_plan.sort(key=lambda x: x[1])

    print("\n=== 커밋 실행 ===")
    for file_abs, dt, fixed_time in grand_plan:
        rel = os.path.relpath(file_abs, repo_path)
        if fixed_time and not rt_enabled:
            ctime = parse_hms(fixed_time)
        else:
            ctime = random_time_between(rt_start, rt_end)

        if dry_run:
            print(f"[DRY RUN] {dt} {ctime.strftime('%H:%M:%S')} -> {rel}")
            continue

        ok, msg = commit_one(repo_path, file_abs, dt, ctime)
        # msg가 None 일 경우 방지
        safe_msg = msg if isinstance(msg, str) and len(msg) else ("OK" if ok else "SKIP")
        print(f"[{dt} {ctime.strftime('%H:%M:%S')}] {rel} -> {'OK' if ok else safe_msg}")

    if not dry_run and push_after:
        print("\n푸시 중...")
        try:
            out = run(["git", "push", remote, branch], cwd=repo_path, extra_env={"LC_ALL":"C.UTF-8","LANG":"C.UTF-8"})
            print(out)
        except Exception as e:
            print("푸시 실패:", e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python distribute_commits.py commit_schedule.json")
        sys.exit(1)
    main(sys.argv[1])
