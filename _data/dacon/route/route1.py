# -*- coding: utf-8 -*-
"""
AGV 경로 최적화 (CVRPTW with Soft Deadlines) - OR-Tools
- 거리: 맨해튼(기본) / 장애물 파일 있으면 격자 BFS 최단거리
- 이동시간: 거리 / AGV 속도(차량별 상이)
- 서비스시간: task.service_time (도착지에서 수행)
- 지각 패널티: lambda * max(0, completion - deadline)
- 제약: capacity, max_distance (왕복 단위로 적용)
- 규칙 반영: DEPOT 복귀 시 capacity/max_distance 초기화, 빈 왕복 금지
- 제출형식: agv_id, route  (예: DEPOT,T0001,DEPOT,T0005,DEPOT)
출력: ./_data/dacon/route/submission.csv
"""

import argparse, os, math, json, hashlib, re, csv
from datetime import datetime, timezone, timedelta
from collections import deque
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# (선택) Δ예측/포인터 초기해 — trips=1일 때만 사용
import joblib
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None  # torch 미설치 환경에서도 안전하게 동작

# ------------------------------
# 경로 & 상수
# ------------------------------
BASE = "./_data/dacon/route/"
AGV_PATH  = BASE + "agv.csv"
TASK_PATH = BASE + "task.csv"
OUT_PATH  = BASE + "submission.csv"
INF = 10**9

# ------------------------------
# 데이터 적재/전처리
# ------------------------------


def load_data(agv_path, task_path):
    agv = pd.read_csv(agv_path)
    task = pd.read_csv(task_path)
    depot = pd.DataFrame([{"task_id":"DEPOT","x":0,"y":0,"service_time":0,"demand":0,"deadline":10**12}])
    task = pd.concat([depot, task], ignore_index=True)   # node 0 = DEPOT
    return agv, task

def load_obstacles(path: str):
    """CSV: x,y 열로 '막힌 셀' 좌표 나열. 없으면 빈 집합."""
    if not path or not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return {(int(x), int(y)) for x, y in zip(df["x"], df["y"])}

def _bbox(points, margin=20):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    margin = max(margin, (abs(maxx-minx)+abs(maxy-miny))//2 + 5)
    return (minx - margin, maxx + margin, miny - margin, maxy + margin)

def _bfs_row(src_xy, targets_set, blocked, bbox):
    xmin, xmax, ymin, ymax = bbox
    q = deque([src_xy])
    dist = {src_xy: 0}
    remain = set(targets_set) - {src_xy}
    out = {}
    while q and remain:
        x, y = q.popleft()
        d = dist[(x, y)]
        if (x, y) in remain:
            out[(x, y)] = d
            remain.remove((x, y))
            if not remain:
                break
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if nx < xmin or nx > xmax or ny < ymin or ny > ymax: continue
            if (nx, ny) in blocked: continue
            if (nx, ny) in dist:    continue
            dist[(nx, ny)] = d + 1
            q.append((nx, ny))
    return out  # {(x,y): steps}

def _dist_cache_key(points, blocked):
    h = hashlib.sha1()
    h.update(json.dumps(sorted(points)).encode())
    h.update(json.dumps(sorted(blocked)).encode())
    return h.hexdigest()[:12]

def _try_load_cache(base_dir, key):
    path = os.path.join(base_dir, f"dist_{key}.npy")
    return (np.load(path) if os.path.exists(path) else None), path

def _save_cache(dist, path):
    np.save(path, dist)

def build_mats(task_df: pd.DataFrame, obstacles_path: str = "", use_cache: bool = True, cache_dir: str = BASE):
    """
    장애물 파일이 있으면 '정확한 격자 최단거리(BFS)' 행렬을 만들고,
    없으면 기존 맨해튼 거리를 사용.
    """
    coords = task_df[["x","y"]].astype(int).to_numpy()
    points = [tuple(p) for p in coords]  # DEPOT 포함
    blocked = load_obstacles(obstacles_path)

    if blocked:
        key = _dist_cache_key(points, blocked) if use_cache else None
        dist_mat, cache_path = (None, None)
        if use_cache:
            dist_mat, cache_path = _try_load_cache(cache_dir, key)
        if dist_mat is None:
            N = len(points)
            dist_mat = np.full((N, N), INF, dtype=np.int32)
            bbox = _bbox(points, margin=20)
            targets = set(points)
            def _job(si):
                src = points[si]
                found = _bfs_row(src, targets, blocked, bbox)
                row = np.full(N, INF, dtype=np.int32)
                for j, pj in enumerate(points):
                    if pj == src: row[j] = 0
                    elif pj in found: row[j] = found[pj]
                return si, row
            with Pool(processes=min(cpu_count(), 8)) as pool:
                for si, row in pool.imap_unordered(_job, range(N), chunksize=1):
                    dist_mat[si, :] = row
            if use_cache and cache_path:
                _save_cache(dist_mat, cache_path)
        dist = dist_mat
    else:
        n = len(points)
        dist = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            xi, yi = points[i]
            for j in range(n):
                xj, yj = points[j]
                dist[i, j] = abs(xi - xj) + abs(yi - yj)

    service  = task_df["service_time"].to_numpy(dtype=np.int64)
    demand   = task_df["demand"].to_numpy(dtype=np.int64)
    deadline = task_df["deadline"].to_numpy(dtype=np.int64)
    return dist, service, demand, deadline

def analyze_obstacle_connectivity(task_df: pd.DataFrame, dist: np.ndarray, INF: int = INF):
    """
    장애물 거리(dist) 기준 연결성 진단:
      - unreachable: DEPOT(0)과 단절된 작업 리스트
      - comp_cnt: 작업-작업(DEPOT 제외) 무방향 그래프의 연결 성분 개수
    """
    N = len(task_df)  # 0번 = DEPOT
    unreachable = []

    # DEPOT ↔ 작업 단절 검사
    for j in range(1, N):
        if dist[0, j] >= INF or dist[j, 0] >= INF:
            unreachable.append(task_df["task_id"].iloc[j])

    # 작업-작업 그래프(DEPOT 제외), 간선: 어느 한쪽이라도 유한거리면 연결로 간주
    adj = [[] for _ in range(N - 1)]
    for i in range(1, N):
        for j in range(i + 1, N):
            if dist[i, j] < INF or dist[j, i] < INF:
                a, b = i - 1, j - 1  # 0-index shift (DEPOT 제외)
                adj[a].append(b)
                adj[b].append(a)

    # 연결 성분 개수
    seen = [False] * (N - 1)
    comp_cnt = 0
    for s in range(N - 1):
        if not seen[s]:
            comp_cnt += 1
            stack = [s]
            seen[s] = True
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)

    return unreachable, comp_cnt

# ------------------------------
# 스코어 & 유틸
# ------------------------------
def local_score(subm, agv, task, lam=1.0):
    lookup = task.set_index("task_id").to_dict(orient="index")
    def xy(tok): return (0,0) if tok=="DEPOT" else (int(lookup[tok]["x"]), int(lookup[tok]["y"]))
    def svc(tok): return 0 if tok=="DEPOT" else int(lookup[tok]["service_time"])
    def ddl(tok): return 10**12 if tok=="DEPOT" else int(lookup[tok]["deadline"])
    def mh(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

    score = 0.0
    for _, r in subm.iterrows():
        agv_id = r["agv_id"]
        speed  = float(agv.loc[agv["agv_id"]==agv_id,"speed_cells_per_sec"].values[0])
        toks   = [t for t in r["route"].split(",") if t]
        t = 0.0
        for i in range(len(toks)-1):
            a, b = toks[i], toks[i+1]
            t += mh(xy(a), xy(b)) / max(speed,1e-6) + svc(b)
            if b != "DEPOT":
                score += lam * max(0.0, t - ddl(b))
        score += t
    return score

def score_one_route(route_tokens, speed, task_df, lam):
    lookup = task_df.set_index("task_id").to_dict(orient="index")
    def xy(t): return (0,0) if t=="DEPOT" else (int(lookup[t]["x"]), int(lookup[t]["y"]))
    def svc(t): return 0 if t=="DEPOT" else int(lookup[t]["service_time"])
    def ddl(t): return 10**12 if t=="DEPOT" else int(lookup[t]["deadline"])
    def mh(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    t = 0.0; cost = 0.0
    for i in range(len(route_tokens)-1):
        a,b = route_tokens[i], route_tokens[i+1]
        t += mh(xy(a), xy(b)) / max(speed,1e-6) + svc(b)
        if b != "DEPOT":
            cost += lam * max(0.0, t - ddl(b))
    cost += t
    return cost

def squash_and_clean_route_str(s: str) -> str:
    s = re.sub(r'\s+', '', str(s))
    s = re.sub(r',+', ',', s.strip(','))
    toks = [t for t in s.split(',') if t]
    if not toks: toks = ["DEPOT","DEPOT"]
    if toks[0] != "DEPOT": toks = ["DEPOT"] + toks
    if toks[-1] != "DEPOT": toks = toks + ["DEPOT"]
    out = []
    for t in toks:
        if t == "DEPOT" and out and out[-1] == "DEPOT":
            continue
        out.append(t)
    return ",".join(out)

def normalize_and_validate_submission(subm: pd.DataFrame, agv_df: pd.DataFrame, task_df: pd.DataFrame) -> pd.DataFrame:
    """
    - 컬럼: agv_id, route
    - 모든 agv_id row 존재/순서 일치
    - route는 'DEPOT,...,DEPOT' 형태, 내부 빈 왕복 금지
    - 모든 task가 전체 라우트에서 정확히 1회
    """
    subm = subm.copy()
    subm.columns = [c.strip().lower() for c in subm.columns]
    if set(subm.columns) != {"agv_id","route"}:
        if "agv_id" not in subm.columns or "route" not in subm.columns:
            raise ValueError(f"[Format] columns must be exactly ['agv_id','route'], got {list(subm.columns)}")
    subm = subm[["agv_id","route"]]
    agv_ids = list(map(str, agv_df["agv_id"].tolist()))
    subm["agv_id"] = subm["agv_id"].astype(str)
    subm = subm.set_index("agv_id").reindex(agv_ids)
    if subm["route"].isna().any():
        miss = subm[subm["route"].isna()].index.tolist()
        raise ValueError(f"[Format] missing rows for agv_id(s): {miss}")
    subm = subm.reset_index().rename(columns={"index":"agv_id"})

    valid_tasks = set(task_df["task_id"].astype(str).tolist())
    valid_tokens = valid_tasks | {"DEPOT"}
    all_seen = []

    def clean_route(s: str) -> str:
        toks = [t.strip() for t in str(s).split(",") if t.strip()]
        if not toks: toks = ["DEPOT","DEPOT"]
        if toks[0] != "DEPOT": toks = ["DEPOT"] + toks
        if toks[-1] != "DEPOT": toks = toks + ["DEPOT"]
        comp = []
        for t in toks:
            if t == "DEPOT" and comp and comp[-1] == "DEPOT":
                continue
            comp.append(t)
        toks = comp
        if len(toks) == 2 and toks[0] == "DEPOT" and toks[1] == "DEPOT":
            return "DEPOT,DEPOT"
        for t in toks:
            if t not in valid_tokens:
                raise ValueError(f"[Route] invalid token '{t}' in: {s}")
        for a, b in zip(toks, toks[1:]):
            if a == "DEPOT" and b == "DEPOT":
                raise ValueError(f"[Route] internal empty trip after compress: {','.join(toks)}")
        all_seen.extend([t for t in toks if t != "DEPOT"])
        return ",".join(toks)

    subm["route"] = subm["route"].map(clean_route)
    tasks_only = [t for t in valid_tasks if t != "DEPOT"]
    seen_cnt = pd.Series(all_seen).value_counts()
    miss = [t for t in tasks_only if t not in seen_cnt.index]
    dup  = [t for t, c in seen_cnt.items() if c > 1]
    if miss:
        raise ValueError(f"[Coverage] missing task(s): {miss[:10]}{' ...' if len(miss)>10 else ''}")
    if dup:
        raise ValueError(f"[Coverage] duplicated task(s): {dup[:10]}{' ...' if len(dup)>10 else ''}")
    subm = subm[["agv_id","route"]].copy()
    subm["agv_id"] = subm["agv_id"].astype(str)
    subm["route"] = subm["route"].astype(str)
    return subm

# === [새 유틸] 트립 분해/검증 ===
def _split_into_trips(tokens):
    """['DEPOT', T.., ..., 'DEPOT', T.., 'DEPOT'] -> [['DEPOT',..., 'DEPOT'], ...] (빈 트립 없음 가정)"""
    trips, cur = [], []
    for t in tokens:
        cur.append(t)
        if t == "DEPOT" and len(cur) > 1:
            trips.append(cur)
            cur = ["DEPOT"]
    return trips

def _feasible_trip(seg, task_idx, cap, maxd):
    """단일 트립(seg: DEPOT~DEPOT)에 대해 수요합/맨해튼 거리합 제약 검사"""
    assert seg[0] == "DEPOT" and seg[-1] == "DEPOT" and len(seg) >= 2
    # 용량: 고객 수요 합
    customers = [t for t in seg if t != "DEPOT"]
    demand_sum = int(task_idx.loc[customers, "demand"].sum()) if customers else 0
    if demand_sum > cap:
        return False
    # 거리: 트립 내 맨해튼 합
    def _xy(tok):
        if tok == "DEPOT": return (0,0)
        d = task_idx.loc[tok]; return int(d["x"]), int(d["y"])
    def _mh(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    dist_sum = sum(_mh(_xy(a), _xy(b)) for a,b in zip(seg, seg[1:]))
    return dist_sum <= maxd

def _feasible_agv_tokens(tokens, aid, task_idx, cap_of, maxd_of):
    """AGV 한 대의 전체 토큰이 모든 트립에서 cap/maxd를 만족하는지"""
    cap  = int(cap_of[aid]); maxd = int(maxd_of[aid])
    trips = _split_into_trips(tokens)
    if not trips:  # 'DEPOT,DEPOT'만 있는 경우 OK
        return True
    return all(_feasible_trip(seg, task_idx, cap, maxd) for seg in trips)

# === [교체] 제약 인지형 LNS ===
def post_lns(subm, agv_df, task_df, lam=1.0, iters=40, ruin_rate=0.15, per_trip_only=False):
    """
    제약 인지형 LNS:
      - 제거/재삽입할 때마다 해당 AGV의 모든 트립이 cap/maxd를 만족하는지 검사.
      - 실패하면 해당 AGV는 롤백.
      - per_trip_only=True면 각 트립 내부에서만 재배열(트립 간 이동 금지; 더 안전).
    """
    import random
    task_idx = task_df.set_index("task_id")
    cap_of   = dict(zip(agv_df["agv_id"], agv_df["capacity"]))
    maxd_of  = dict(zip(agv_df["agv_id"], agv_df["max_distance"]))
    speed_of = dict(zip(agv_df["agv_id"], agv_df["speed_cells_per_sec"]))

    def toks(s): return [t for t in s.split(",") if t]
    def score(tokens, v): return score_one_route(tokens, v, task_df, lam)

    best = subm.copy()

    for _ in range(iters):
        cand = best.copy()
        improved = False

        for i, r in cand.iterrows():
            aid = r["agv_id"]; v = float(speed_of[aid])
            tokens = toks(r["route"])
            cust   = [t for t in tokens if t != "DEPOT"]
            if len(cust) < 2:
                continue

            original = tokens[:]  # 롤백용

            if per_trip_only:
                # --- 트립별 독립 LNS (안전모드) ---
                trips = _split_into_trips(tokens)
                new_trips = []
                ok_agv = True
                for seg in trips:
                    inner = [t for t in seg if t != "DEPOT"]
                    if len(inner) < 2:
                        new_trips.append(seg); continue
                    k = max(1, int(len(inner) * ruin_rate))
                    remove = set(random.sample(inner, k))
                    remain = [t for t in seg if (t == "DEPOT" or t not in remove)]
                    # 삽입 후보(트립 내부)
                    for rem in sorted(remove, key=lambda t:int(task_idx.loc[t, "deadline"])):
                        best_pos, best_cost, best_seg = None, float("inf"), None
                        for pos in range(1, len(remain)):  # DEPOT 사이 어디든
                            trial = remain[:pos] + [rem] + remain[pos:]
                            if not _feasible_trip(trial, task_idx, int(cap_of[aid]), int(maxd_of[aid])):
                                continue
                            c = score(trial, v)
                            if c < best_cost:
                                best_cost, best_pos, best_seg = c, pos, trial
                        if best_seg is None:
                            ok_agv = False; break
                        remain = best_seg
                    if not ok_agv: break
                    new_trips.append(remain)
                if not ok_agv:
                    cand.at[i, "route"] = ",".join(original); continue
                # 트립 연결(DEPOT 중복 제거)
                merged = []
                for seg in new_trips:
                    if not merged: merged.extend(seg)
                    else:
                        if merged and merged[-1] == "DEPOT" and seg[0] == "DEPOT":
                            merged.extend(seg[1:])
                        else:
                            merged.extend(seg)
                if score(merged, v) + 1e-9 < score(original, v):
                    # 최종 제약 한 번 더 확인
                    if _feasible_agv_tokens(merged, aid, task_idx, cap_of, maxd_of):
                        cand.at[i, "route"] = ",".join(merged); improved = True
                    else:
                        cand.at[i, "route"] = ",".join(original)
                else:
                    cand.at[i, "route"] = ",".join(original)

            else:
                # --- 일반 LNS (트립 간 이동 허용하지만 제약검사 필수) ---
                k = max(1, int(len(cust) * ruin_rate))
                remove = set(random.sample(cust, k))
                remain = [t for t in tokens if (t == "DEPOT" or t not in remove)]
                for rem in sorted(remove, key=lambda t:int(task_idx.loc[t, "deadline"])):
                    best_pos, best_cost, best_seq = None, float("inf"), None
                    for pos in range(1, len(remain)):
                        trial = remain[:pos] + [rem] + remain[pos:]
                        if not _feasible_agv_tokens(trial, aid, task_idx, cap_of, maxd_of):
                            continue
                        c = score(trial, v)
                        if c < best_cost:
                            best_cost, best_pos, best_seq = c, pos, trial
                    if best_seq is None:
                        # 전체 롤백
                        best_seq = original; remain = original; break
                    remain = best_seq
                new_tokens = remain
                if new_tokens != original and score(new_tokens, v) + 1e-9 < score(original, v):
                    if _feasible_agv_tokens(new_tokens, aid, task_idx, cap_of, maxd_of):
                        cand.at[i, "route"] = ",".join(new_tokens); improved = True
                    else:
                        cand.at[i, "route"] = ",".join(original)
                else:
                    cand.at[i, "route"] = ",".join(original)

        if improved and local_score(cand, agv_df, task_df, lam) + 1e-6 < local_score(best, agv_df, task_df, lam):
            best = cand

    return best

def check_rules_per_trip(subm: pd.DataFrame, agv_df: pd.DataFrame, task_df: pd.DataFrame):
    """DEPOT~DEPOT 구간별 capacity/max_distance 위반 여부를 로컬에서 미리 검출"""
    task_idx = task_df.set_index("task_id")
    cap_of   = dict(zip(agv_df["agv_id"], agv_df["capacity"]))
    maxd_of  = dict(zip(agv_df["agv_id"], agv_df["max_distance"]))

    def xy(t):
        if t == "DEPOT": return (0,0)
        d = task_idx.loc[t]
        return int(d["x"]), int(d["y"])

    def mh(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    problems = []
    for _, r in subm.iterrows():
        aid   = r["agv_id"]
        toks  = [t for t in r["route"].split(",") if t]
        # DEPOT 기준으로 트립 분해
        trip, trips = [], []
        for t in toks:
            trip.append(t)
            if t == "DEPOT" and len(trip) > 1:
                trips.append(trip[:])
                trip = [t]  # 다음 트립 시작(DEPOT 포함)

        # 각 트립 검사
        for k, seg in enumerate(trips, 1):
            customers = [t for t in seg if t != "DEPOT"]
            if not customers:
                continue
            demand_sum = int(task_idx.loc[customers, "demand"].sum())
            if demand_sum > int(cap_of[aid]):
                problems.append(f"[CAP] agv={aid} trip#{k} demand_sum={demand_sum} > cap={cap_of[aid]}")
            dist_sum = sum(mh(xy(a), xy(b)) for a,b in zip(seg, seg[1:]))
            if dist_sum > int(maxd_of[aid]):
                problems.append(f"[DIST] agv={aid} trip#{k} dist_sum={dist_sum} > maxd={maxd_of[aid]}")
    if problems:
        print("[RuleCheck] Violations detected:")
        for msg in problems[:30]:
            print(" ", msg)
        raise ValueError("Rule violations found; see logs above.")
    else:
        print("[RuleCheck] All trips satisfy capacity & max_distance.")

def enforce_per_trip_limits(subm: pd.DataFrame, agv_df: pd.DataFrame, task_df: pd.DataFrame) -> pd.DataFrame:
    """
    각 AGV 라우트를 좌->우로 훑으면서
    - 현재 트립에 고객을 1명 더 넣었을 때 cap/maxd를 넘기면,
      그 지점에서 트립을 'DEPOT'으로 끊어 새로운 트립을 시작한다.
    - 순서는 바꾸지 않고 'DEPOT'만 추가하는 안전 리페어.
    - 단일 작업 자체가 cap 또는 maxd를 초과하는 경우는(데이터 자체 문제)
      preflight_check에서 이미 경고가 나와야 정상.
    """
    task_idx = task_df.set_index("task_id")
    cap_of   = dict(zip(agv_df["agv_id"], agv_df["capacity"]))
    maxd_of  = dict(zip(agv_df["agv_id"], agv_df["max_distance"]))

    def xy(tok):
        if tok == "DEPOT": return (0,0)
        d = task_idx.loc[tok]
        return int(d["x"]), int(d["y"])
    def mh(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    new_rows = []
    for _, r in subm.iterrows():
        aid = r["agv_id"]
        cap  = int(cap_of[aid])
        maxd = int(maxd_of[aid])

        toks = [t for t in str(r["route"]).split(",") if t]
        if not toks or toks[0] != "DEPOT":
            toks = ["DEPOT"] + toks
        if toks[-1] != "DEPOT":
            toks = toks + ["DEPOT"]

        customers = [t for t in toks if t != "DEPOT"]

        # 현재 트립 상태
        new_tokens = []          # 리페어 후 최종 토큰
        cur_seg = ["DEPOT"]      # 진행 중인 트립 (항상 DEPOT으로 시작)
        cur_dem = 0              # 현 트립 수요합
        last = "DEPOT"
        dist_without_return = 0  # 현 트립에서 'DEPOT 복귀 제외' 이동합

        for c in customers:
            d_c = int(task_idx.loc[c, "demand"])
            # 이 고객을 추가했을 때, '바로 DEPOT으로 복귀'한다고 가정한 총거리
            cand_without_return = dist_without_return + mh(xy(last), xy(c))
            cand_total_if_close = cand_without_return + mh(xy(c), (0,0))

            if (cur_dem + d_c) <= cap and cand_total_if_close <= maxd:
                # 트립에 수용 가능 → 추가
                cur_seg.append(c)
                cur_dem = cur_dem + d_c
                dist_without_return = cand_without_return
                last = c
            else:
                # 현 트립을 여기서 끊고 DEPOT으로 닫아 저장
                cur_seg.append("DEPOT")
                if not new_tokens:
                    new_tokens.extend(cur_seg)
                else:
                    # 직전이 DEPOT으로 끝났으니 앞의 DEPOT 하나는 제거
                    new_tokens.extend(cur_seg[1:])
                # 새 트립 시작 (DEPOT→c)
                cur_seg = ["DEPOT", c]
                cur_dem = d_c
                dist_without_return = mh((0,0), xy(c))
                last = c

        # 마지막 트립 닫기
        cur_seg.append("DEPOT")
        if not new_tokens:
            new_tokens.extend(cur_seg)
        else:
            new_tokens.extend(cur_seg[1:])

        new_rows.append({"agv_id": aid, "route": ",".join(new_tokens)})

    repaired = pd.DataFrame(new_rows)
    return repaired

# ------------------------------
# 사전 진단 / 하한 추정 / 폴백
# ------------------------------
def preflight_check(agv_df, task_df):
    ok = True
    max_cap = float(agv_df["capacity"].max())
    bad_cap = task_df[(task_df["task_id"]!="DEPOT") & (task_df["demand"] > max_cap)]
    if len(bad_cap):
        ok = False
        print("[Preflight] Capacity infeasible tasks:",
              bad_cap[["task_id","demand"]].to_dict(orient="records"))

    task_df2 = task_df[task_df["task_id"]!="DEPOT"].copy()
    task_df2["min_roundtrip"] = 2*(task_df2["x"].abs() + task_df2["y"].abs())
    max_maxdist = float(agv_df["max_distance"].max())
    bad_dist = task_df2[task_df2["min_roundtrip"] > max_maxdist]
    if len(bad_dist):
        ok = False
        print("[Preflight] MaxDistance infeasible tasks:",
              bad_dist[["task_id","x","y","min_roundtrip"]].to_dict(orient="records"))

    bad_speed = agv_df[(agv_df["speed_cells_per_sec"]<=0) | (agv_df["speed_cells_per_sec"].isna())]
    if len(bad_speed):
        ok = False
        print("[Preflight] Invalid AGV speed records:", bad_speed.to_dict(orient="records"))
    return ok

def suggest_trips(agv_df, task_df):
    total_demand = float(task_df.loc[task_df["task_id"]!="DEPOT","demand"].sum())
    cap_per_trip_sum = float(agv_df["capacity"].sum())
    lb_cap = 1 if cap_per_trip_sum == 0 else math.ceil(total_demand / cap_per_trip_sum)

    task_df2 = task_df[task_df["task_id"]!="DEPOT"].copy()
    total_min_round = float((2*(task_df2["x"].abs()+task_df2["y"].abs())).sum())
    maxd_per_trip_sum = float(agv_df["max_distance"].sum())
    lb_dist = 1 if maxd_per_trip_sum == 0 else math.ceil(total_min_round / maxd_per_trip_sum)

    lb = max(lb_cap, lb_dist)
    return int(max(1, lb))

def greedy_fallback_one_task_per_trip(agv_df, task_df, force_skip_infeasible=True):
    tasks = task_df[task_df["task_id"] != "DEPOT"].copy()
    tasks = tasks.sort_values("deadline").reset_index(drop=True)

    agv_info = {}
    for _, row in agv_df.iterrows():
        aid = row["agv_id"]
        agv_info[aid] = {
            "speed": max(float(row["speed_cells_per_sec"]), 1e-6),
            "cap":   int(row["capacity"]),
            "maxd":  int(row["max_distance"]),
            "time":  0.0,
            "route": ["DEPOT"],
        }

    t_lookup = task_df.set_index("task_id").to_dict(orient="index")
    def mh(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    dropped = []
    for _, r in tasks.iterrows():
        tid = r["task_id"]; tx, ty = int(r["x"]), int(r["y"])
        dem = int(r["demand"]); serv = int(r["service_time"]); dead = int(r["deadline"])
        best = None
        for aid, info in agv_info.items():
            if dem > info["cap"]: continue
            roundtrip_dist = 2 * (abs(tx) + abs(ty))
            if roundtrip_dist > info["maxd"]: continue
            to_task = mh((0,0), (tx,ty)) / info["speed"]
            finish  = info["time"] + to_task + serv
            tard    = max(0.0, finish - dead)
            key = (tard, finish)
            if (best is None) or (key < best[0]):
                best = (key, aid, to_task)
        if best is None:
            msg = f"[Fallback] 작업 {tid}는 모든 AGV에서 단독 왕복도 불가 (demand/cap 또는 max_distance 위반)"
            if force_skip_infeasible:
                print(msg + " -> skip"); dropped.append(tid); continue
            else:
                raise RuntimeError(msg)
        (_, aid, to_task) = best
        info = agv_info[aid]
        if info["route"] and info["route"][-1] == "DEPOT":
            info["route"].extend([tid, "DEPOT"])
        else:
            info["route"].extend(["DEPOT", tid, "DEPOT"])
        back = mh((tx,ty), (0,0)) / info["speed"]
        info["time"] += to_task + serv + back

    if dropped:
        print(f"[Fallback] Skipped infeasible tasks (len={len(dropped)}): {dropped[:20]}{'...' if len(dropped)>20 else ''}")

    out = []
    for aid, info in agv_info.items():
        route = info["route"]
        if route == ["DEPOT"]:
            route = ["DEPOT","DEPOT"]
        out.append({"agv_id": aid, "route": ",".join(route)})
    return pd.DataFrame(out)

# ------------------------------
# (옵션) 초기해 플러그인들 (trips=1에서만)
# ------------------------------
DELTA_FEATS = [
    "agv_speed","agv_cap","agv_mdist",
    "prev_x","prev_y","next_x","next_y",
    "j_x","j_y","j_service","j_deadline",
    "d_prev_j","d_j_next","d_prev_next"
]

def _route_tokens_to_vehicle_routes(routes_by_agv, agv_df, task_df):
    id_map = task_df["task_id"].tolist()
    id_to_node = {tid:i for i,tid in enumerate(id_map)}
    vrs = []
    for aid in agv_df["agv_id"]:
        toks = [t for t in routes_by_agv[aid].split(",") if t]
        if len(toks)>=2 and toks[0]=="DEPOT" and toks[-1]=="DEPOT":
            toks = toks[1:-1]
        vrs.append([id_to_node[t] for t in toks])
    return vrs

def build_init_with_delta(agv_df, task_df, lam, model_path, top_k=5):
    if not os.path.exists(model_path):
        print(f"[Δ-init] model not found: {model_path}; skip")
        return None
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"[Δ-init] load failed: {e}"); return None

    lookup = task_df.set_index("task_id").to_dict(orient="index")
    def xy(tok): return (0,0) if tok=="DEPOT" else (int(lookup[tok]["x"]), int(lookup[tok]["y"]))
    def svc(tok): return 0 if tok=="DEPOT" else int(lookup[tok]["service_time"])
    def ddl(tok): return 10**12 if tok=="DEPOT" else int(lookup[tok]["deadline"])
    def mh(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

    routes = {aid:"DEPOT,DEPOT" for aid in agv_df["agv_id"]}
    remaining = task_df["task_id"].tolist()[1:]
    remaining.sort(key=lambda t: ddl(t))

    def base_df():
        return pd.DataFrame({"agv_id": agv_df["agv_id"], "route":[routes[a] for a in agv_df["agv_id"]]})

    while remaining:
        task_tok = remaining.pop(0)
        cand_meta, feats_rows = [], []
        for _, row in agv_df.iterrows():
            aid   = row["agv_id"]
            speed = float(row["speed_cells_per_sec"])
            cap   = float(row["capacity"])
            mdist = float(row["max_distance"])
            toks = routes[aid].split(",")
            for pos in range(1, len(toks)):
                prev_tok, next_tok = toks[pos-1], toks[pos]
                px,py = xy(prev_tok); nx,ny = xy(next_tok); jx,jy = xy(task_tok)
                feats_rows.append({
                    "agv_speed":speed, "agv_cap":cap, "agv_mdist":mdist,
                    "prev_x":px,"prev_y":py,"next_x":nx,"next_y":ny,
                    "j_x":jx,"j_y":jy,
                    "j_service":svc(task_tok),"j_deadline":ddl(task_tok),
                    "d_prev_j":mh((px,py),(jx,jy)),
                    "d_j_next":mh((jx,jy),(nx,ny)),
                    "d_prev_next":mh((px,py),(nx,ny))
                })
                cand_meta.append((aid, pos, toks))
        if not feats_rows: break
        X = pd.DataFrame(feats_rows)[DELTA_FEATS]
        preds = model.predict(X)
        order = np.argsort(preds)[:max(1, top_k)]
        best_sc, best_choice = float("inf"), None
        for idx in order:
            aid, pos, toks = cand_meta[int(idx)]
            trial = base_df()
            t2 = toks[:pos] + [task_tok] + toks[pos:]
            trial.loc[trial["agv_id"]==aid,"route"] = ",".join(t2)
            sc = local_score(trial, agv_df, task_df, lam)
            if sc < best_sc:
                best_sc, best_choice = sc, (aid, t2)
        if best_choice is None:
            aid = agv_df["agv_id"].iloc[0]
            toks = routes[aid].split(","); toks.insert(1, task_tok)
            routes[aid] = ",".join(toks)
        else:
            aid, t2 = best_choice
            routes[aid] = ",".join(t2)
    return _route_tokens_to_vehicle_routes(routes, agv_df, task_df)

class Pointer(nn.Module):
    def __init__(self, d=64, h=4, in_dim=5):
        super().__init__()
        self.emb = nn.Linear(in_dim, d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=h, batch_first=True),
            num_layers=2
        )
        self.proj = nn.Linear(d, 1)
    def forward(self, feats, mask=None):
        x = self.emb(feats)
        x = self.enc(x, src_key_padding_mask=mask)
        return self.proj(x).squeeze(-1)

def build_init_with_pointer(agv_df, task_df, pointer_path, lam):
    if torch is None or not os.path.exists(pointer_path):
        print("[ptr-init] skip (torch or model missing)")
        return None
    lookup = task_df.set_index("task_id").to_dict(orient="index")
    def feat(tok):
        if tok=="DEPOT": return [0,0,0,10**12,0]
        d = lookup[tok]; return [float(d["x"]), float(d["y"]),
                                 float(d["service_time"]), float(d["deadline"]),
                                 float(d["demand"])]
    model = Pointer(in_dim=5)
    model.load_state_dict(torch.load(pointer_path, map_location="cpu"))
    model.eval()
    tasks = task_df["task_id"].tolist()[1:]
    feats = torch.tensor([feat(t) for t in tasks], dtype=torch.float32)[None, ...]
    with torch.no_grad():
        order = torch.argsort(model(feats), dim=-1, descending=True).squeeze(0).tolist()
    ordered = [tasks[i] for i in order]
    routes = {aid:"DEPOT,DEPOT" for aid in agv_df["agv_id"]}
    def base_df():
        return pd.DataFrame({"agv_id": agv_df["agv_id"], "route":[routes[a] for a in agv_df["agv_id"]]})
    for task_tok in ordered:
        best_sc, best_choice = float("inf"), None
        for aid in agv_df["agv_id"]:
            toks = routes[aid].split(",")
            for pos in range(1, len(toks)):
                trial = base_df()
                t2 = toks[:pos] + [task_tok] + toks[pos:]
                trial.loc[trial["agv_id"]==aid,"route"] = ",".join(t2)
                sc = local_score(trial, agv_df, task_df, lam)
                if sc < best_sc:
                    best_sc, best_choice = sc, (aid, t2)
        if best_choice is None:
            aid = agv_df["agv_id"].iloc[0]
            toks = routes[aid].split(","); toks.insert(1, task_tok)
            routes[aid] = ",".join(toks)
        else:
            aid, t2 = best_choice
            routes[aid] = ",".join(t2)
    return _route_tokens_to_vehicle_routes(routes, agv_df, task_df)

def submission_to_init_routes(subm, agv_df, task_df):
    """submission.csv -> OR-Tools ReadAssignmentFromRoutes용 init_routes(list[list[int]])"""
    id_map = task_df["task_id"].tolist()
    id_to_node = {tid: i for i, tid in enumerate(id_map)}
    routes_by_agv = dict(zip(subm["agv_id"], subm["route"]))
    init = []
    for aid in agv_df["agv_id"]:
        toks = [t for t in routes_by_agv[aid].split(",") if t]
        if len(toks) >= 2 and toks[0] == "DEPOT" and toks[-1] == "DEPOT":
            toks = toks[1:-1]  # depot 제외
        init.append([id_to_node[t] for t in toks])
    return init

# ------------------------------
# OR-Tools 모델링 & 풀이
# ------------------------------
def solve(agv_df, task_df, lam=1.0, timelimit=55, time_scale=100,
          trips_per_agv=1, allow_drop=False, drop_penalty=10**9,
          use_delta=False, delta_model_path="./_data/dacon/route/_save/delta_lgb.pkl", delta_topk=5,
          use_pointer=False, pointer_path="./_data/dacon/route/_save/pointer.pt",
          obstacles: str = "",
          init_routes=None,
          meta_sel: str = "GLS",
          seed_sel: int = -1,
          fs_override=None):

    dist, service, demand, deadline = build_mats(task_df, obstacles_path=obstacles, use_cache=True, cache_dir=BASE)
    n_nodes = dist.shape[0]
    n_real  = len(agv_df)
    n_veh   = n_real * max(1, int(trips_per_agv))

    speeds = np.repeat(agv_df["speed_cells_per_sec"].astype(float).to_numpy(), repeats=trips_per_agv)
    caps   = np.repeat(agv_df["capacity"].astype(int).to_numpy(),             repeats=trips_per_agv)
    maxds  = np.repeat(agv_df["max_distance"].astype(int).to_numpy(),         repeats=trips_per_agv)

    starts = [0]*n_veh
    ends   = [0]*n_veh
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_veh, starts, ends)
    routing = pywrapcp.RoutingModel(manager)
    nid = lambda idx: manager.IndexToNode(idx)

    # Capacity
    def demand_cb(idx): return int(demand[nid(idx)])
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, list(map(int, caps)), True, "Capacity")

    # Distance
    def dist_cb(i,j): return int(dist[nid(i), nid(j)])
    dist_idx = routing.RegisterTransitCallback(dist_cb)
    routing.AddDimension(dist_idx, 0, 10**12, True, "Distance")
    dist_dim = routing.GetDimensionOrDie("Distance")
    for v in range(n_veh):
        dist_dim.CumulVar(routing.End(v)).SetMax(int(maxds[v]))

    # Time (vehicle-dependent)
    time_cb_idx = []
    for v in range(n_veh):
        spd = max(float(speeds[v]), 1e-6)
        def make_cb(speed):
            def cb(i,j):
                from_n, to_n = nid(i), nid(j)
                travel = dist[from_n, to_n] / speed
                val = (travel + service[to_n]) * time_scale
                return int(math.ceil(val))
            return cb
        idx = routing.RegisterTransitCallback(make_cb(spd))
        time_cb_idx.append(idx)

    try:
        routing.AddDimensionWithVehicleTransit(time_cb_idx, 0, [10**12]*n_veh, True, "Time")
    except AttributeError:
        routing.AddDimension(time_cb_idx[0], 0, 10**12, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # Soft deadlines
    penalty_per_unit = max(1, int(round(lam * time_scale)))
    for node in range(1, n_nodes):
        idx = manager.NodeToIndex(node)
        latest = int(deadline[node] * time_scale)
        time_dim.SetCumulVarSoftUpperBound(idx, latest, penalty_per_unit)

    # Start times
    for v in range(n_veh):
        time_dim.CumulVar(routing.Start(v)).SetValue(0)

    # Objective
    for v in range(n_veh):
        routing.SetArcCostEvaluatorOfVehicle(time_cb_idx[v], v)

    # Optional drop
    if allow_drop:
        penalty = int(drop_penalty)
        for node in range(1, n_nodes):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    def make_params(tlim, fs, meta="GLS", seed=-1):
        p = pywrapcp.DefaultRoutingSearchParameters()
        p.first_solution_strategy = fs
        if meta == "SA":
            p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
        elif meta == "TABU":
            p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
        else:
            p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        p.time_limit.FromSeconds(int(tlim))
        p.log_search = True
        if seed is not None and seed >= 0:
            try:
                p.random_seed = int(seed)
            except AttributeError:
                pass
        return p

    first_strats_default = [
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    ]
    first_strats = [fs_override] if fs_override is not None else first_strats_default

    # 초기해 (trips_per_agv==1일 때만)
    if init_routes is None and trips_per_agv == 1:
        if use_pointer:
            init_routes = build_init_with_pointer(agv_df, task_df, pointer_path, lam)
        if init_routes is None and use_delta:
            init_routes = build_init_with_delta(agv_df, task_df, lam, delta_model_path, delta_topk)

    sol = None
    for fs in first_strats:
        p = make_params(timelimit, fs, meta=meta_sel, seed=seed_sel)
        if init_routes is not None:
            assignment0 = routing.ReadAssignmentFromRoutes(init_routes, True)
            sol = routing.SolveFromAssignmentWithParameters(assignment0, p)
        else:
            sol = routing.SolveWithParameters(p)
        if sol is not None:
            print(f"[Info] solved with first_solution_strategy={fs}, meta={meta_sel}, seed={seed_sel}")
            break

    if sol is None:
        print("[Warn] OR-Tools failed. Switching to greedy fallback (one-task-per-trip).")
        return greedy_fallback_one_task_per_trip(agv_df, task_df, force_skip_infeasible=True)

    # 결과 추출 (가상 차량→실제 AGV로 합치기)
    id_map = task_df["task_id"].tolist()
    routes_by_real_agv = {aid: [] for aid in agv_df["agv_id"]}

    for v in range(n_veh):
        idx = routing.Start(v)
        seq_nodes = []
        while not routing.IsEnd(idx):
            seq_nodes.append(nid(idx))
            idx = sol.Value(routing.NextVar(idx))
        seq_nodes.append(nid(idx))
        customer_nodes = [n for n in seq_nodes if n != 0]
        if len(customer_nodes) == 0:
            continue
        labels = [id_map[n] for n in seq_nodes]
        real_idx = v % n_real
        real_id  = agv_df["agv_id"].iloc[real_idx]
        routes_by_real_agv[real_id].append(labels)

    flat_routes = {}
    for aid in agv_df["agv_id"]:
        trips = routes_by_real_agv[aid]
        if not trips:
            route_tokens = ["DEPOT","DEPOT"]
        else:
            route_tokens = []
            for tkns in trips:
                if route_tokens and route_tokens[-1] == "DEPOT" and tkns[0] == "DEPOT":
                    route_tokens.extend(tkns[1:])
                else:
                    route_tokens.extend(tkns)
        flat_routes[aid] = ",".join(route_tokens)

    subm = pd.DataFrame({"agv_id": agv_df["agv_id"], "route": agv_df["agv_id"].map(flat_routes)})
    return subm

# ------------------------------
# main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--timelimit", type=int, default=60)
    ap.add_argument("--trips", dest="trips", type=int, default=1, help="각 AGV의 허용 왕복 수(가상 차량 수)")
    ap.add_argument("--meta", type=str, default="GLS", choices=["GLS","SA","TABU"],
                    help="로컬 메타휴리스틱: GLS(기본) / SA / TABU")
    ap.add_argument("--seed", type=int, default=-1, help="OR-Tools random_seed (-1이면 설정 안 함)")
    # 디버그/옵션
    ap.add_argument("--allow_drop", action="store_true")
    ap.add_argument("--drop_penalty", type=int, default=10**9)
    ap.add_argument("--obstacles", type=str, default="", help="장애물 CSV 경로(x,y 컬럼). 없으면 맨해튼 사용")
    # 초기해 플러그인 (trips=1에서만 적용)
    ap.add_argument("--use_delta", action="store_true")
    ap.add_argument("--delta_model_path", type=str, default="./_data/dacon/route/_save/delta_lgb.pkl")
    ap.add_argument("--delta_topk", type=int, default=5)
    ap.add_argument("--use_pointer", action="store_true")
    ap.add_argument("--pointer_path", type=str, default="./_data/dacon/route/_save/pointer.pt")
    args = ap.parse_args()

    agv_df, task_df = load_data(AGV_PATH, TASK_PATH)
    preflight_check(agv_df, task_df)
    # --- trips 하한 결정: (용량/거리 하한 ∪ 연결성 하한) ---
    lb_capdist = suggest_trips(agv_df, task_df)

    conn_lb = 1
    if args.obstacles:
        # 장애물 거리로 연결성 분석
        dist_obs, _, _, _ = build_mats(task_df, obstacles_path=args.obstacles, use_cache=True, cache_dir=BASE)
        unreachable, comp_cnt = analyze_obstacle_connectivity(task_df, dist_obs, INF=INF)
        if unreachable:
            print(f"[Connectivity] Unreachable from DEPOT (len={len(unreachable)}): {unreachable[:20]}{'...' if len(unreachable)>20 else ''}")
        # 같은 트립에서 묶으려면 작업 그래프가 연결이어야 하므로 연결 성분 수만큼 트립 필요
        conn_lb = max(1, comp_cnt)

    hard_lb = max(lb_capdist, conn_lb)

    if args.trips <= 0:
        print(f"[Info] auto trips lower-bound = {hard_lb} (cap/dist={lb_capdist}, connectivity={conn_lb})")
        args.trips = hard_lb
    elif args.trips < hard_lb:
        print(f"[Info] bump trips from {args.trips} -> {hard_lb} (cap/dist={lb_capdist}, connectivity={conn_lb})")
        args.trips = hard_lb


    # === MultiRun 2-Stage Orchestration ===
    candidates = [
        {"trips": args.trips,   "timelimit": args.timelimit,                     "meta": args.meta, "seed": args.seed, "fs": None},
        {"trips": args.trips+1, "timelimit": max(40, args.timelimit//2 + 20),   "meta": "SA",      "seed": (args.seed if args.seed>=0 else 7), "fs": routing_enums_pb2.FirstSolutionStrategy.SAVINGS},
    ]

    best_subm, best_sc = None, float("inf")
    for cfg in candidates:
        # 1단계(근사 거리; 장애물 무시) 빠른 초기해
        s1 = solve(
            agv_df, task_df,
            lam=args.lam, timelimit=max(20, cfg["timelimit"]//2), time_scale=100,
            trips_per_agv=cfg["trips"],
            allow_drop=args.allow_drop, drop_penalty=args.drop_penalty,
            use_delta=args.use_delta, delta_model_path=args.delta_model_path, delta_topk=args.delta_topk,
            use_pointer=args.use_pointer, pointer_path=args.pointer_path,
            obstacles="",  # 근사
            meta_sel=cfg["meta"], seed_sel=cfg["seed"], fs_override=cfg["fs"],
        )
        init_routes = submission_to_init_routes(s1, agv_df, task_df)
        # 2단계(정확 거리; 장애물 반영) 폴리싱 + 초기해 주입
        s2 = solve(
            agv_df, task_df,
            lam=args.lam, timelimit=cfg["timelimit"], time_scale=100,
            trips_per_agv=cfg["trips"],
            allow_drop=args.allow_drop, drop_penalty=args.drop_penalty,
            use_delta=False, use_pointer=False,
            obstacles=args.obstacles,
            init_routes=init_routes,
            meta_sel=cfg["meta"], seed_sel=cfg["seed"], fs_override=cfg["fs"],
        )
        # 사후 LNS
        s2 = post_lns(s2, agv_df, task_df, lam=args.lam, iters=40, ruin_rate=0.1, per_trip_only=True)
        sc2 = local_score(s2, agv_df, task_df, lam=args.lam)
        if sc2 < best_sc:
            best_sc, best_subm = sc2, s2
    subm = best_subm
    print(f"[MultiRun] best local score = {best_sc:.2f}")

    # --- 후처리 순서 ---
    # 1) 제약 인지형 LNS (권장: per_trip_only=True)
    subm = post_lns(subm, agv_df, task_df, lam=args.lam, iters=50, ruin_rate=0.1, per_trip_only=True)

    # 2) 라우트 문자열 압축/정리
    subm["route"] = subm["route"].map(squash_and_clean_route_str)

    # 2.5) ★ 트립 한도 강제 리페어 ★
    subm = enforce_per_trip_limits(subm, agv_df, task_df)
    subm["route"] = subm["route"].map(squash_and_clean_route_str)

    # 3) 제출형식 정규화/커버리지 검사
    subm = normalize_and_validate_submission(subm, agv_df, task_df)

    # 4) 트립 단위 규칙(용량/거리) 최종검사 — 이제 통과해야 정상
    check_rules_per_trip(subm, agv_df, task_df)


    # 5) 저장
    KST = timezone(timedelta(hours=9))
    ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    out_path_ts = os.path.join(BASE, f"submission_{ts}.csv")
    subm.to_csv(out_path_ts, index=False, encoding="utf-8")
    subm.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"[OK] saved -> {out_path_ts}")
    print(f"[OK] saved -> {OUT_PATH}")


    # 참고용 로컬 스코어
    try:
        sc = local_score(subm, agv_df, task_df, lam=args.lam)
        print(f"[Local Score λ={args.lam}] {sc:.4f}")
    except Exception as e:
        print(f"[Warn] local scoring failed: {e}")

if __name__ == "__main__":
    main()