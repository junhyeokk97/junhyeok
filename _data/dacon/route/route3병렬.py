# agv_optimizer.py
# EDF Greedy + ALNS(Shaw/Random/Worst) + Regret-3 + 2-opt / 2-opt* / Or-opt / relocate
# Grid shortest path: Manhattan / Dijkstra / A* (with cache & obstacle CSV)
# Multi-method ensemble search -> evaluate on a single fair matrix
# Save with timestamp

import os
import pandas as pd
import numpy as np
import random
from copy import deepcopy
from datetime import datetime
from math import exp, atan2
import heapq, hashlib, json
from multiprocessing import get_context, cpu_count

# -------------------- user paths --------------------
PATH = './_data/dacon/route/'
SAVE_PATH = './_data/dacon/route/'
os.makedirs(SAVE_PATH, exist_ok=True)

# -------------------- constants / tunables --------------------
DEPOT_XY = (0, 0)
DEPOT = "DEPOT"

# 거리/장애물
OBSTACLE_CSV = os.path.join(PATH, "obstacles.csv")   # 없으면 맨해튼
GRID_SHORTEST = "astar"                              # 기본 탐색 메소드
ENSEMBLE_METHODS = ["manhattan", "dijkstra", "astar"]  # 앙상블 시도
USE_CACHE = True

# 대회 점수 관련
LAMBDA = 0.69

# 탐색/무브 파라미터
RANDOM_SEED = 190
LOCAL_ITERS = 12000           # 약간 늘려서 탐색 여유
SA_START_TEMP = 1.45
SA_END_TEMP   = 0.10
REHEAT_AFTER  = 800           # 재가열을 조금 더 자주
REHEAT_TEMP   = 0.65

# 지각 억제 쪽으로 destroy 강도 상향
BASE_DESTROY_FRAC = 0.18
HI_DESTROY_FRAC   = 0.28

# 초기화는 빠르게, 그러나 품질도 챙기기
USE_FAST_INIT          = True     # (느리면 True 권장)
TIME_BUDGET_INIT_SEC   = 20       # 초기 빌더 시간 예산 줄여 대기 방지
BATCH_SIZE_INIT        = 32       # 한 라운드 작업 수
MAX_POS_PER_ROUTE      = 6        # 라우트당 삽입 위치 샘플 축소
INIT_2OPT_EVERY        = 50       # 너무 자주 2-opt 안함(속도↑)

USE_REGRET_INIT        = True
REGRET_INIT_K          = 3
REGRET_INIT_SAMPLES    = 12       # 작업당 위치 샘플 축소(속도↑)
DO_INIT_2OPT_POLISH    = True


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

INF = 10**9

# ---- boosting model (optional) ----
USE_GBM = True
GBM_PATH = os.path.join(SAVE_PATH, 'delta_lgb.pkl')
try:
    import joblib
    GBM_MODEL = joblib.load(GBM_PATH) if USE_GBM and os.path.exists(GBM_PATH) else None
except Exception:
    GBM_MODEL = None

# -------------------- helpers (distance / obstacles) --------------------
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def load_obstacles(path):
    if not path or not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return {(int(x), int(y)) for x, y in zip(df["x"], df["y"])}

def _bbox(points, margin=20):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
    margin = max(margin, (abs(maxx-minx)+abs(maxy-miny))//2 + 5)
    return (minx - margin, maxx + margin, miny - margin, maxy + margin)

def _dijkstra_row(src_xy, targets_set, blocked, bbox):
    xmin, xmax, ymin, ymax = bbox
    pq = [(0, src_xy)]
    dist = {src_xy: 0}
    remain = set(targets_set) - {src_xy}
    out = {}
    while pq and remain:
        d, (x, y) = heapq.heappop(pq)
        if d != dist[(x, y)]:
            continue
        if (x, y) in remain:
            out[(x, y)] = d
            remain.remove((x, y))
            if not remain:
                break
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if nx < xmin or nx > xmax or ny < ymin or ny > ymax: continue
            if (nx, ny) in blocked: continue
            nd = d + 1
            if nd < dist.get((nx, ny), INF):
                dist[(nx, ny)] = nd
                heapq.heappush(pq, (nd, (nx, ny)))
    return out

def _astar_one(src_xy, tgt_xy, blocked, bbox):
    if src_xy == tgt_xy: return 0
    xmin, xmax, ymin, ymax = bbox
    tx, ty = tgt_xy
    def h(ax, ay): return abs(ax - tx) + abs(ay - ty)
    g = {src_xy: 0}
    pq = [(h(*src_xy), 0, src_xy)]  # (f, g, (x,y))
    seen = set()
    while pq:
        f, gc, (x,y) = heapq.heappop(pq)
        if (x,y) in seen: continue
        seen.add((x,y))
        if (x,y) == tgt_xy: return gc
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if nx < xmin or nx > xmax or ny < ymin or ny > ymax: continue
            if (nx, ny) in blocked: continue
            ng = gc + 1
            if ng < g.get((nx,ny), INF):
                g[(nx,ny)] = ng
                heapq.heappush(pq, (ng + h(nx,ny), ng, (nx,ny)))
    return INF

def _dist_cache_key(points, blocked, method):
    h = hashlib.sha1()
    h.update(json.dumps(sorted(points)).encode())
    h.update(json.dumps(sorted(blocked)).encode())
    h.update(method.encode())
    return h.hexdigest()[:12]

def _try_load_cache(cache_dir, key):
    path = os.path.join(cache_dir, f"dist_{key}.npy")
    return (np.load(path) if os.path.exists(path) else None), path

def _save_cache(dist, path):
    np.save(path, dist)

def build_dist_matrix(points, blocked, method="astar", use_cache=True, cache_dir=SAVE_PATH):
    """
    points: [(x,y)] including DEPOT first, then tasks
    method: "manhattan" | "dijkstra" | "astar"
    """
    if method == "manhattan" or not blocked:
        n = len(points)
        dist = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            xi, yi = points[i]
            for j in range(n):
                xj, yj = points[j]
                dist[i, j] = abs(xi-xj) + abs(yi-yj)
        return dist

    key = _dist_cache_key(points, blocked, method) if use_cache else None
    dist_mat, cache_path = (None, None)
    if use_cache:
        dist_mat, cache_path = _try_load_cache(cache_dir, key)
    if dist_mat is not None:
        return dist_mat

    N = len(points)
    dist_mat = np.full((N, N), INF, dtype=np.int32)
    for i in range(N): dist_mat[i, i] = 0
    bbox = _bbox(points, margin=20)
    targets = set(points)

    if method == "dijkstra":
        def _row(src): return _dijkstra_row(src, targets, blocked, bbox)
        for i, src in enumerate(points):
            found = _row(src)
            for j, pj in enumerate(points):
                if pj in found: dist_mat[i, j] = found[pj]
    else:  # astar
        for i, src in enumerate(points):
            for j in range(i+1, N):
                tgt = points[j]
                d = _astar_one(src, tgt, blocked, bbox)
                dist_mat[i, j] = d
                dist_mat[j, i] = d

    if use_cache and cache_path:
        _save_cache(dist_mat, cache_path)
    return dist_mat

# 전역 거리 행렬/인덱스
DIST_MAT = None
XY_TO_IDX = None
def prepare_grid_distance(tasks_dict, method_override=None):
    global DIST_MAT, XY_TO_IDX
    points = [DEPOT_XY] + [tasks_dict[tid]["pos"] for tid in tasks_dict.keys()]
    blocked = load_obstacles(OBSTACLE_CSV)
    method = (method_override or GRID_SHORTEST).lower()
    if method not in ("manhattan","dijkstra","astar"):
        method = "manhattan"
    DIST_MAT = build_dist_matrix(points, blocked, method=method, use_cache=USE_CACHE, cache_dir=SAVE_PATH)
    XY_TO_IDX = {xy: i for i, xy in enumerate(points)}

def grid_distance_xy(a_xy, b_xy):
    if DIST_MAT is None or XY_TO_IDX is None:
        return manhattan(a_xy, b_xy)
    ia = XY_TO_IDX.get(a_xy); ib = XY_TO_IDX.get(b_xy)
    if ia is None or ib is None:
        return manhattan(a_xy, b_xy)
    return int(DIST_MAT[ia, ib])

def move_time(u, v, speed):
    return grid_distance_xy(u, v) / speed

# -------------------- load data --------------------
agv_path  = os.path.join(PATH, "agv.csv")
task_path = os.path.join(PATH, "task.csv")
agv_df  = pd.read_csv(agv_path)
task_df = pd.read_csv(task_path)
agv_df.columns  = [c.strip().lower() for c in agv_df.columns]
task_df.columns = [c.strip().lower() for c in task_df.columns]

agvs = {
    r["agv_id"]: dict(
        speed=float(r["speed_cells_per_sec"]),
        capacity=float(r["capacity"]),
        max_distance=float(r["max_distance"])
    )
    for _, r in agv_df.iterrows()
}
tasks = {
    r["task_id"]: dict(
        pos=(int(r["x"]), int(r["y"])),
        service_time=float(r["service_time"]),
        deadline=float(r["deadline"]),
        demand=float(r.get("demand", 0))
    )
    for _, r in task_df.iterrows()
}

# 전역 거리 준비
prepare_grid_distance(tasks)

task_ids = list(tasks.keys())
agv_ids = list(agvs.keys())

# -------------------- AGV state class --------------------
class AGVState:
    def __init__(self, agv_id, speed, capacity, max_distance):
        self.agv_id = agv_id
        self.speed = float(speed)
        self.capacity = float(capacity)
        self.max_distance = float(max_distance)
        self.route = [DEPOT]
        self.last_xy = DEPOT_XY
        self.tour_dist = 0.0
        self.tour_load = 0.0
        self.tour_has_task = False
        self.clock = 0.0

    def can_add_now(self, task):
        (tx, ty), dem = task["pos"], task["demand"]
        d_to = grid_distance_xy(self.last_xy, (tx, ty))
        d_back = grid_distance_xy((tx, ty), DEPOT_XY)
        if d_to >= INF or d_back >= INF: return False
        if self.tour_dist + d_to + d_back > self.max_distance: return False
        if self.tour_load + dem > self.capacity: return False
        return True

    def add_task(self, tid, task):
        (tx, ty) = task["pos"]
        d = grid_distance_xy(self.last_xy, (tx, ty))
        self.clock += d / self.speed
        self.last_xy = (tx, ty)
        self.tour_dist += d
        self.tour_load += task["demand"]
        self.route.append(tid)
        self.clock += task["service_time"]
        self.tour_has_task = True

    def close_tour(self):
        if self.route and self.route[-1] != DEPOT:
            d_back = grid_distance_xy(self.last_xy, DEPOT_XY)
            self.clock += d_back / self.speed
            self.last_xy = DEPOT_XY
            self.route.append(DEPOT)
        self.tour_dist = 0.0
        self.tour_load = 0.0
        self.tour_has_task = False

# -------------------- scoring & validity --------------------
def score_from_solution(sol_routes, verbose=False):
    total_move = 0.0
    total_service = 0.0
    total_penalty = 0.0
    completion_times = {}

    for agv_id, route in sol_routes.items():
        spec = agvs[agv_id]; speed = spec["speed"]
        clock = 0.0; pos = DEPOT_XY
        tour_dist_acc = 0.0; tour_load_acc = 0.0
        for tok in route[1:]:
            if tok == DEPOT:
                d = grid_distance_xy(pos, DEPOT_XY)
                if d >= INF: return float("inf")
                clock += d / speed; total_move += d / speed
                tour_dist_acc = 0.0; tour_load_acc = 0.0
                pos = DEPOT_XY
            else:
                task = tasks[tok]
                d = grid_distance_xy(pos, task["pos"])
                if d >= INF: return float("inf")
                clock += d / speed; total_move += d / speed
                tour_dist_acc += d
                clock += task["service_time"]; total_service += task["service_time"]
                pos = task["pos"]; tour_load_acc += task["demand"]
                completion_times[tok] = clock

    for tid, tinfo in tasks.items():
        compl = completion_times.get(tid, None)
        total_penalty += (1e6 if compl is None else max(0.0, compl - tinfo["deadline"]))

    total_score = total_move + total_service + LAMBDA * total_penalty
    if verbose:
        print(f"move: {total_move:.3f}, svc: {total_service:.3f}, pen: {total_penalty:.3f}")
    return total_score

def score_with_matrix(sol_routes, dist_mat, xy_to_idx, lam=LAMBDA, verbose=False):
    def _dist(a, b):
        ia = xy_to_idx.get(a); ib = xy_to_idx.get(b)
        if ia is None or ib is None: return manhattan(a,b)
        return int(dist_mat[ia, ib])

    total_move = 0.0; total_service = 0.0; total_penalty = 0.0
    completion = {}
    for agv_id, route in sol_routes.items():
        spec = agvs[agv_id]; speed = spec["speed"]
        clock = 0.0; pos = DEPOT_XY
        for tok in route[1:]:
            if tok == DEPOT:
                d = _dist(pos, DEPOT_XY)
                if d >= INF: return float("inf")
                clock += d / speed; total_move += d / speed
                pos = DEPOT_XY
            else:
                t = tasks[tok]
                d = _dist(pos, t["pos"])
                if d >= INF: return float("inf")
                clock += d / speed; total_move += d / speed
                clock += t["service_time"]; total_service += t["service_time"]
                pos = t["pos"]; completion[tok] = clock
    for tid, tinfo in tasks.items():
        c = completion.get(tid)
        total_penalty += (1e6 if c is None else max(0.0, c - tinfo["deadline"]))
    sc = total_move + total_service + lam * total_penalty
    if verbose:
        print(f"[EVAL] move={total_move:.3f} svc={total_service:.3f} pen={total_penalty:.3f} -> {sc:.3f}")
    return sc

def check_validity(sol_routes):
    visited = set()
    for agv_id, route in sol_routes.items():
        spec = agvs[agv_id]; max_d = spec["max_distance"]; cap = spec["capacity"]
        pos = DEPOT_XY; trip_dist = 0; trip_load = 0
        for tok in route[1:]:
            if tok == DEPOT:
                d_back = grid_distance_xy(pos, DEPOT_XY)
                if d_back >= INF: return False
                trip_dist += d_back
                if trip_dist > max_d + 1e-6 or trip_load > cap + 1e-6: return False
                pos = DEPOT_XY; trip_dist = 0; trip_load = 0
            else:
                if tok in visited: return False
                visited.add(tok)
                t = tasks[tok]
                d = grid_distance_xy(pos, t["pos"])
                if d >= INF: return False
                trip_dist += d; trip_load += t["demand"]; pos = t["pos"]
    return len(visited) == len(tasks)

def get_completion_and_lateness(sol_routes):
    completion, lateness = {}, {}
    for agv_id, route in sol_routes.items():
        speed = agvs[agv_id]["speed"]; clock = 0.0; pos = DEPOT_XY
        for tok in route[1:]:
            if tok == DEPOT:
                d = grid_distance_xy(pos, DEPOT_XY); 
                if d >= INF: return {}, {}
                clock += d / speed; pos = DEPOT_XY
            else:
                t = tasks[tok]; d = grid_distance_xy(pos, t["pos"])
                if d >= INF: return {}, {}
                clock += d / speed; clock += t["service_time"]; pos = t["pos"]
                completion[tok] = clock
    for tid, tinfo in tasks.items():
        c = completion.get(tid, None)
        lateness[tid] = (1e9 if c is None else max(0.0, c - tinfo["deadline"]))
    return completion, lateness

# -------------------- initial greedy (EDF) --------------------
def build_initial_solution():
    agv_states = [AGVState(aid, spec["speed"], spec["capacity"], spec["max_distance"])
                  for aid, spec in agvs.items()]
    ordered = sorted(tasks.keys(), key=lambda tid: tasks[tid]["deadline"])
    for tid in ordered:
        t = tasks[tid]; best = None
        for i, agv in enumerate(agv_states):
            if agv.can_add_now(t):
                finish = agv.clock + move_time(agv.last_xy, t["pos"], agv.speed) + t["service_time"]
                if best is None or finish < best[0]: best = (finish, i, False)
            else:
                dmin = grid_distance_xy(DEPOT_XY, t["pos"]) + grid_distance_xy(t["pos"], DEPOT_XY)
                if dmin < INF and dmin <= agv.max_distance and t["demand"] <= agv.capacity:
                    finish = (agv.clock + move_time(agv.last_xy, DEPOT_XY, agv.speed)
                              + move_time(DEPOT_XY, t["pos"], agv.speed) + t["service_time"])
                    if best is None or finish < best[0]: best = (finish, i, True)
        if best is None:
            # force open on first feasible agv
            for i, agv in enumerate(agv_states):
                dmin = grid_distance_xy(DEPOT_XY, t["pos"]) + grid_distance_xy(t["pos"], DEPOT_XY)
                if dmin < INF and dmin <= agv.max_distance and t["demand"] <= agv.capacity:
                    agv.close_tour(); agv.add_task(tid, t); break
        else:
            _, i, need_close = best
            if need_close: agv_states[i].close_tour()
            agv_states[i].add_task(tid, t)

    sol = {}
    for a in agv_states:
        if a.tour_has_task: a.close_tour()
        cleaned = []
        for tok in a.route:
            if cleaned and cleaned[-1] == DEPOT and tok == DEPOT: continue
            cleaned.append(tok)
        if cleaned == [DEPOT]: continue
        if cleaned[-1] != DEPOT: cleaned.append(DEPOT)
        sol[a.agv_id] = cleaned
    return sol

def build_initial_solution_regret_fast():
    """Cheapest-insertion 근사 + regret-k 선택 + 시간예산. 장애물 없거나 약할 때 매우 빠름."""
    start_t = time.time()
    # 각 AGV 라우트 시작은 [DEPOT, DEPOT]
    sol = {aid: [DEPOT, DEPOT] for aid in agv_ids}
    specs = {aid: agvs[aid] for aid in agv_ids}

    remaining = list(tasks.keys())
    # 마감 임박 순으로 정렬
    remaining.sort(key=lambda t: tasks[t]["deadline"])

    inserted = 0
    while remaining:
        if time.time() - start_t > TIME_BUDGET_INIT_SEC:
            break  # 시간예산 초과 시 중단(남은 건 이후 탐색에서 채움)

        batch = remaining[:min(BATCH_SIZE_INIT, len(remaining))]
        best_pick = None  # (regret, tid, best_aid, best_pos, best_delta)

        for tid in batch:
            # 라우트별 pos 후보를 윈도우 샘플링
            cand_costs = []
            for aid, r in sol.items():
                pos_list = _trip_windows(r, MAX_POS_PER_ROUTE)
                for pos in pos_list:
                    if not _feasible_local_after_insert(specs[aid], r, pos, tid):
                        continue
                    delta = _delta_travel_on_insert(r, pos, tid)
                    cand_costs.append((delta, aid, pos))

            if not cand_costs:
                continue

            cand_costs.sort(key=lambda z: z[0])
            k = min(REGRET_INIT_K, len(cand_costs))
            regret = 0.0 if k==1 else (cand_costs[1][0] - cand_costs[0][0])
            pick = (regret, tid, cand_costs[0][1], cand_costs[0][2], cand_costs[0][0])
            if (best_pick is None) or (pick[0] > best_pick[0]):
                best_pick = pick

        if best_pick is None:
            # 어디에도 못 넣은 경우: 제일 짧은 라우트 끝에 강제 삽입 시도
            tid = remaining[0]
            lens = [(sum(1 for x in sol[a] if x!=DEPOT), a) for a in sol]
            aid = min(lens)[1]
            pos = len(sol[aid])-1
            sol[aid] = _cleanup(sol[aid][:pos] + [tid] + sol[aid][pos:])
            remaining.pop(0); inserted += 1
        else:
            _, tid, aid, pos, _ = best_pick
            sol[aid] = _cleanup(sol[aid][:pos] + [tid] + sol[aid][pos:])
            remaining.remove(tid); inserted += 1

            # 주기적으로 해당 trip 2-opt
            if DO_2OPT and inserted % INIT_2OPT_EVERY == 0:
                sol[aid] = two_opt_intra(sol[aid])

    # 남은 작업이 있다면(시간예산 때문에), 기존 EDF로 밀어넣기
    if remaining:
        # 간단 EDF로 나머지 채우기
        agv_states = [AGVState(aid, specs[aid]["speed"], specs[aid]["capacity"], specs[aid]["max_distance"])
                      for aid in agv_ids]
        # 현재 sol을 states에 로드
        for st in agv_states:
            st.route = sol[st.agv_id][:]
            # last_xy, tour_dist/load는 rough로만 복구; close_tour로 초기화 의존
            st.last_xy = DEPOT_XY
            st.tour_dist = 0.0; st.tour_load = 0.0; st.tour_has_task = False; st.clock = 0.0
        for tid in remaining:
            t = tasks[tid]
            # 가장 빨리 끝낼 수 있는 AGV
            best = None
            for st in agv_states:
                # 강제 새 트립 열기
                finish = st.clock + move_time(st.last_xy, DEPOT_XY, st.speed) \
                                   + move_time(DEPOT_XY, t["pos"], st.speed) + t["service_time"]
                if (best is None) or (finish < best[0]): best = (finish, st.agv_id)
            sol[best[1]] = _cleanup(sol[best[1]][:-1] + [tid] + [DEPOT])

    # 최종 클린업
    for a in sol.keys():
        sol[a] = _cleanup(sol[a])
    # 안전 차원에서 1회 전역 valid 체크 후 불가하면 기존 빌더로 폴백
    if not check_validity(sol):
        return build_initial_solution()  # EDF fallback
    return sol

def build_initial_solution_regret():
    """빈 해에서 시작해 모든 작업을 regret-k 기준으로 '최적 위치'에 삽입."""
    # AGV 루트 초기화: 각 AGV를 [DEPOT, DEPOT]로 시작(삽입 편하게)
    sol = {aid: [DEPOT, DEPOT] for aid in agv_ids}

    remaining = set(tasks.keys())
    # 우선순위: 마감 임박 순
    ordered = sorted(remaining, key=lambda t: tasks[t]["deadline"])

    while remaining:
        # 급한 작업 위주로 샘플링
        sample = ordered[:max(1, min(len(ordered), 64))]
        best_pick = None  # (regret, tid, best_trial, best_score)

        for tid in sample:
            candidates = []
            for a_id, r in sol.items():
                positions = list(range(1, len(r)))  # 맨 앞 DEPOT 앞은 금지
                if len(positions) > REGRET_INIT_SAMPLES:
                    positions = random.sample(positions, REGRET_INIT_SAMPLES)
                for pos in positions:
                    tri, sc = _try_insert(sol, tid, a_id, pos)
                    if tri is not None:
                        candidates.append((sc, tri))

            if not candidates:
                # 어디에도 삽입 실패 → 말단에 강제 삽입 시도
                for a_id in sol.keys():
                    tri, sc = _try_insert(sol, tid, a_id, len(sol[a_id]) - 1)
                    if tri is not None:
                        candidates = [(sc, tri)]
                        break
                if not candidates:
                    continue

            candidates.sort(key=lambda z: z[0])
            top = candidates[:max(1, REGRET_INIT_K)]
            regret = 0.0 if len(top) == 1 else (top[1][0] - top[0][0])
            pick = (regret, tid, top[0][1], top[0][0])
            if (best_pick is None) or (pick[0] > best_pick[0]):
                best_pick = pick

        if best_pick is None:
            # 비정상 보호: 아무데나 하나 박고 진행
            tid = next(iter(remaining))
            for a_id in sol.keys():
                tri, sc = _try_insert(sol, tid, a_id, len(sol[a_id]) - 1)
                if tri is not None:
                    sol = tri
                    ordered.remove(tid); remaining.remove(tid)
                    break
            continue

        _, tid, best_tri, _ = best_pick
        sol = best_tri
        ordered.remove(tid)
        remaining.remove(tid)

    # 마무리: 클린업 + 초기 2-opt 폴리시(옵션)
    for a in sol.keys():
        sol[a] = _cleanup_route_tokens(sol[a])
    if DO_INIT_2OPT_POLISH:
        for a in list(sol.keys()):
            sol[a] = two_opt_intra(sol[a])
    return sol
# -------------------- local moves --------------------
def _cleanup(tokens):
    out = []
    for x in tokens:
        if out and out[-1] == DEPOT and x == DEPOT: continue
        out.append(x)
    if not out or out[0] != DEPOT: out = [DEPOT] + out
    if out[-1] != DEPOT: out.append(DEPOT)
    return out

def two_opt_intra(route):
    # between two depots; reverse subpath
    splits, last = [], 0
    for i, x in enumerate(route):
        if x == DEPOT:
            splits.append((last, i))
            last = i
    segs = [s for s in splits if s[1]-s[0] >= 3]
    if not segs: return route
    s, e = random.choice(segs)
    a = random.randint(s+1, e-2)
    b = random.randint(a+1, e-1)
    return route[:a] + list(reversed(route[a:b+1])) + route[b+1:]

def _next_depot_idx(route, i):
    j = i
    while j < len(route) and route[j] != DEPOT: j += 1
    return j  # j is index of next DEPOT (or len)

def two_opt_star_inter(sol_routes):
    keys = list(sol_routes.keys())
    if len(keys) < 2: return sol_routes
    a1, a2 = random.sample(keys, 2)
    r1, r2 = sol_routes[a1], sol_routes[a2]
    pos1 = [i for i,x in enumerate(r1) if x != DEPOT]
    pos2 = [i for i,x in enumerate(r2) if x != DEPOT]
    if not pos1 or not pos2: return sol_routes
    i = random.choice(pos1); j = random.choice(pos2)
    e1 = _next_depot_idx(r1, i); e2 = _next_depot_idx(r2, j)
    tail1 = r1[i:e1]; tail2 = r2[j:e2]
    new_r1 = _cleanup(r1[:i] + tail2 + r1[e1:])
    new_r2 = _cleanup(r2[:j] + tail1 + r2[e2:])
    new = deepcopy(sol_routes); new[a1] = new_r1; new[a2] = new_r2
    return new if check_validity(new) else sol_routes

def or_opt(sol_routes, chain_len=1):
    # pick a route and chain within a trip, relocate to best global position
    cand = deepcopy(sol_routes)
    all_spans = []
    for a_id, r in cand.items():
        depots = [idx for idx,x in enumerate(r) if x==DEPOT]
        for k in range(len(depots)-1):
            s, e = depots[k], depots[k+1]
            for i in range(s+1, e):
                if i+chain_len-1 >= e: break
                all_spans.append((a_id, i, i+chain_len))  # [i, j)
    if not all_spans: return sol_routes
    a_id, i, j = random.choice(all_spans)
    chain = cand[a_id][i:j]
    cand[a_id] = _cleanup(cand[a_id][:i] + cand[a_id][j:])
    best = None
    for b_id, r in cand.items():
        for pos in range(1, len(r)):
            trial = deepcopy(cand)
            trial[b_id] = r[:pos] + chain + r[pos:]
            trial[b_id] = _cleanup(trial[b_id])
            if not check_validity(trial): continue
            sc = score_from_solution(trial)
            if (best is None) or sc < best[0]: best = (sc, trial)
    return best[1] if best else sol_routes

def swap_between_routes(sol_routes):
    keys = list(sol_routes.keys())
    if len(keys) < 2: return sol_routes
    a1, a2 = random.sample(keys, 2)
    r1, r2 = sol_routes[a1], sol_routes[a2]
    t1 = [i for i,x in enumerate(r1) if x != DEPOT]
    t2 = [i for i,x in enumerate(r2) if x != DEPOT]
    if not t1 or not t2: return sol_routes
    i1 = random.choice(t1); i2 = random.choice(t2)
    new_r1 = r1[:i1] + [r2[i2]] + r1[i1+1:]
    new_r2 = r2[:i2] + [r1[i1]] + r2[i2+1:]
    new = deepcopy(sol_routes); new[a1]=_cleanup(new_r1); new[a2]=_cleanup(new_r2)
    return new if check_validity(new) else sol_routes

def relocate_random_best(sol_routes, max_trials=30):
    candidate = deepcopy(sol_routes)
    all_tasks = []
    for a_id, r in candidate.items():
        for i, tok in enumerate(r):
            if tok != DEPOT: all_tasks.append((a_id, i, tok))
    if not all_tasks: return sol_routes
    for _ in range(max_trials):
        a_id, idx, tid = random.choice(all_tasks)
        tmp = deepcopy(candidate)
        tmp[a_id].pop(idx); tmp[a_id] = _cleanup(tmp[a_id])
        # best insertion anywhere
        best = None
        for b_id, r in tmp.items():
            for pos in range(1, len(r)):
                trial = deepcopy(tmp)
                trial[b_id] = _cleanup(r[:pos] + [tid] + r[pos:])
                if not check_validity(trial): continue
                sc = score_from_solution(trial)
                if (best is None) or sc < best[0]: best = (sc, trial)
        if best and best[0] + 1e-9 < score_from_solution(candidate): return best[1]
    return sol_routes

import time

def _trip_windows(route, max_pos=MAX_POS_PER_ROUTE):
    """DEPOT-DEPOT 구간마다 균등 윈도우로 일부 pos만 샘플."""
    depots = [i for i,x in enumerate(route) if x==DEPOT]
    pos_list = []
    for k in range(len(depots)-1):
        s, e = depots[k], depots[k+1]  # [s, e]
        cand = list(range(s+1, e+1))   # 삽입 pos: s+1..e
        if len(cand) > max_pos:
            step = max(1, len(cand)//max_pos)
            cand = cand[::step]
            if cand and cand[-1] != e: cand.append(e)
        pos_list.extend(cand)
    return pos_list

def _delta_travel_on_insert(r, pos, tid):
    """여행거리 근사Δ: (prev,tid)+(tid,next)-(prev,next). 장애물 없을 때 매우 정확."""
    if pos<1 or pos>len(r): return float("inf")
    prev = r[pos-1]
    nxt  = r[pos] if pos < len(r) else DEPOT
    px = DEPOT_XY if prev==DEPOT else tasks[prev]["pos"]
    nx = DEPOT_XY if nxt ==DEPOT else tasks[nxt ]["pos"]
    tx = tasks[tid]["pos"]
    return manhattan(px,tx)+manhattan(tx,nx)-manhattan(px,nx)

def _feasible_local_after_insert(state_spec, route, pos, tid):
    """용량/최대거리 대략 체크: 해당 trip 내 거리합만 근사로 본다."""
    cap = state_spec["capacity"]; max_d = state_spec["max_distance"]
    # trip 경계 찾기
    s = pos-1
    while s>=0 and route[s]!=DEPOT: s-=1
    e = pos
    while e<len(route) and route[e]!=DEPOT: e+=1
    trip = route[s+1:e]  # 기존 trip tasks
    # 수요
    load = sum(tasks[x]["demand"] for x in trip if x!=DEPOT) + tasks[tid]["demand"]
    if load > cap + 1e-9: return False
    # 거리(맨해튼 근사)
    nodes = [DEPOT] + trip + [DEPOT]
    d0 = 0
    for i in range(len(nodes)-1):
        a = DEPOT_XY if nodes[i]==DEPOT else tasks[nodes[i]]["pos"]
        b = DEPOT_XY if nodes[i+1]==DEPOT else tasks[nodes[i+1]]["pos"]
        d0 += manhattan(a,b)
    # 삽입 후 근사거리
    r2 = route[:pos] + [tid] + route[pos:]
    # trip 재계산
    trip2 = r2[s+1 : e+1]
    nodes2 = [DEPOT] + trip2 + [DEPOT]
    d1 = 0
    for i in range(len(nodes2)-1):
        a = DEPOT_XY if nodes2[i]==DEPOT else tasks[nodes2[i]]["pos"]
        b = DEPOT_XY if nodes2[i+1]==DEPOT else tasks[nodes2[i+1]]["pos"]
        d1 += manhattan(a,b)
    return d1 <= max_d + 1e-6

def _cleanup_route_tokens(tokens):
    out = []
    for x in tokens:
        if out and out[-1] == DEPOT and x == DEPOT:
            continue
        out.append(x)
    if not out or out[0] != DEPOT:
        out = [DEPOT] + out
    if out[-1] != DEPOT:
        out.append(DEPOT)
    return out

def _try_insert(sol_routes, tid, a_id, pos):
    """a_id 경로의 pos 위치에 tid 삽입 시도 → (trial_sol, score) 또는 (None, inf)"""
    tri = deepcopy(sol_routes)
    r = tri[a_id]
    tri[a_id] = _cleanup_route_tokens(r[:pos] + [tid] + r[pos:])
    if not check_validity(tri):
        return None, float("inf")
    return tri, score_from_solution(tri)

# -------------------- ALNS operators --------------------
def _remove_task_once(sol_routes, tid):
    sol = deepcopy(sol_routes)
    for a_id, r in sol.items():
        if tid in r:
            idx = r.index(tid)
            r.pop(idx); sol[a_id] = _cleanup(r); return sol
    return sol

def random_remove(sol_routes, q):
    all_t = []
    for a_id, r in sol_routes.items():
        for t in r:
            if t != DEPOT: all_t.append(t)
    removed = set(random.sample(all_t, min(q, len(all_t))))
    cur = deepcopy(sol_routes)
    for t in removed: cur = _remove_task_once(cur, t)
    return cur, list(removed)

def worst_lateness_remove(sol_routes, q):
    _, late = get_completion_and_lateness(sol_routes)
    pos = [t for t,v in late.items() if v>0]
    if not pos:
        return random_remove(sol_routes, q)
    pos.sort(key=lambda t: late[t], reverse=True)
    removed = pos[:min(q, len(pos))]
    cur = deepcopy(sol_routes)
    for t in removed: cur = _remove_task_once(cur, t)
    return cur, removed

def shaw_remove(sol_routes, q, alpha=1.0, beta=0.3, gamma=0.1):
    all_t = []
    for a_id, r in sol_routes.items():
        for t in r:
            if t != DEPOT: all_t.append(t)
    if not all_t: return deepcopy(sol_routes), []
    # seed: worst lateness if exists, else random
    _, late = get_completion_and_lateness(sol_routes)
    late_pos = [t for t,v in late.items() if v>0]
    seed = random.choice(late_pos) if late_pos else random.choice(all_t)
    def rel(i,j):
        Ti, Tj = tasks[i], tasks[j]
        return (alpha*manhattan(Ti["pos"], Tj["pos"])
             +  beta*abs(Ti["deadline"]-Tj["deadline"])
             +  gamma*abs(Ti["demand"]  -Tj["demand"]))
    cand = [t for t in all_t if t != seed]
    cand.sort(key=lambda t: rel(seed, t))
    removed = [seed] + cand[:max(0, q-1)]
    cur = deepcopy(sol_routes)
    for t in removed: cur = _remove_task_once(cur, t)
    return cur, removed

def regret_insert(sol_routes, removed_list, k=3):
    cur = deepcopy(sol_routes)
    rem = removed_list[:]
    while rem:
        best_pick = None
        for tid in rem:
            costs = []
            for a_id, r in cur.items():
                for pos in range(1, len(r)):
                    trial = deepcopy(cur)
                    trial[a_id] = _cleanup(r[:pos] + [tid] + r[pos:])
                    if not check_validity(trial): continue
                    costs.append(score_from_solution(trial))
            if not costs: continue
            costs.sort()
            regret = costs[min(k-1, len(costs)-1)] - costs[0]
            pick = (regret, tid, costs[0])
            if (best_pick is None) or (pick[0] > best_pick[0]): best_pick = pick
        if best_pick is None: break
        _, tid, best_sc = best_pick
        best_trial = None
        for a_id, r in cur.items():
            for pos in range(1, len(r)):
                trial = deepcopy(cur)
                trial[a_id] = _cleanup(r[:pos] + [tid] + r[pos:])
                if not check_validity(trial): continue
                sc = score_from_solution(trial)
                if abs(sc - best_sc) < 1e-9: best_trial = trial; break
            if best_trial: break
        cur = best_trial if best_trial else cur
        rem.remove(tid)
    return cur

def guided_swap(candidate, K=12, S=96):
    keys = list(candidate.keys())
    pairs = []
    for _ in range(S):
        if len(keys) < 2: break
        a1, a2 = random.sample(keys, 2)
        r1, r2 = candidate[a1], candidate[a2]
        tpos1 = [i for i,x in enumerate(r1) if x!=DEPOT]
        tpos2 = [i for i,x in enumerate(r2) if x!=DEPOT]
        if not tpos1 or not tpos2: continue
        i1 = random.choice(tpos1); i2 = random.choice(tpos2)
        pairs.append((a1,a2,i1,i2))
    if not pairs: return candidate
    if GBM_MODEL is None:
        a1,a2,i1,i2 = random.choice(pairs)
        new = deepcopy(candidate)
        new[a1] = _cleanup(candidate[a1][:i1] + [candidate[a2][i2]] + candidate[a1][i1+1:])
        new[a2] = _cleanup(candidate[a2][:i2] + [candidate[a1][i1]] + candidate[a2][i2+1:])
        return new if check_validity(new) else candidate
    X, metas = [], []
    for a1,a2,i1,i2 in pairs:
        t1, t2 = candidate[a1][i1], candidate[a2][i2]
        if t1==DEPOT or t2==DEPOT: continue
        T1,T2 = tasks[t1], tasks[t2]; A1,A2 = agvs[a1], agvs[a2]
        X.append(np.array([
            A1["speed"],A1["capacity"],A1["max_distance"],
            A2["speed"],A2["capacity"],A2["max_distance"],
            T1["pos"][0],T1["pos"][1],T1["service_time"],T1["deadline"],T1["demand"],
            T2["pos"][0],T2["pos"][1],T2["service_time"],T2["deadline"],T2["demand"]
        ], dtype=float))
        metas.append((a1,a2,i1,i2))
    if not X: return candidate
    preds = GBM_MODEL.predict(np.vstack(X))
    order = np.argsort(preds)[:K]
    for idx in order:
        a1,a2,i1,i2 = metas[int(idx)]
        new = deepcopy(candidate)
        new[a1] = _cleanup(candidate[a1][:i1] + [candidate[a2][i2]] + candidate[a1][i1+1:])
        new[a2] = _cleanup(candidate[a2][:i2] + [candidate[a1][i1]] + candidate[a2][i2+1:])
        if check_validity(new): return new
    return candidate

# -------------------- optimization driver (ALNS + SA) --------------------
def local_search(initial_sol):
    best_sol = deepcopy(initial_sol)
    best_score = score_from_solution(best_sol)
    cur_sol = deepcopy(best_sol)
    cur_score = best_score

    if not check_validity(cur_sol):
        print("Initial solution invalid (some tasks unassigned). Proceeding anyway.")

    # 적응 가중치
    ops = ["worst","random","shaw"]
    op_score = {op:1.0 for op in ops}
    op_count = {op:1 for op in ops}

    iters = LOCAL_ITERS
    no_improve = 0

    for it in range(iters):
        frac = it / max(1, iters-1)
        base_temp = SA_START_TEMP * (1-frac) + SA_END_TEMP * frac
        temp = max(base_temp, REHEAT_TEMP*SA_START_TEMP) if no_improve >= REHEAT_AFTER else base_temp
        q_frac = HI_DESTROY_FRAC if no_improve >= REHEAT_AFTER else BASE_DESTROY_FRAC

        # ---- ALNS destroy&repair (확률 0.65) ----
        use_alns = (random.random() < 0.75)
        if use_alns:
            q = max(1, int(q_frac * len(tasks)))
            weights = [op_score[o]/op_count[o] for o in ops]
            chosen = random.choices(ops, weights=weights, k=1)[0]

            if chosen == "worst":
                removed_sol, removed_list = worst_lateness_remove(cur_sol, q)
            elif chosen == "random":
                removed_sol, removed_list = random_remove(cur_sol, q)
            else:
                removed_sol, removed_list = shaw_remove(cur_sol, q)

            candidate = regret_insert(removed_sol, removed_list, k=3)
        else:
            # 로컬 무브 믹스
            candidate = deepcopy(cur_sol)
            r = random.random()
            if DO_2OPT and r < 0.25:
                aid = random.choice(list(candidate.keys()))
                candidate[aid] = two_opt_intra(candidate[aid])
            elif DO_2OPT_STAR and r < 0.50:
                candidate = two_opt_star_inter(candidate)
            elif DO_OROPT and r < 0.75:
                L = random.choice([1,2,3])
                candidate = or_opt(candidate, chain_len=L)
            elif DO_SWAP:
                try:
                    candidate = guided_swap(candidate, K=12, S=96)
                except NameError:
                    candidate = swap_between_routes(candidate)
            else:
                candidate = relocate_random_best(candidate, max_trials=30)

        if not check_validity(candidate):
            continue

        cand_score = score_from_solution(candidate)
        delta = cand_score - cur_score
        accepted = (delta < 0) or (random.random() < exp(-delta / max(1e-9, temp)))

        if accepted:
            cur_sol = candidate; cur_score = cand_score
            if cur_score + 1e-9 < best_score:
                best_score = cur_score; best_sol = deepcopy(cur_sol)
                no_improve = 0
                print(f"[iter {it}] improved -> {best_score:.3f}")
                if use_alns:
                    op_score[chosen] += 6.0; op_count[chosen] += 1
            else:
                no_improve += 1
                if use_alns:
                    op_score[chosen] += 2.0; op_count[chosen] += 1
                    
        # --- 미세 폴리싱: 지각 최악 작업이 든 라우트에 intra 2-opt 1회 ---
        _, late_map = get_completion_and_lateness(cur_sol)
        if late_map:
            worst_tid = max(late_map.items(), key=lambda z: z[1])[0]
            for _aid, _r in cur_sol.items():
                if worst_tid in _r:
                    tweaked = deepcopy(cur_sol)
                    tweaked[_aid] = two_opt_intra(_r)
                    if check_validity(tweaked):
                        sc2 = score_from_solution(tweaked)
                        if sc2 + 1e-9 < cur_score:
                            cur_sol = tweaked
                            cur_score = sc2
                    break
        else:
            no_improve += 1
            if use_alns:
                op_count[chosen] += 1  # 시도만 증가

    return best_sol, best_score

# -------------------- run pipeline --------------------
if __name__ == "__main__":
    blocked_exists = os.path.exists(OBSTACLE_CSV)
    print(f"Obstacles: {'YES' if blocked_exists else 'NO'}")
    print(f"Default search method = {GRID_SHORTEST}, Ensemble = {ENSEMBLE_METHODS if ENSEMBLE_METHODS else 'OFF'}")

    # 공정한 평가 기준: 장애물 있으면 A*, 없으면 맨해튼
    points_eval = [DEPOT_XY] + [tasks[tid]["pos"] for tid in tasks.keys()]
    blocked = load_obstacles(OBSTACLE_CSV)
    eval_method = "astar" if blocked else "manhattan"
    EVAL_DIST = build_dist_matrix(points_eval, blocked, method=eval_method, use_cache=USE_CACHE, cache_dir=SAVE_PATH)
    EVAL_XY2I = {xy: i for i, xy in enumerate(points_eval)}

    # 장애물이 없으면 맨해튼만: 계산/캐시 오버헤드 줄이고 초기 안정화
    if not blocked:
        methods_to_try = ["manhattan"]
    else:
        methods_to_try = ENSEMBLE_METHODS if ENSEMBLE_METHODS else [GRID_SHORTEST]

    overall_best_sol = None
    overall_best_eval = float("inf")

    for m in methods_to_try:
        print("="*60)
        print(f"[RUN] method = {m}")
        prepare_grid_distance(tasks, method_override=m)

        print("Building initial solution (FAST regret-k)...")
        init_sol = build_initial_solution_regret_fast()

        init_score_internal = score_from_solution(init_sol, verbose=True)
        print(f"[{m}] initial (internal) = {init_score_internal:.3f}")

        print("Starting ALNS + SA local search...")
        best_sol_m, best_score_internal = local_search(init_sol)
        print(f"[{m}] best (internal) = {best_score_internal:.3f}")

        # 후처리 폴리싱: worst lateness ruin&repair 2~3라운드
        def _polish(sol):
            cur = deepcopy(sol)
            for _ in range(3):
                cur2, rm = worst_lateness_remove(cur, max(1, int(0.15*len(tasks))))
                cur = regret_insert(cur2, rm, k=3)
            return cur
        polished = _polish(best_sol_m)
        pol_sc = score_from_solution(polished)
        if pol_sc + 1e-9 < best_score_internal:
            best_sol_m, best_score_internal = polished, pol_sc
            print(f"[{m}] improved by polish -> {best_score_internal:.3f}")

        # 공통 EVAL 매트릭스로 재채점
        best_score_eval = score_with_matrix(best_sol_m, EVAL_DIST, EVAL_XY2I, lam=LAMBDA, verbose=True)
        print(f"[{m}] best (EVAL={eval_method}) = {best_score_eval:.3f}")

        if best_score_eval < overall_best_eval:
            overall_best_eval = best_score_eval
            overall_best_sol = deepcopy(best_sol_m)

    if overall_best_sol is None:
        overall_best_sol = init_sol
        overall_best_eval = score_with_matrix(init_sol, EVAL_DIST, EVAL_XY2I, lam=LAMBDA)

    print("="*60)
    print(f"[FINAL] best score on EVAL({eval_method}) = {overall_best_eval:.3f}")

    rows = []
    for agv in sorted(overall_best_sol.keys()):
        rows.append({"agv_id": agv, "route": ",".join(overall_best_sol[agv])})
    out_df = pd.DataFrame(rows, columns=["agv_id","route"]).sort_values("agv_id")

    ts = datetime.now().strftime("%m%d%H%M%S")
    save_file = os.path.join(SAVE_PATH, f"submission_{ts}.csv")
    out_df.to_csv(save_file, index=False)
    print(f"[제출 파일 저장] {save_file}")
    print(out_df.head())

    report_file = os.path.join(SAVE_PATH, f"score_report_{ts}.txt")
    with open(report_file, "w") as f:
        f.write(f"best_eval_score: {overall_best_eval}\n")
        f.write(f"eval_method: {eval_method}\n")
        f.write(f"methods_tried: {methods_to_try}\n")
        f.write(f"LAMBDA: {LAMBDA}\n")
        f.write(f"LOCAL_ITERS: {LOCAL_ITERS}\n")
        f.write(f"RANDOM_SEED: {RANDOM_SEED}\n")
    print(f"[레포트 저장] {report_file}")
