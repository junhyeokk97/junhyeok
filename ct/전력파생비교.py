import pandas as pd
import numpy as np
import re
import os
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
warnings.filterwarnings("ignore")

import holidays
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ----------------- 로깅 -----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _normalize_btype_col(df: pd.DataFrame, allowed_types: set[str]) -> pd.DataFrame:
    """
    '호텔호텔호텔' 같이 동일 토큰 반복되면 한 번으로 압축.
    allowed_types(= building_info에 있는 정식 유형) 안에 있을 때만 축소.
    """
    s = df['건물유형'].astype(str).str.strip()

    # 케이스 1: 정식 토큰의 반복만으로 이루어진 경우 (ex. (호텔)+ )
    for tok in sorted([t for t in allowed_types if isinstance(t, str) and t], key=len, reverse=True):
        pat = re.compile(rf'^(?:{re.escape(tok)})+$')  # "호텔"만 반복되는 문자열에 매치
        m = s.str.fullmatch(pat)
        if m.any():
            s.loc[m] = tok

    # 케이스 2: 일반 반복 패턴(길이 1~6 토큰)도 allowed인 경우만 축소
    def _collapse_general(x: str) -> str:
        m = re.fullmatch(r'^(.{1,6})\1{1,}$', x)  # 같은 토큰이 2회 이상 반복
        if m:
            cand = m.group(1)
            if cand in allowed_types:
                return cand
        return x

    df['건물유형'] = s.apply(_collapse_general)
    return df

# ----------------- 설정 -----------------
class Config:
    # (기존) 재현성/기본 랜덤 스테이트
    RANDOM_STATE = 42

    # === Optuna(탐색) 쪽 ===
    OPTUNA_TRIALS = 75        # 하이퍼파라미터 탐색 시 시도 횟수
    OPTUNA_SEED = 75          # TPE sampler 시드

    # === CV/분할 설정 ===
    CV_FOLDS = 3
    MIN_VAL_HOURS = 24
    INNER_VAL_RATIO = 0.2
    

    VAL_DATE = datetime(2024, 8, 18)

    # === 학습(반복) 쪽 ===
    # 같은 최적 파라미터로 학습을 TRAIN_REPEATS번 수행(씨드만 바꿔) → 예측 평균
    TRAIN_REPEATS = 3
    TRAIN_SEED_BASE = 42

    # 앙상블 가중치 (초기값)
    ENSEMBLE_WEIGHTS = {'xgb': 0.6, 'lgb': 0.35, 'cat': 0.05}

    # ==== 중요도 기반 피처 가중치(NEW) ====
    IMPORTANCE_WEIGHTING = True        # 중요도 가중치 적용 여부
    ENABLE_FEATURE_SELECTION = False   # 중요도 임계/TopK로 피처를 '제거'하지 않음
    FEATURE_WEIGHT_MIN = 0.6           # 가중치 하한 (0~1) — 낮을수록 약하게, 1이면 변화 없음
    FEATURE_WEIGHT_GAMMA = 1.3         # 샤프닝 지수(>1이면 상위 피처 더 강조)
    APPLY_FW = {'xgb': True, 'lgb': True, 'cat': True}  # 모델별 가중치 적용

    # 플롯 저장 여부(요청: 저장하지 않음)
    SAVE_FI_PLOT = False

    # 증강 설정(학습 데이터에만 적용)
    USE_AUGMENT = True
    AUG_N_COPIES = 3        # 복제본 수(원본 + n개)
    AUG_NOISE_STD = 0.05    # 표준편차 스케일(컬럼 표준편차에 곱해 사용)
    
    # (참고용) 중요도 계산 파라미터
    USE_FEATURE_IMPORTANCE = True
    IMPORTANCE_THRESHOLD = 0.01
    TOP_K_FEATURES = 180

# ----------------- 경로 -----------------
path = './_data/dacon/electric/'

# ----------------- 지표 -----------------
def smape(y_true, y_pred):
    """SMAPE 계산 함수"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    eps = 1e-9
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

# ----------------- 데이터 로드 -----------------
def load_data():
    """데이터 로드"""
    try:
        train_df = pd.read_csv(os.path.join(path, 'train.csv'), index_col=0)
        test_df = pd.read_csv(os.path.join(path, 'test.csv'), index_col=0)
        building_df = pd.read_csv(os.path.join(path, 'building_info.csv'))
        building_df = building_df.rename(columns={'건물번호': '건물번호'})

        logger.info(f"데이터 로드 완료 - Train: {train_df.shape}, Test: {test_df.shape}")
        return train_df, test_df, building_df
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        raise

# ----------------- 파생피처 -----------------
def _add_time_features(df):
    df['year'] = df['일시'].dt.year
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['weekday'] = df['일시'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # 주기형
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    # 업무시간/피크 플래그
    df['is_workhour'] = ((df['hour'] >= 9) & (df['hour'] < 18)).astype(int)
    df['is_peak'] = df['hour'].isin([10, 11, 18, 19, 20]).astype(int)

    return df

def _add_holiday_features(df):
    kr_holidays = holidays.KR()
    d = df['일시'].dt.date
    df['is_holiday'] = d.map(lambda x: 1 if x in kr_holidays else 0)
    # 전/후 공휴일
    prev_dates = (df['일시'] - pd.Timedelta(days=1)).dt.date
    next_dates = (df['일시'] + pd.Timedelta(days=1)).dt.date
    df['is_prev_holiday'] = prev_dates.map(lambda x: 1 if x in kr_holidays else 0)
    df['is_next_holiday'] = next_dates.map(lambda x: 1 if x in kr_holidays else 0)
    df['is_workingday'] = ((df['is_holiday'] == 0) & (df['is_weekend'] == 0)).astype(int)
    return df

def _add_capacity_building_features(df):
    # 숫자 변환
    for c in ['ESS저장용량(kWh)', '태양광용량(kW)', 'PCS용량(kW)', '연면적(m2)', '냉방면적(m2)']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 존재 플래그
    df['has_pv']  = (df.get('태양광용량(kW)', 0).fillna(0) > 0).astype(int)
    df['has_ess'] = (df.get('ESS저장용량(kWh)', 0).fillna(0) > 0).astype(int)
    df['has_pcs'] = (df.get('PCS용량(kW)', 0).fillna(0) > 0).astype(int)

    # 냉방면적비
    eps = 1e-6
    if '연면적(m2)' in df.columns and '냉방면적(m2)' in df.columns:
        df['냉방면적비'] = (df['냉방면적(m2)'] / (df['연면적(m2)'] + eps)).clip(lower=0)

    # 용량/면적 로그 스케일 (단위 혼재 → 합산치/면적을 log1p)
    cap_sum = df.get('태양광용량(kW)', 0).fillna(0).astype(float) \
              + df.get('PCS용량(kW)', 0).fillna(0).astype(float) \
              + df.get('ESS저장용량(kWh)', 0).fillna(0).astype(float)
    area = df.get('연면적(m2)', 0).fillna(0).astype(float) + 1e-3
    df['cap_per_area_log1p'] = np.log1p((cap_sum / area).values)

    return df

def _build_type_profiles(ref_df: pd.DataFrame) -> dict:
    g = ref_df.copy()

    # 안전: 필요한 컬럼 존재 보장
    need_cols = ['건물유형', 'hour', 'weekday', 'month', 'is_workhour', 'is_workingday',
                 '전력소비량(kWh)', '연면적(m2)', '기온(°C)', '습도(%)']
    miss = [c for c in need_cols if c not in g.columns]
    if miss:
        raise ValueError(f"프로파일 생성에 필요한 컬럼 없음: {miss}")

    # ====== 시간대별/요일별/월별 유형 평균 ======
    type_hour = (
        g.groupby(['건물유형', 'hour'])['전력소비량(kWh)']
         .mean().unstack('hour').add_prefix('type_hour_mean_')
    )
    type_wday = (
        g.groupby(['건물유형', 'weekday'])['전력소비량(kWh)']
         .mean().unstack('weekday').add_prefix('type_wday_mean_')
    )
    type_month = (
        g.groupby(['건물유형', 'month'])['전력소비량(kWh)']
         .mean().unstack('month').add_prefix('type_month_mean_')
    )

    # ====== 정적 프로파일 ======
    by_type = g.groupby('건물유형', dropna=False)

    def _per_area_mean(df):
        area = pd.to_numeric(df['연면적(m2)'], errors='coerce').replace(0, np.nan)
        return float(np.nanmean(df['전력소비량(kWh)'] / area))

    type_per_area = by_type.apply(_per_area_mean).rename('type_kwh_per_area_mean')

    wh_ratio = (
        by_type.apply(lambda df: df.loc[df['is_workhour']==1, '전력소비량(kWh)'].mean()) /
        by_type.apply(lambda df: df.loc[df['is_workhour']==0, '전력소비량(kWh)'].mean())
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0).rename('type_workhour_ratio')

    hol_ratio = (
        by_type.apply(lambda df: df.loc[df['is_workingday']==0, '전력소비량(kWh)'].mean()) /
        by_type.apply(lambda df: df.loc[df['is_workingday']==1, '전력소비량(kWh)'].mean())
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0).rename('type_holiday_ratio')

    def _slope(df, col):
        x = pd.to_numeric(df[col], errors='coerce')
        y = pd.to_numeric(df['전력소비량(kWh)'], errors='coerce')
        vx = np.nanvar(x)
        if not np.isfinite(vx) or vx == 0:
            return 0.0
        cov = np.nanmean((x - np.nanmean(x)) * (y - np.nanmean(y)))
        return float(cov / vx)

    type_temp_beta = by_type.apply(lambda d: _slope(d, '기온(°C)')).rename('type_temp_beta')
    type_hum_beta  = by_type.apply(lambda d: _slope(d, '습도(%)')).rename('type_humidity_beta')

    cap_area_mean   = by_type['cap_per_area_log1p'].mean().rename('type_avg_cap_per_area_log1p')
    cool_ratio_mean = by_type['냉방면적비'].mean().rename('type_avg_cool_ratio')
    has_ess_mean    = by_type['has_ess'].mean().rename('type_avg_has_ess')

    # [ADD] 야간/주말 비중(share) 계산
    night_mask   = g['hour'].isin([0,1,2,3,4,5,6])
    weekend_mask = (g['weekday'] >= 5)

    sum_all     = by_type['전력소비량(kWh)'].sum()
    sum_night   = g.loc[night_mask].groupby('건물유형', dropna=False)['전력소비량(kWh)'].sum()
    sum_weekend = g.loc[weekend_mask].groupby('건물유형', dropna=False)['전력소비량(kWh)'].sum()

    type_night_share   = (sum_night   / sum_all).reindex(sum_all.index).fillna(0.0).rename('type_night_share')
    type_weekend_share = (sum_weekend / sum_all).reindex(sum_all.index).fillna(0.0).rename('type_weekend_share')

    mn_night = _mean_on(g.loc[night_mask],   ['건물유형'], '전력소비량(kWh)')
    mn_day   = _mean_on(g.loc[~night_mask],  ['건물유형'], '전력소비량(kWh)')
    type_night_ratio = (
        mn_night.merge(mn_day, on='건물유형', how='outer', suffixes=('_night','_day'))
                .set_index('건물유형')
                .pipe(lambda df: (df['mean_night']/df['mean_day'])
                                .replace([np.inf,-np.inf], np.nan)
                                .fillna(1.0)
                                .rename('type_night_ratio'))
    )

    mn_weekend = _mean_on(g.loc[weekend_mask],  ['건물유형'], '전력소비량(kWh)')
    mn_weekday = _mean_on(g.loc[~weekend_mask], ['건물유형'], '전력소비량(kWh)')
    type_weekend_ratio = (
        mn_weekend.merge(mn_weekday, on='건물유형', how='outer', suffixes=('_weekend','_weekday'))
                .set_index('건물유형')
                .pipe(lambda df: (df['mean_weekend']/df['mean_weekday'])
                                .replace([np.inf,-np.inf], np.nan)
                                .fillna(1.0)
                                .rename('type_weekend_ratio'))
    )


    type_static = pd.concat(
        [
            # 기존 항목들
            type_per_area, wh_ratio, hol_ratio, type_temp_beta, type_hum_beta,
            # 여기에 3개 평균을 포함
            cap_area_mean, cool_ratio_mean, has_ess_mean,
            # (이미 있다면) 야간/주말 집중도/비율 등도 이어서 포함
            type_night_share, type_weekend_share, type_night_ratio, type_weekend_ratio,
        ],
        axis=1
    )

    return {
        'type_hour':  type_hour.reset_index(),
        'type_wday':  type_wday.reset_index(),
        'type_month': type_month.reset_index(),
        'type_static': type_static.reset_index()
    }

def _impute_target_timewise(df: pd.DataFrame, ref_df: pd.DataFrame,
                            target_col: str = '전력소비량(kWh)') -> pd.DataFrame:
    out = df.copy()
    # 숫자화
    if target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors='coerce')

    # 정렬 및 키
    out = out.sort_values(['건물번호', '일시'])
    out['__date'] = out['일시'].dt.date

    # (1) 같은 건물 × 같은 날짜 내에서 ffill/bfill
    out[target_col] = (
        out.groupby(['건물번호', '__date'])[target_col]
           .transform(lambda s: s.ffill().bfill())
    )

    # (2) 그래도 남으면 같은 건물 전체 시계열에서 ffill/bfill
    out[target_col] = out.groupby('건물번호')[target_col].ffill().bfill()

    # 마무리
    out.drop(columns=['__date'], inplace=True, errors='ignore')
    out[target_col] = pd.to_numeric(out[target_col], errors='coerce').astype(float).clip(lower=0)
    return out




def add_building_type_features(train_df: pd.DataFrame,
                               other_df: pd.DataFrame,
                               ref_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prof = _build_type_profiles(ref_df)

    def _merge_type_features(df):
        out = df.copy()

        # == 여기서 tbl 안 써도 됨: ==
        for key in ('hour', 'wday', 'month'):
            out = out.merge(prof[f'type_{key}'], on='건물유형', how='left')

        out = out.merge(prof['type_static'], on='건물유형', how='left')

        # 숫자화는 안전하게
        type_cols = [c for c in out.columns if c.startswith('type_')]
        if type_cols:
            out[type_cols] = out[type_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        return out

    return _merge_type_features(train_df), _merge_type_features(other_df)

def _add_degree_days(df, base_cool=24.0, base_heat=18.0):
    if '기온(°C)' in df.columns:
        t = pd.to_numeric(df['기온(°C)'], errors='coerce')
        df['CDD'] = np.maximum(0.0, (t - base_cool).astype(float))
        df['HDD'] = np.maximum(0.0, (base_heat - t).astype(float))
    else:
        df['CDD'] = 0.0
        df['HDD'] = 0.0
    return df

def _add_extra_weather_calendar_features_noaa(
    df: pd.DataFrame,
    default_lat: float = 36.5,
    default_lon: float = 127.5,
    tz_hours: int = 9) -> pd.DataFrame:
    out = df.copy()

    for c in ['기온(°C)', '습도(%)', '풍속(m/s)', '강수량(mm)']:
        if c not in out.columns:
            out[c] = 0.0
    out['습도(%)'] = pd.to_numeric(out['습도(%)'], errors='coerce').clip(0, 100)
    out['기온(°C)'] = pd.to_numeric(out['기온(°C)'], errors='coerce')
    out['풍속(m/s)'] = pd.to_numeric(out['풍속(m/s)'], errors='coerce').clip(lower=0)
    out['강수량(mm)'] = pd.to_numeric(out['강수량(mm)'], errors='coerce').clip(lower=0)

    t = out['일시']
    hour_f = t.dt.hour + t.dt.minute/60.0 + t.dt.second/3600.0
    out['day_of_year']     = t.dt.dayofyear.astype(int)
    out['SIN_day_of_year'] = np.sin(2*np.pi*out['day_of_year']/365.25)
    out['COS_day_of_year'] = np.cos(2*np.pi*out['day_of_year']/365.25)

    out['COS_월']   = np.cos(2*np.pi*out['month']/12.0)
    out['SIN_요일'] = np.sin(2*np.pi*out['weekday']/7.0)
    out['COS_요일'] = np.cos(2*np.pi*out['weekday']/7.0)
    out['peak_time'] = out['hour'].isin([10, 11, 18, 19, 20]).astype(int)

    T  = out['기온(°C)']
    RH = out['습도(%)']
    W  = out['풍속(m/s)']

    a, b = 17.27, 237.7
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = (a*T/(b+T)) + np.log((RH/100.0).replace(0, np.nan))
    dewpt = (b*gamma)/(a-gamma)
    out['이슬점온도'] = dewpt.fillna(dewpt.median())
    out['이슬점차이'] = (T - out['이슬점온도']).clip(-50, 50)

    e = 6.105 * np.exp((17.27*T)/(237.7+T)) * (RH/100.0)
    apparent = T + 0.33*e - 0.70*W - 4.0
    out['체감온도'] = apparent
    out['불쾌지수'] = 0.81*T + 0.01*RH*(0.99*T - 14.3) + 46.3
    out['perceived_temperature'] = out['체감온도']

    dep = (T - out['이슬점온도']).abs()
    out['cloudy_based_on_humidity'] = (RH/100.0).clip(0, 1)
    out['cloud_cover_dewpoint'] = (1.0 - np.tanh(dep/10.0)).clip(0, 1)
    out['cloud_cover_rain'] = np.where(out['강수량(mm)'] <= 0,
                                       0.0,
                                       (out['강수량(mm)']/(out['강수량(mm)']+2)).clip(0, 1))
    out['cloud_cover_humidity'] = out['cloudy_based_on_humidity']
    out['cloud_cover_combined'] = (
        0.5*out['cloud_cover_humidity'] +
        0.3*out['cloud_cover_dewpoint'] +
        0.2*out['cloud_cover_rain']
    ).clip(0, 1)
    out['cloud_cover_weighted'] = out['cloud_cover_combined']

    out['low_cloud_prob']  = np.clip(0.7*out['cloud_cover_dewpoint'] + 0.3*out['cloud_cover_humidity'], 0, 1)
    out['mid_cloud_prob']  = np.clip(0.5*out['cloud_cover_humidity'] + 0.5*out['cloud_cover_rain'], 0, 1)
    out['high_cloud_prob'] = np.clip(0.6*out['cloud_cover_humidity'] + 0.4*(1 - out['cloud_cover_dewpoint']), 0, 1)
    out['cumulonimbus_prob'] = np.clip(0.6*np.maximum(T-26, 0)/10.0 + 0.4*(out['강수량(mm)']>5).astype(float), 0, 1)

    out['cloud_okta'] = (out['cloud_cover_combined']*8).round().astype(int).clip(0, 8)
    def _sky_from_okta(okta):
        if okta <= 1: return 'Clear'
        if okta <= 2: return 'Few'
        if okta <= 4: return 'Scattered'
        if okta <= 7: return 'Broken'
        return 'Overcast'
    out['sky_condition'] = out['cloud_okta'].map(_sky_from_okta)

    lat_deg = out['위도'] if '위도' in out.columns else pd.Series(default_lat, index=out.index)
    lon_deg = out['경도'] if '경도' in out.columns else pd.Series(default_lon, index=out.index)
    lat = np.deg2rad(pd.to_numeric(lat_deg, errors='coerce').fillna(default_lat))
    lon = pd.to_numeric(lon_deg, errors='coerce').fillna(default_lon)

    N = out['day_of_year'].astype(float)
    gamma = 2.0*np.pi*(N - 1 + (hour_f - 12.0)/24.0)/365.0

    EoT = 229.18*(0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                  - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    delta = (0.006918
             - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
             - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
             - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma))

    time_offset = EoT + 4.0*(lon - 15.0*tz_hours)
    TST = (hour_f*60.0 + time_offset) % 1440.0

    H_deg = (TST/4.0) - 180.0
    H = np.deg2rad(H_deg)

    sin_alt = np.sin(lat)*np.sin(delta) + np.cos(lat)*np.cos(delta)*np.cos(H)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    alt_deg = np.rad2deg(np.arcsin(sin_alt))
    out['solar_elevation'] = np.clip(alt_deg, -5.0, 90.0)
    out['solar_rel_pos']   = (out['solar_elevation'].clip(lower=0)/90.0).fillna(0.0)

    cosH0 = (np.cos(np.deg2rad(90.833)) - np.sin(lat)*np.sin(delta)) / (np.cos(lat)*np.cos(delta))
    cosH0 = np.clip(cosH0, -1.0, 1.0)
    H0_deg = np.rad2deg(np.arccos(cosH0))

    sunrise_min = 720.0 - 4.0*(lon + H0_deg) - EoT + 60.0*tz_hours
    sunset_min  = 720.0 - 4.0*(lon - H0_deg) - EoT + 60.0*tz_hours

    out['sunrise_hour'] = np.clip(sunrise_min/60.0, 0.0, 24.0)
    out['sunset_hour']  = np.clip(sunset_min/60.0,  0.0, 24.0)
    out['daylight']     = np.clip(out['sunset_hour'] - out['sunrise_hour'], 0.0, 24.0)

    solar_noon = (out['sunrise_hour'] + out['sunset_hour'])/2.0
    out['정오거리']      = (hour_f - solar_noon).abs()
    out['정오거리_INV'] = 1.0/(1.0 + out['정오거리'])

    I_sc = 1367.0
    E0 = (1.00011 + 0.034221*np.cos(gamma) + 0.00128*np.sin(gamma)
          + 0.000719*np.cos(2*gamma) + 0.000077*np.sin(2*gamma))
    cosZ = np.cos(np.deg2rad(90.0 - out['solar_elevation'].clip(lower=0)))
    out['extraterrestrial_rad'] = (I_sc * E0 * np.maximum(cosZ, 0)).fillna(0.0)

    out['cloud_blocking_factor'] = (out['cloud_cover_combined'] * (1 - out['solar_rel_pos'])).clip(0, 1)

    sort_idx = ['건물번호', '일시'] if '건물번호' in out.columns else ['일시']
    out = out.sort_values(sort_idx)
    grp = out.groupby(out['건물번호']) if '건물번호' in out.columns else [slice(None)]

    dT = grp['기온(°C)'].diff().fillna(0)
    dH = grp['습도(%)'].diff().fillna(0)
    dW = grp['풍속(m/s)'].diff().fillna(0)

    out['기온_급변'] = dT.abs()
    out['습도_급변'] = dH.abs()
    out['풍속_급변'] = dW.abs()

    prev_rain = grp['강수량(mm)'].shift(1).fillna(0)
    now_rain  = out['강수량(mm)']
    out['강수시작'] = ((prev_rain <= 0) & (now_rain > 0)).astype(int)
    out['강수끝']   = ((prev_rain >  0) & (now_rain <= 0)).astype(int)

    bins  = [-1e-9, 0, 1, 5, 10, np.inf]
    labels = [0, 1, 2, 3, 4]
    rain_code = pd.cut(now_rain, bins=bins, labels=labels).astype(int)
    for k in labels:
        out[f'강수강도_{k}'] = (rain_code == k).astype(int)

    window = 6
    def _roll_abs_sum(s): return s.abs().rolling(window, min_periods=1).sum()
    instab = (_roll_abs_sum(dT)/3.0 + _roll_abs_sum(dH)/10.0 + _roll_abs_sum(dW)/2.0
              + out['강수시작']*1.0 + out['강수끝']*1.0)
    out['날씨불안정지수'] = instab.fillna(0)

    out = out.sort_index()
    return out

# ==== 날씨 래그 & 롤링 ====
def _add_weather_lags_rollups(df: pd.DataFrame, lags=(1,3,6), rolls=(3,6,24)) -> pd.DataFrame:
    out = df.sort_values(['건물번호','일시']).copy()
    grp = out.groupby('건물번호', sort=False)

    # Lags
    for col, prefix in [('기온(°C)','T'), ('습도(%)','RH'), ('풍속(m/s)','W')]:
        if col not in out.columns: 
            continue
        x = pd.to_numeric(out[col], errors='coerce')
        for k in lags:
            out[f'{prefix}_lag_{k}'] = grp[x.name].shift(k)

    # Rolling means/stds
    if '기온(°C)' in out.columns:
        T = pd.to_numeric(out['기온(°C)'], errors='coerce')
        for w in rolls:
            out[f'T_rollmean_{w}'] = grp[T.name].apply(lambda s: s.rolling(w, min_periods=1).mean())
        out['T_rollstd_24'] = grp[T.name].apply(lambda s: s.rolling(24, min_periods=2).std())

    # Rain rollups
    if '강수량(mm)' in out.columns:
        R = pd.to_numeric(out['강수량(mm)'], errors='coerce')
        for w in (3,6,24):
            out[f'rain_sum_{w}'] = grp[R.name].apply(lambda s: s.rolling(w, min_periods=1).sum())
        out['rain_last_6h'] = (grp[R.name].apply(lambda s: s.shift(1).rolling(6, min_periods=1).sum()) > 0).astype(int)

    return out

# ==== 비선형 온도 민감도(구간) + 온도 변화 분해 ====
def _add_piecewise_degree_days(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    T = pd.to_numeric(out['기온(°C)'], errors='coerce')
    for b in [18,20,22,24]:
        out[f'CDD_{b}'] = np.maximum(T - b, 0)
    for b in [16,18,20]:
        out[f'HDD_{b}'] = np.maximum(b - T, 0)

    out = out.sort_values(['건물번호','일시'])
    dT = out.groupby('건물번호')['기온(°C)'].diff()
    out['dT_pos'] = dT.clip(lower=0)
    out['dT_neg'] = (-dT).clip(lower=0)
    return out

# ==== 절대습도/수증기압(잠열 지표) ====
def _add_abs_humidity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    T  = pd.to_numeric(out.get('기온(°C)'), errors='coerce')
    RH = pd.to_numeric(out.get('습도(%)'), errors='coerce').clip(0, 100)
    # Magnus (T in °C) → hPa
    es = 6.112 * np.exp((17.67*T)/(T+243.5))
    e  = es * (RH/100.0)  # vapor pressure (hPa)
    # Absolute humidity (g/m^3)
    out['vapor_pressure'] = e
    out['abs_humidity']   = 216.7 * (e / (T + 273.15))
    return out

# ==== 유형×시간/날씨 상호작용 (라벨인코딩 후 호출) ====
def _add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 간단 캘린더 플래그
    out['is_night'] = out['hour'].isin([0,1,2,3,4,5,6]).astype(int)
    out['is_eve_of_holiday'] = out.get('is_next_holiday', 0).astype(int)
    out['is_after_holiday']  = out.get('is_prev_holiday', 0).astype(int)

    # 날씨×운영시간
    if 'CDD_24' in out.columns:
        out['CDD24_x_workhour'] = out['CDD_24'] * out.get('is_workhour', 0)
        out['CDD24_x_coolarea'] = out['CDD_24'] * out.get('냉방면적비', 0)

    # 유형×운영시간/요일 (건물유형은 이미 숫자 인코딩 가정)
    if '건물유형' in out.columns:
        out['type_x_workhour'] = out['건물유형'] * out.get('is_workhour', 0)
        out['type_x_weekday']  = out['건물유형'] * out.get('weekday', 0)

    return out

# ==== 건물 베이스라인(훈련 참조로 만들고 대상에 merge) ====
def _attach_building_baselines(ref_like_train: pd.DataFrame, df_to_attach: pd.DataFrame) -> pd.DataFrame:
    ref = ref_like_train.copy()
    ref['hour']    = ref['일시'].dt.hour
    ref['weekday'] = ref['일시'].dt.weekday

    b_mean = ref.groupby('건물번호')['전력소비량(kWh)'].median().rename('b_mean').reset_index()
    b_h    = ref.groupby(['건물번호','hour'])['전력소비량(kWh)'].median().rename('b_hour_median').reset_index()
    b_wh   = ref.groupby(['건물번호','weekday','hour'])['전력소비량(kWh)'].median().rename('b_wday_hour_median').reset_index()

    out = df_to_attach.copy()
    out['hour']    = out['일시'].dt.hour
    out['weekday'] = out['일시'].dt.weekday

    out = out.merge(b_mean, on='건물번호', how='left') \
             .merge(b_h, on=['건물번호','hour'], how='left') \
             .merge(b_wh, on=['건물번호','weekday','hour'], how='left')
    return out


# ----------------- 결측치 처리 -----------------
def _fill_weather_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['기온(°C)', '습도(%)', '강수량(mm)', '풍속(m/s)']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['강수량(mm)', '풍속(m/s)']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    date_key = df['일시'].dt.date
    orig_index = df.index
    df = df.sort_values(['건물번호', '일시'])

    for col in ['기온(°C)', '습도(%)']:
        if col not in df.columns:
            continue
        s = df.groupby(['건물번호', date_key], sort=False)[col].ffill()
        s = s.where(~s.isna(), df.groupby(['건물번호', date_key], sort=False)[col].bfill())
        df[col] = s
        if df[col].isna().any():
            if 'hour' in df.columns:
                by_hour_med = df.groupby(['건물번호', 'hour'])[col].transform('median')
                df[col] = df[col].fillna(by_hour_med)
            df[col] = df[col].fillna(df[col].median())

    df = df.loc[orig_index]
    return df

def _mean_on(df: pd.DataFrame, group_cols, target_col: str) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, dropna=False)[target_col]
          .mean()
          .reset_index()
          .rename(columns={target_col: 'mean'})
    )
    return agg

# ----------------- 전처리 메인 -----------------
def preprocess_data(train_df, test_df, building_df):
    """공통 전처리 + 파생피처(학습/검증/테스트 동일 적용)"""
    logger.info("데이터 전처리 시작...")

    # ---- 기본 머지/타입 ----
    train_df['일시'] = pd.to_datetime(train_df['일시'])
    test_df['일시'] = pd.to_datetime(test_df['일시'])
    
    train_df = train_df.merge(building_df, on='건물번호', how='left')
    test_df = test_df.merge(building_df, on='건물번호', how='left')
    
        # [B] ★여기★ 건물유형 정규화(반복 토큰 → 1회)
    allowed_types = set(building_df['건물유형'].astype(str).str.strip().unique())
    train_df = _normalize_btype_col(train_df, allowed_types)
    test_df  = _normalize_btype_col(test_df,  allowed_types)

    # [C] is_* 더미는 문자열 상태에서 생성
    for df in (train_df, test_df):
        s = df['건물유형'].astype(str).fillna('')
        df['is_school']   = s.str.contains('학교').astype(int)
        df['is_hospital'] = s.str.contains('병원').astype(int)
        df['is_hotel']    = s.str.contains('호텔').astype(int)

    # [D] 라벨인코딩(이 시점에 실행)
    for col in ['건물유형']:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))

    # ---- 캘린더/결측/건물 스펙 ----
    train_df = _add_time_features(train_df)
    test_df  = _add_time_features(test_df)

    train_df = _fill_weather_missing(train_df)
    test_df  = _fill_weather_missing(test_df)

    train_df = _add_holiday_features(train_df)
    test_df  = _add_holiday_features(test_df)

    train_df = _add_capacity_building_features(train_df)
    test_df  = _add_capacity_building_features(test_df)

    # ---- HDD/CDD & 기상-천문 파생 ----
    train_df = _add_degree_days(train_df)
    test_df  = _add_degree_days(test_df)

    train_df = _add_extra_weather_calendar_features_noaa(train_df)
    test_df  = _add_extra_weather_calendar_features_noaa(test_df)

    # ---- (변경) 래그/롤링: train+test 합쳐서 계산 후 다시 분리 ----
    _all = pd.concat(
        [train_df.assign(__is_train=1), test_df.assign(__is_train=0)],
        axis=0, ignore_index=True
    ).sort_values(['건물번호','일시'])

    _all = _add_weather_lags_rollups(_all, lags=(1,3,6), rolls=(3,6,24))

    train_df = _all[_all['__is_train'] == 1].drop(columns='__is_train').copy()
    test_df  = _all[_all['__is_train'] == 0].drop(columns='__is_train').copy()

    # ---- 비선형 온도 민감도 & 절대습도 ----
    train_df = _add_piecewise_degree_days(train_df)
    test_df  = _add_piecewise_degree_days(test_df)

    train_df = _add_abs_humidity(train_df)
    test_df  = _add_abs_humidity(test_df)

    train_df = _add_interactions(train_df)
    test_df  = _add_interactions(test_df)

    # ---- 제출 키 ----
    for df in (train_df, test_df):
        df['num_date_time'] = (
            df['건물번호'].astype(int).astype(str) + '_' +
            df['일시'].dt.strftime('%Y%m%d %H')
        )

    logger.info("데이터 전처리 완료")
    return train_df, test_df


# ----------------- 피처 중요도 계산 및 관리 -----------------
class FeatureImportanceManager:
    def __init__(self):
        self.feature_importances = {}
        self.selected_features = None
        self.importance_scores = {}

    def calculate_feature_importance(self, X_train, y_train, feature_names, method='xgb'):
        importances = {}
        if method in ['xgb', 'combined']:
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=800, 
                    random_state=Config.RANDOM_STATE,
                    n_jobs=-1,
                    eval_metric="rmse",
                    tree_method="gpu_hist",
                    predictor="gpu_predictor"
                )
                xgb_model.fit(X_train, y_train, verbose=False)
                importances['xgb'] = xgb_model.feature_importances_
            except Exception as e:
                logger.warning(f"XGBoost 중요도 계산 실패: {e}")
                importances['xgb'] = np.ones(len(feature_names)) / len(feature_names)
        if method in ['lgb', 'combined']:
            try:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=800,
                    random_state=Config.RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                    eval_metric="rmse",
                    device='gpu',
                )
                lgb_model.fit(X_train, y_train)
                importances['lgb'] = lgb_model.feature_importances_
            except Exception as e:
                logger.warning(f"LightGBM 중요도 계산 실패: {e}")
                importances['lgb'] = np.ones(len(feature_names)) / len(feature_names)
        if method in ['et', 'combined']:
            try:
                et_model = ExtraTreesRegressor(
                    n_estimators=800,
                    random_state=Config.RANDOM_STATE,
                    n_jobs=-1
                )
                et_model.fit(X_train, y_train)
                importances['et'] = et_model.feature_importances_
            except Exception as e:
                logger.warning(f"ExtraTreesRegressor 중요도 계산 실패: {e}")
                importances['et'] = np.ones(len(feature_names)) / len(feature_names)

        if method == 'combined':
            combined_importance = np.mean(list(importances.values()), axis=0)
        else:
            combined_importance = importances[method]

        feature_importance_dict = dict(zip(feature_names, combined_importance))
        return feature_importance_dict

    def select_features_by_importance(self, feature_importance_dict, threshold=None, top_k=None):
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        selected = []
        if top_k:
            selected = [feat for feat, imp in sorted_features[:top_k]]
        elif threshold:
            selected = [feat for feat, imp in sorted_features if imp >= threshold]
        else:
            selected = [feat for feat, imp in sorted_features]
        logger.info(f"피처 선택 완료: {len(selected)}개 피처 선택 (전체 {len(feature_importance_dict)}개 중)")
        return selected

    # === NEW: 중요도를 per-feature 가중치로 변환 ===
    def build_weights(self, features, importance_dict,
                      min_w: float = 0.6, gamma: float = 1.2) -> np.ndarray:
        """
        features 순서에 맞는 가중치 벡터를 만든다.
        중요도 -> [0,1] 정규화 -> (·)^gamma -> min_w..1 선형 스케일
        """
        imps = np.array([importance_dict.get(f, 0.0) for f in features], dtype=float)
        if not np.isfinite(imps).any() or np.allclose(imps, 0):
            return np.ones(len(features), dtype=float)
        imp_min, imp_max = float(np.min(imps)), float(np.max(imps))
        if imp_max - imp_min < 1e-12:
            z = np.ones_like(imps)
        else:
            z = (imps - imp_min) / (imp_max - imp_min)
        z = np.power(z, gamma)
        weights = min_w + (1.0 - min_w) * z
        return weights.astype(float)

# ----------------- 앙상블 가중치 최적화 -----------------
class EnsembleWeightOptimizer:
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.best_weights = None

    def optimize_weights_scipy(self, predictions_dict, y_true):
        def objective(weights):
            weights = weights / np.sum(weights)
            ensemble_pred = np.zeros_like(y_true, dtype=float)
            for i, (_, pred) in enumerate(predictions_dict.items()):
                ensemble_pred += weights[i] * pred
            return smape(y_true, ensemble_pred)

        n_models = len(predictions_dict)
        initial_weights = np.ones(n_models) / n_models
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            self.best_weights = result.x
            model_names = list(predictions_dict.keys())
            return dict(zip(model_names, result.x))
        else:
            model_names = list(predictions_dict.keys())
            equal_weights = np.ones(n_models) / n_models
            return dict(zip(model_names, equal_weights))

# ----------------- 앙상블 모델 -----------------
class OptimizedEnsembleModel:
    def __init__(self):
        base_core = [
            # 날씨
            '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
            # 건물/용량
            '연면적(m2)', '냉방면적(m2)', '건물유형', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
            'has_pv', 'has_ess', 'has_pcs', '냉방면적비', 'cap_per_area_log1p',
            # 시간/캘린더
            'year','month','day','hour','weekday','is_weekend','is_workingday',
            'is_holiday','is_prev_holiday','is_next_holiday',
            'is_workhour','is_peak',
            # 주기형
            'hour_sin','hour_cos','month_sin','month_cos','day_sin','day_cos',
            # HDD/CDD
            'HDD','CDD'
        ]

        base_extra = [
            'COS_월','SIN_요일','COS_요일','정오거리','정오거리_INV','peak_time',
            '이슬점온도','이슬점차이','체감온도','불쾌지수','perceived_temperature',
            'cloudy_based_on_humidity','day_of_year','SIN_day_of_year','COS_day_of_year',
            'sunrise_hour','sunset_hour','daylight','solar_elevation','solar_rel_pos','extraterrestrial_rad',
            'cloud_cover_humidity','cloud_cover_dewpoint','cloud_cover_rain',
            'cloud_cover_combined','cloud_cover_weighted',
            'low_cloud_prob','mid_cloud_prob','high_cloud_prob','cumulonimbus_prob',
            'cloud_okta','cloud_blocking_factor',
            '기온_급변','습도_급변','풍속_급변','강수시작','강수끝',
            '강수강도_0','강수강도_1','강수강도_2','강수강도_3','강수강도_4',
            '날씨불안정지수', 'type_night_share','type_weekend_share'
        ]

        # === PATCH: base_core / base_extra 에 누락 파생피처 합치기 ===
        # 1) CDD/HDD 세분화 & 온도 변화 분해
        base_core += [f'CDD_{b}' for b in (18,20,22,24)]
        base_core += [f'HDD_{b}' for b in (16,18,20)]
        base_core += ['dT_pos', 'dT_neg']

        # 2) 절대습도/수증기압
        base_extra += ['vapor_pressure', 'abs_humidity']

        # 3) 날씨 래그/롤업
        base_extra += [f'T_lag_{k}'  for k in (1,3,6)]
        base_extra += [f'RH_lag_{k}' for k in (1,3,6)]
        base_extra += [f'W_lag_{k}'  for k in (1,3,6)]
        base_extra += [f'T_rollmean_{w}' for w in (3,6,24)] + ['T_rollstd_24']
        base_extra += [f'rain_sum_{w}' for w in (3,6,24)] + ['rain_last_6h']

        # 4) 유형 피벗 평균(시간/요일/월)
        base_extra += [f'type_hour_mean_{h}'  for h in range(24)]
        base_extra += [f'type_wday_mean_{d}'  for d in range(7)]
        base_extra += [f'type_month_mean_{m}' for m in range(1,13)]

        # 5) 운영/상호작용
        base_extra += [
            'is_night', 'is_eve_of_holiday', 'is_after_holiday',
            'CDD24_x_workhour', 'CDD24_x_coolarea',
            'type_x_workhour', 'type_x_weekday'
        ]

        # 6) 유형 정적/집중도(혹시 누락 대비 재확인 차원)
        base_core += [
            'type_kwh_per_area_mean','type_workhour_ratio','type_holiday_ratio',
            'type_temp_beta','type_humidity_beta',
            'type_avg_cap_per_area_log1p','type_avg_cool_ratio','type_avg_has_ess',
            'type_night_share','type_weekend_share','type_night_ratio','type_weekend_ratio'
        ]

        # 7) 빌딩 베이스라인/집중도(혹시 누락 대비 재확인 차원)
        base_core += [
            'b_mean','b_hour_median','b_wday_hour_median',
            'b_night_share','b_weekend_share','b_workhour_share',
            'b_workhour_ratio','b_night_ratio','b_weekend_ratio'
        ]

        # 중복 제거(순서 보존)
        def _dedupe(seq): return list(dict.fromkeys(seq))
        base_core  = _dedupe(base_core)
        base_extra = _dedupe(base_extra)

        self.base_features = base_core + base_extra
        self.features = self.base_features.copy()
        
        self.best_params = {}
        self.models = {}
        self.ensemble_weights = Config.ENSEMBLE_WEIGHTS.copy()
        self.weight_optimizer = EnsembleWeightOptimizer()
        
        self.feature_manager = FeatureImportanceManager()
        self.feature_weights = None   # NEW: per-feature weight vector
        self.global_best_params = None


        self.augment_cont_cols = [
            '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
            '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)',
            '냉방면적비', 'cap_per_area_log1p',
            'hour_sin','hour_cos','month_sin','month_cos','day_sin','day_cos',
            'HDD','CDD',
            '정오거리','정오거리_INV',
            '이슬점온도','이슬점차이','체감온도','불쾌지수','perceived_temperature',
            'day_of_year','SIN_day_of_year','COS_day_of_year',
            'sunrise_hour','sunset_hour','daylight','solar_elevation','solar_rel_pos','extraterrestrial_rad',
            'cloud_cover_humidity','cloud_cover_dewpoint','cloud_cover_rain',
            'cloud_cover_combined','cloud_cover_weighted',
            'low_cloud_prob','mid_cloud_prob','high_cloud_prob','cumulonimbus_prob',
            'cloud_okta','cloud_blocking_factor'
        ]

    def analyze_feature_importance(self, train_df, method='combined'):
        logger.info("피처 중요도 분석 시작...")

        # 새 파생 접두어(동적 포함)
        dyn_prefixes = (
            'type_', 'CDD_', 'HDD_', 'T_lag_', 'RH_lag_', 'W_lag_',
            'T_rollmean_', 'T_rollstd_', 'rain_sum_', 'rain_last_',
            'b_', 'dT_pos', 'dT_neg'
        )
        dynamic_cols = [c for c in train_df.columns if c.startswith(dyn_prefixes)]
        for extra in (
            'abs_humidity', 'vapor_pressure',
            'CDD24_x_workhour', 'CDD24_x_coolarea',
            'type_x_workhour', 'type_x_weekday',
            'is_night', 'is_eve_of_holiday', 'is_after_holiday'
        ):
            if extra in train_df.columns:
                dynamic_cols.append(extra)

        # (기존) base + 동적 후보
        available_features = list({*self.base_features, *dynamic_cols})
        available_features = [f for f in available_features if f in train_df.columns]

        # === [ADD] 중요도 계산은 '숫자형'만 사용 ===
        import pandas as pd
        non_numeric = [c for c in available_features if not pd.api.types.is_numeric_dtype(train_df[c])]
        if non_numeric:
            logger.warning(f"[FI] Dropping non-numeric features from importance calc: {non_numeric[:10]}... "
                           f"total={len(non_numeric)}")

        numeric_cols = [c for c in available_features if pd.api.types.is_numeric_dtype(train_df[c])]

        X = train_df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        y = pd.to_numeric(train_df['전력소비량(kWh)'], errors='coerce').fillna(0).values

        feature_importance_dict = self.feature_manager.calculate_feature_importance(
            X, y, numeric_cols, method=method
        )

        if Config.SAVE_FI_PLOT:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                plt.figure(figsize=(12, 8))
                top = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
                ft, im = zip(*top) if top else ([], [])
                sns.barplot(x=list(im), y=list(ft))
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig(os.path.join(path, 'feature_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"피처 중요도 플롯 스킵: {e}")

        if Config.ENABLE_FEATURE_SELECTION:
            selected_features = self.feature_manager.select_features_by_importance(
                feature_importance_dict,
                threshold=Config.IMPORTANCE_THRESHOLD,
                top_k=Config.TOP_K_FEATURES
            )
            self.features = selected_features
            logger.info(f"[선택모드] 중요도 기반 피처 선택: {len(selected_features)}개")
        else:
            # === [CHANGE] 숫자형만 유지 ===
            self.features = numeric_cols
            logger.info(f"[가중치모드] 모든 숫자형 피처 사용: {len(numeric_cols)}개")

        if Config.IMPORTANCE_WEIGHTING:
            self.feature_weights = self.feature_manager.build_weights(
                self.features,
                feature_importance_dict,
                min_w=Config.FEATURE_WEIGHT_MIN,
                gamma=Config.FEATURE_WEIGHT_GAMMA
            )
        else:
            self.feature_weights = np.ones(len(self.features), dtype=float)

        self.feature_manager.importance_scores = feature_importance_dict
        self.feature_manager.selected_features = self.features
        return feature_importance_dict

    def get_feature_importance_summary(self):
        if not self.feature_manager.importance_scores:
            return "피처 중요도가 계산되지 않았습니다."
        sorted_features = sorted(
            self.feature_manager.importance_scores.items(),
            key=lambda x: x[1], reverse=True
        )[:10]
        summary = "=== Top 10 피처 중요도 ===\n"
        for i, (feature, importance) in enumerate(sorted_features, 1):
            summary += f"{i:2d}. {feature:20s}: {importance:.4f}\n"
        summary += f"\n총 사용 피처 수: {len(self.features)}"
        summary += f"\n가중치 적용 여부: {Config.IMPORTANCE_WEIGHTING}"
        return summary

    def _augment_train_df(self, df: pd.DataFrame, n_copies=1, noise_std=0.02) -> pd.DataFrame:
        if not Config.USE_AUGMENT or n_copies <= 0:
            return df
        base = df.copy()
        cont_cols = [c for c in self.augment_cont_cols if c in base.columns and c in self.features]
        if not cont_cols:
            return base
        std = base[cont_cols].std().replace(0, 1.0)
        aug_list = [base]
        for _ in range(n_copies):
            aug = base.copy()
            noise = np.random.normal(0.0, 1.0, size=aug[cont_cols].shape)
            aug[cont_cols] = aug[cont_cols] + noise * (std.values * noise_std)
            if '강수량(mm)' in aug.columns: aug['강수량(mm)'] = aug['강수량(mm)'].clip(lower=0)
            if '풍속(m/s)' in aug.columns:  aug['풍속(m/s)'] = aug['풍속(m/s)'].clip(lower=0)
            if '습도(%)' in aug.columns:    aug['습도(%)'] = aug['습도(%)'].clip(0, 100)
            for scc in ['hour_sin','hour_cos','month_sin','month_cos','day_sin','day_cos']:
                if scc in aug.columns:
                    aug[scc] = aug[scc].clip(-1, 1)
            if '냉방면적비' in aug.columns:  aug['냉방면적비'] = aug['냉방면적비'].clip(lower=0)
            aug_list.append(aug)
        out = pd.concat(aug_list, axis=0, ignore_index=True)
        return out

    # --------- 하이퍼파라미터 최적화(옵투나) ---------
    # (가중치는 최종 학습에 적용; 탐색 단계에서는 기본 데이터로 운영해도 OK)
    def optimize_xgb_params(self, X_train, y_train, X_val, y_val):
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': trial.suggest_float('learning_rate', 0.004, 0.05),
                'max_depth': trial.suggest_int('max_depth', 2, 7),
                'min_child_weight': trial.suggest_int('min_child_weight', 7, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'n_estimators': trial.suggest_int('n_estimators', 400, 1200),
                'random_state': Config.RANDOM_STATE,
                'n_jobs': -1
            }
            model = xgb.XGBRegressor(**params, tree_method="gpu_hist", predictor="gpu_predictor")
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      eval_metric='rmse', verbose=False, early_stopping_rounds=20)
            y_pred = model.predict(X_val)
            return smape(y_val, y_pred)

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=Config.OPTUNA_SEED)
        )
        study.optimize(objective, n_trials=Config.OPTUNA_TRIALS)
        return study.best_params

    def optimize_lgb_params(self, X_train, y_train, X_val, y_val):
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.004, 0.05),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'n_estimators': trial.suggest_int('n_estimators', 400, 1200),
                'random_state': Config.RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            }
            params['device'] = 'gpu'
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse',
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            y_pred = model.predict(X_val)
            return smape(y_val, y_pred)

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=Config.OPTUNA_SEED)
        )
        study.optimize(objective, n_trials=Config.OPTUNA_TRIALS)
        return study.best_params

    def optimize_cat_params(self, X_train, y_train, X_val, y_val):
        def objective(trial):
            params = {
                'loss_function': 'RMSE',
                'learning_rate': trial.suggest_float('learning_rate', 0.004, 0.05),
                'depth': trial.suggest_int('depth', 4, 12),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'random_seed': Config.RANDOM_STATE,
                'thread_count': -1,
                'verbose': False
            }
            params['task_type'] = 'GPU'
            params['devices'] = '0'
            model = cb.CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      early_stopping_rounds=20, verbose=False)
            y_pred = model.predict(X_val)
            return smape(y_val, y_pred)

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=Config.OPTUNA_SEED)
        )
        study.optimize(objective, n_trials=Config.OPTUNA_TRIALS)
        return study.best_params

    # --------- 내부 유틸: 가중치 적용 ---------
    def _apply_fw(self, X: np.ndarray, model_key: str) -> np.ndarray:
        """LightGBM/CatBoost용: 입력 특성에 가중치 스케일을 곱함"""
        if (not Config.IMPORTANCE_WEIGHTING) or (self.feature_weights is None):
            return X
        if model_key not in Config.APPLY_FW or not Config.APPLY_FW[model_key]:
            return X
        # XGBoost는 DMatrix.feature_weights로 처리하므로 여기선 lgb/cat만 스케일링
        if model_key in ('lgb', 'cat'):
            return X * self.feature_weights
        return X

    # --------- 건물 단위 학습/예측 ---------
    def train_building_ensemble(self, building_train, building_pred, building_num,
                                has_label_in_pred=False, optimize_params=True):
        if building_train.empty or building_pred.empty:
            return np.full(len(building_pred), np.nan)

        # 검증 분할
        if has_label_in_pred:
            train_df_use = building_train.copy()
            val_df_use   = building_pred.copy()
        else:
            df_sorted = building_train.sort_values('일시')
            n = len(df_sorted)
            val_size = max(Config.MIN_VAL_HOURS, int(n * Config.INNER_VAL_RATIO))
            split = max(1, n - val_size)
            train_df_use = df_sorted.iloc[:split].copy()
            val_df_use   = df_sorted.iloc[split:].copy()

        # 학습 데이터 증강
        if Config.USE_AUGMENT:
            train_df_aug = self._augment_train_df(
                train_df_use,
                n_copies=Config.AUG_N_COPIES,
                noise_std=Config.AUG_NOISE_STD
            )
        else:
            train_df_aug = train_df_use

        X_train = train_df_aug[self.features].fillna(0).values
        y_train = train_df_aug['전력소비량(kWh)'].values
        X_val   = val_df_use[self.features].fillna(0).values
        y_val   = val_df_use['전력소비량(kWh)'].values
        X_pred  = building_pred[self.features].fillna(0).values

        X_train, y_train = _clean_xy(X_train, y_train, 'train')
        X_val,   y_val   = _clean_xy(X_val,   y_val,   'val')

        if y_train.size == 0:
            logger.error(f"[b{building_num}] no train after cleaning → returning NaN (no median fallback)")
            return np.full(len(X_pred), np.nan, dtype=float)

        no_val_mode = (y_val.size == 0)

        # 하이퍼파라미터 최적화(처음 1회)
        def _clean_xy(X, y, tag):
            y = np.asarray(y, dtype=np.float64).ravel()
            m = np.isfinite(y)
            if m.sum() < len(y):
                logger.warning(f"[{tag}][b{building_num}] drop {len(y)-m.sum()} rows with non-finite target")
            X, y = X[m], y[m]
            return X, y

        # 중앙값/패턴 반환 금지: 검증 재구성 or 검증 없이 학습
        no_val_mode = False

        if (y_train.size == 0) or (y_val.size == 0):
            logger.warning(f"[b{building_num}] empty train/val after cleaning → no-median fallback")

            # 1) 건물 학습 데이터에서 유효 타깃만 모아 '작은 검증셋' 재구성 시도
            df_nn = building_train.copy()
            df_nn['__y__'] = pd.to_numeric(df_nn['전력소비량(kWh)'], errors='coerce')
            df_nn = df_nn[np.isfinite(df_nn['__y__'])].drop(columns='__y__', errors='ignore')
            df_nn = df_nn.sort_values('일시')

            if len(df_nn) >= 48:
                # 뒤에서 최소 24~168시간을 검증으로 떼기 (데이터 양에 따라 자동)
                val_take = min(168, max(24, int(len(df_nn) * 0.2)))
                train_df_use = df_nn.iloc[:-val_take].copy()
                val_df_use   = df_nn.iloc[-val_take:].copy()

                X_train = train_df_use[self.features].fillna(0).values
                y_train = pd.to_numeric(train_df_use['전력소비량(kWh)'], errors='coerce').values
                X_val   = val_df_use[self.features].fillna(0).values
                y_val   = pd.to_numeric(val_df_use['전력소비량(kWh)'], errors='coerce').values
                no_val_mode = False
            else:
                # 2) 검증 없이 학습(얼리스탑/검증셋 없이 fit)
                no_val_mode = True


        predictions = {}

        # ---------- XGBoost (진짜 feature_weights 지원) ----------
        try:
            default_xgb = {
                'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 1200,
                'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0,
                'reg_alpha': 0.0, 'reg_lambda': 1.0,
                'random_state': Config.TRAIN_SEED_BASE, 'n_jobs': -1
            }

            bsp   = self.best_params.get(building_num, {}) if isinstance(self.best_params, dict) else {}
            gbest = getattr(self, 'global_best_params', None)
            base_params = (
                bsp.get('xgb') or
                (gbest.get('xgb') if gbest else None) or
                default_xgb
            )
            xgb_preds = []
            for r in range(Config.TRAIN_REPEATS):
                # DMatrix에 feature_weights 적용
                if Config.IMPORTANCE_WEIGHTING and Config.APPLY_FW.get('xgb', True) and (self.feature_weights is not None):
                    dtrain = xgb.DMatrix(X_train, label=y_train,
                                         feature_names=self.features,
                                         feature_weights=self.feature_weights)
                    dval   = xgb.DMatrix(X_val,   label=y_val,
                                         feature_names=self.features,
                                         feature_weights=self.feature_weights)
                    dpred  = xgb.DMatrix(X_pred,  feature_names=self.features,
                                         feature_weights=self.feature_weights)
                else:
                    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.features)
                    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=self.features)
                    dpred  = xgb.DMatrix(X_pred,  feature_names=self.features)

                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'learning_rate': base_params.get('learning_rate', 0.01),
                    'max_depth': base_params.get('max_depth', 6),
                    'min_child_weight': base_params.get('min_child_weight', 1),
                    'subsample': base_params.get('subsample', 1.0),
                    'colsample_bytree': base_params.get('colsample_bytree', 1.0),
                    'reg_alpha': base_params.get('reg_alpha', 0.0),
                    'reg_lambda': base_params.get('reg_lambda', 1.0),
                    'seed': Config.TRAIN_SEED_BASE + r,
                    'nthread': base_params.get('n_jobs', -1)
                }
                num_boost_round = int(base_params.get('n_estimators', 1200))

                if no_val_mode:
                    bst = xgb.train(
                        params, dtrain, num_boost_round=num_boost_round,
                        evals=[], verbose_eval=False
                    )
                else:
                    bst = xgb.train(
                        params, dtrain, num_boost_round=num_boost_round,
                        evals=[(dval, 'valid')], early_stopping_rounds=17, verbose_eval=False
                    )

                # 예측
                if hasattr(bst, 'best_ntree_limit') and bst.best_ntree_limit is not None:
                    pred = bst.predict(dpred, ntree_limit=bst.best_ntree_limit)
                else:
                    pred = bst.predict(dpred)
                xgb_preds.append(pred)

            predictions['xgb'] = np.mean(np.vstack(xgb_preds), axis=0)
        except Exception as e:
            logger.warning(f"xgb 훈련 실패: {e}")

        # ---------- LightGBM (입력 스케일링으로 근사) ----------
        try:
            default_lgb = {
                'learning_rate': 0.01, 'num_leaves': 31, 'n_estimators': 1200,
                'feature_fraction': 1.0, 'bagging_fraction': 1.0, 'bagging_freq': 0,
                'min_child_samples': 20, 'reg_alpha': 0.0, 'reg_lambda': 0.0,
                'random_state': Config.TRAIN_SEED_BASE, 'n_jobs': -1, 'verbose': -1
            }

            bsp   = self.best_params.get(building_num, {}) if isinstance(self.best_params, dict) else {}
            gbest = getattr(self, 'global_best_params', None)
            base_params = (
                bsp.get('lgb') or
                (gbest.get('lgb') if gbest else None) or
                default_lgb
            )
            lgb_preds = []
            for r in range(Config.TRAIN_REPEATS):
                lgb_params = base_params.copy()
                lgb_params['random_state'] = Config.TRAIN_SEED_BASE + r

                Xtr = self._apply_fw(X_train, 'lgb')
                Xva = self._apply_fw(X_val,   'lgb')
                Xte = self._apply_fw(X_pred,  'lgb')

                lgb_params['metric'] = 'None' 
                lgb_params['device'] = 'gpu'
                lgb_model = lgb.LGBMRegressor(**lgb_params)
                if no_val_mode:
                    lgb_model.fit(Xtr, y_train)
                else:
                    lgb_model.fit(
                        Xtr, y_train,
                        eval_set=[(Xva, y_val)], eval_metric="rmse",
                        callbacks=[lgb.early_stopping(17), lgb.log_evaluation(0)]
                    )


            predictions['lgb'] = np.mean(np.vstack(lgb_preds), axis=0)
        except Exception as e:
            logger.warning(f"LightGBM 훈련 실패: {e}")

        # ---------- CatBoost (GPU + SMAPE eval + best model) ----------
        try:
            default_cat = {
                'learning_rate': 0.01,
                'depth': 6,
                'iterations': 1000,
                'l2_leaf_reg': 3,
                'border_count': 128,
                'random_seed': Config.TRAIN_SEED_BASE,
                'thread_count': -1,
                'verbose': False,
                'loss_function': 'RMSE',   # 학습 안정성은 RMSE로
                'task_type': 'GPU',        # GPU 고정
                'devices': '0',            # 사용 GPU 지정
                'eval_metric': 'SMAPE',    # 검증/얼리스탑은 sMAPE 기준
            }

            bsp   = self.best_params.get(building_num, {}) if isinstance(self.best_params, dict) else {}
            gbest = getattr(self, 'global_best_params', None)
            base_params = (
                bsp.get('cat') or
                (gbest.get('cat') if gbest else None) or
                default_cat
            )

            cat_preds = []
            for r in range(Config.TRAIN_REPEATS):
                cat_params = base_params.copy()
                cat_params['random_seed'] = Config.TRAIN_SEED_BASE + r
                # 안전: GPU/지표 보장 (혹시 상위에서 빠졌을 경우)
                cat_params.setdefault('task_type', 'GPU')
                cat_params.setdefault('devices', '0')
                cat_params.setdefault('loss_function', 'RMSE')
                cat_params['eval_metric'] = 'SMAPE'

                Xtr = self._apply_fw(X_train, 'cat')
                Xva = self._apply_fw(X_val,   'cat')
                Xte = self._apply_fw(X_pred,  'cat')

                cat_model = cb.CatBoostRegressor(**cat_params)
                if no_val_mode:
                    cat_params['use_best_model'] = False
                    cat_model = cb.CatBoostRegressor(**cat_params)
                    cat_model.fit(Xtr, y_train, verbose=False)
                else:
                    cat_params['use_best_model'] = True
                    cat_model = cb.CatBoostRegressor(**cat_params)
                    cat_model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], early_stopping_rounds=17, verbose=False)



                # use_best_model=True 이면 predict가 자동으로 best-iter를 사용
                cat_preds.append(cat_model.predict(Xte))

            predictions['cat'] = np.mean(np.vstack(cat_preds), axis=0)

        except Exception as e:
            logger.warning(f"CatBoost 훈련 실패: {e}")

        # --- NEW: 안전한 앙상블 결합 ---
        present = {
            k: v for k, v in predictions.items()
            if isinstance(v, np.ndarray) and v.shape[0] == len(X_pred) and np.all(np.isfinite(v))
        }
        if not present:
            logger.warning(f"[b{building_num}] all base models failed; using baseline.")
            baseline = np.nanmedian(pd.to_numeric(building_train['전력소비량(kWh)'], errors='coerce').to_numpy())
            if not np.isfinite(baseline): baseline = 0.0
            return np.full(len(X_pred), float(baseline))

        weights = {k: float(self.ensemble_weights.get(k, 0.0)) for k in present.keys()}
        wsum = sum(weights.values())
        if wsum <= 0:
            # 동일 가중치
            w = {k: 1.0/len(present) for k in present.keys()}
        else:
            w = {k: weights[k]/wsum for k in present.keys()}

        ensemble_pred = np.zeros(len(X_pred), dtype=np.float64)
        for k, pred in present.items():
            ensemble_pred += w[k] * pred
        return ensemble_pred


    def optimize_ensemble_weights(self, train_df, optimize_method='scipy'):
        logger.info(f"앙상블 가중치 최적화 시작 (방법: {optimize_method})...")
        try:
            kfold = KFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
            X = train_df[self.features].fillna(0).values
            y = train_df['전력소비량(kWh)'].values
            preds_dict = {'xgb': [], 'lgb': [], 'cat': []}
            y_all = []

            for tr_idx, va_idx in kfold.split(X):
                X_tr, X_va = X[tr_idx], X[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                # 간단 버전(가중치 미적용) — 빠르게 앙상블 비중만 조정
                try:
                    m_xgb = xgb.XGBRegressor(
                        n_estimators=800,
                        random_state=Config.RANDOM_STATE,
                        n_jobs=-1,
                        eval_metric="rmse",
                        tree_method="gpu_hist",
                        predictor="gpu_predictor"
                    )
                    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                              verbose=False, early_stopping_rounds=20)
                    preds_dict['xgb'].extend(m_xgb.predict(X_va))
                except:
                    preds_dict['xgb'].extend([y_tr.mean()] * len(va_idx))

                try:
                    m_lgb = lgb.LGBMRegressor(
                        n_estimators=800, random_state=Config.RANDOM_STATE, n_jobs=-1, verbose=-1, eval_metric="rmse",
                    )
                    m_lgb.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
                    )
                    preds_dict['lgb'].extend(m_lgb.predict(X_va))
                except:
                    preds_dict['lgb'].extend([y_tr.mean()] * len(va_idx))

                try:
                    m_cat = cb.CatBoostRegressor(
                        iterations=600, random_seed=Config.RANDOM_STATE, thread_count=-1, verbose=False
                    )
                    m_cat.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        early_stopping_rounds=20,
                        verbose=False
                    )
                    preds_dict['cat'].extend(m_cat.predict(X_va))
                except:
                    preds_dict['cat'].extend([y_tr.mean()] * len(va_idx))

                y_all.extend(y_va)

            preds_dict = {k: np.array(v) for k, v in preds_dict.items()}
            y_all = np.array(y_all)

            opt = EnsembleWeightOptimizer(n_folds=Config.CV_FOLDS, random_state=Config.RANDOM_STATE)
            optimal_weights = opt.optimize_weights_scipy(preds_dict, y_all)
            self.ensemble_weights.update(optimal_weights)
            logger.info(f"최적화된 앙상블 가중치: {optimal_weights}")
            return optimal_weights
        except Exception as e:
            logger.warning(f"가중치 최적화 실패: {e}. 기본 가중치 사용")
            return self.ensemble_weights

    def train_and_predict(self, train_df, pred_df, optimize_params=True, optimize_weights=True):
        """전체 앙상블 모델 훈련 및 예측 (중요도 가중치 포함)"""
        if Config.USE_FEATURE_IMPORTANCE:
            self.analyze_feature_importance(train_df, method='combined')
        
        if optimize_weights and not ('전력소비량(kWh)' in pred_df.columns):
            self.optimize_ensemble_weights(train_df, optimize_method='scipy')

        preds = np.full(len(pred_df), np.nan)
        has_label_in_pred = ('전력소비량(kWh)' in pred_df.columns)

        building_numbers = train_df['건물번호'].unique()
        for building_num in tqdm(building_numbers, desc="앙상블 모델 훈련"):
            train_building = train_df[train_df['건물번호'] == building_num].copy()
            pred_mask = (pred_df['건물번호'] == building_num)
            pred_building = pred_df.loc[pred_mask].copy()
            try:
                building_preds = self.train_building_ensemble(
                    train_building, pred_building, building_num,
                    has_label_in_pred, optimize_params
                )
                preds[pred_mask.to_numpy()] = building_preds
            except Exception as e:
                logger.error(f"건물 {building_num} 처리 중 오류: {e}")
                global_mean = train_df['전력소비량(kWh)'].mean()
                preds[pred_mask.to_numpy()] = global_mean

        if np.isnan(preds).any():
            global_mean = float(train_df['전력소비량(kWh)'].mean())
            preds = np.where(np.isnan(preds), global_mean, preds)
        return preds

    def load_params(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            if isinstance(save_data, dict):
                self.best_params = save_data.get('best_params', {})
                self.global_best_params = save_data.get('global_best_params', None)  # <<< 추가
                if 'ensemble_weights' in save_data:
                    self.ensemble_weights = save_data['ensemble_weights']
                    logger.info(f"로드된 앙상블 가중치: {self.ensemble_weights}")
                if 'selected_features' in save_data:
                    self.features = save_data['selected_features']
                    logger.info(f"로드된 선택 피처 수: {len(self.features)}")
                if 'feature_importances' in save_data:
                    self.feature_manager.importance_scores = save_data['feature_importances']
                    logger.info("피처 중요도 정보 로드 완료")
            else:
                self.best_params = save_data
            logger.info(f"파라미터 로드 완료: {filepath}")
            return True
        except FileNotFoundError:
            logger.info("저장된 파라미터가 없습니다. 새로 최적화를 진행합니다.")
            return False

    def save_params(self, filepath):
        """파라미터와 기타 정보를 저장"""
        try:
            save_data = {
                'best_params': self.best_params,
                'global_best_params': getattr(self, 'global_best_params', None),
                'ensemble_weights': self.ensemble_weights,
                'selected_features': self.features,
                'feature_importances': getattr(self.feature_manager, 'importance_scores', {})
            }
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            logger.info(f"파라미터 저장 완료: {filepath}")
        except Exception as e:
            logger.error(f"파라미터 저장 실패: {e}")

# ----------------- 메인 -----------------
def main():
    logger.info("피처 중요도 기반 앙상블 모델 훈련 시작...")

    # 1) 로드
    train_df, test_df, building_df = load_data()

    # 2) 전처리(공통) + 파생
    train_df_processed, test_df_processed = preprocess_data(train_df, test_df, building_df)

    # 3) 모델
    ensemble_model = OptimizedEnsembleModel()

    # 저장된 파라미터 로드 시도
    param_path = os.path.join(path, f'best_params_with_features_{timestamp}.pkl')
    use_saved_params = ensemble_model.load_params(param_path)

    # 4) 홀드아웃 검증
    logger.info("검증 데이터셋으로 모델 성능 평가 시작...")
    
    train_df_processed['전력소비량(kWh)'] = pd.to_numeric(train_df_processed['전력소비량(kWh)'], errors='coerce')
    
    train_val = train_df_processed[train_df_processed['일시'] < Config.VAL_DATE].copy()
    val_df    = train_df_processed[train_df_processed['일시'] >= Config.VAL_DATE].copy()
    # 타깃 결측 보강 (split 별 ref)
    train_val = _impute_target_timewise(train_val, ref_df=train_val, target_col='전력소비량(kWh)')
    val_df    = _impute_target_timewise(val_df,    ref_df=train_val, target_col='전력소비량(kWh)')


    # (추가) 베이스라인 피처 부착
    train_val, val_df = add_building_type_features(train_val, val_df, ref_df=train_val)
    train_val = _attach_building_baselines(train_val, train_val)
    val_df    = _attach_building_baselines(train_val, val_df)

    if not use_saved_params:
        # 중요도/피처 확정 (전역 파라미터 탐색에 사용할 피처셋)
        ensemble_model.analyze_feature_importance(train_val, method='combined')
        X_tr = train_val[ensemble_model.features].fillna(0).values
        y_tr = train_val['전력소비량(kWh)'].values
        X_va = val_df[ensemble_model.features].fillna(0).values
        y_va = val_df['전력소비량(kWh)'].values

        logger.info("전역 하이퍼파라미터 최적화 시작...")
        try:
            global_best = {
                'xgb': ensemble_model.optimize_xgb_params(X_tr, y_tr, X_va, y_va),
                'lgb': ensemble_model.optimize_lgb_params(X_tr, y_tr, X_va, y_va),
                'cat': ensemble_model.optimize_cat_params(X_tr, y_tr, X_va, y_va),
            }
            ensemble_model.global_best_params = global_best
            logger.info(f"전역 파라미터 완료: {global_best}")
        except Exception as e:
            logger.warning(f"전역 파라미터 최적화 실패: {e}. 기본 파라미터 사용")
    else:
        logger.info("저장된 파라미터 사용(전역 파라미터 로드됨)")

    # 유형 기반 파생 추가 (검증은 과거(train_val)로만 프로파일)
    train_val, val_df = add_building_type_features(train_val, val_df, ref_df=train_val)
    
    val_pred = ensemble_model.train_and_predict(
        train_val, val_df,
        optimize_params=False,                   # per-building 최적화 끔
        optimize_weights=not use_saved_params   # 앙상블 가중치만 필요시 최적화
    )


    logger.info("\n" + ensemble_model.get_feature_importance_summary())

    if not use_saved_params:
        ensemble_model.save_params(param_path)

    y_true_val = val_df['전력소비량(kWh)'].to_numpy(dtype=np.float64)
    val_score = smape(y_true_val, val_pred)
    logger.info(f"검증 데이터셋 sMAPE 점수: {val_score:.4f}")

    # 5) 최종 예측(제출)
    logger.info("전체 학습 데이터로 최종 예측 시작...")

    # 최종은 전체 학습을 ref로 사용해 타깃 결측 보강
    train_full = _impute_target_timewise(
        train_df_processed, 
        ref_df=train_df_processed,
        target_col='전력소비량(kWh)'
    )

    # (유형 파생을 쓴다면 보강된 train_full 기준으로 프로파일 생성)
    train_full_with_type, test_with_type = add_building_type_features(
        train_full, test_df_processed, ref_df=train_full
    )

    train_full_with_type = _attach_building_baselines(train_full, train_full_with_type)
    test_with_type       = _attach_building_baselines(train_full, test_with_type)

    extra_feats = [
        c for c in train_full_with_type.columns
        if (c.startswith(('type_', 'b_')) and c not in ('전력소비량(kWh)', 'num_date_time')
            and c not in ensemble_model.base_features)
    ]
    if extra_feats:
        ensemble_model.base_features += extra_feats
        # features는 train_and_predict 내부 analyze_feature_importance가 다시 잡아줌

    final_pred = ensemble_model.train_and_predict(
        train_full_with_type, test_with_type,
        optimize_params=False,   # (옵투나 전역파라미터를 이미 세팅/로드했다면 False로)
        optimize_weights=False   # (가중치도 이미 최적화/로드했다면 False로)
    )

    # 6) 저장
    sample_path_1 = os.path.join(path, 'sample_submission.csv')
    sample_path_2 = '/mnt/data/sample_submission.csv'
    if os.path.exists(sample_path_1):
        sample = pd.read_csv(sample_path_1)
    elif os.path.exists(sample_path_2):
        sample = pd.read_csv(sample_path_2)
    else:
        raise FileNotFoundError('sample_submission.csv를 찾을 수 없습니다.')

    # (안전) test 키 중복 체크: test_with_type 기준
    if not test_with_type['num_date_time'].is_unique:
        dup_cnt = int(test_with_type['num_date_time'].duplicated(keep=False).sum())
        raise ValueError(f"[제출 키 중복] num_date_time 중복 {dup_cnt}개. 전처리/병합 로직 확인 필요.")

    # 문자열 키 통일
    sample['num_date_time'] = sample['num_date_time'].astype(str)
    test_with_type['num_date_time'] = test_with_type['num_date_time'].astype(str)

    # 1) sample에 test 메타와 예측을 한 번만 조인 (이게 '베이스'가 됨)
    df_pred = sample[['num_date_time']].merge(
        test_with_type[['num_date_time', '건물번호', '일시']],
        on='num_date_time', how='left'
    )
    final_pred = np.asarray(final_pred, dtype=float)
    test_with_type['answer'] = final_pred  # 예측을 테스트 프레임에 먼저 붙이고
    submission = sample[['num_date_time']].merge(
        test_with_type[['num_date_time', 'answer']],
        on='num_date_time', how='left'
    )



    df_pred = df_pred.sort_values(['건물번호', '일시']).copy()
    df_pred['answer'] = pd.to_numeric(df_pred['answer'], errors='coerce')

    def _time_only(g):
        g = g.copy().sort_values('일시')
        s = pd.Series(g['answer'].values, index=pd.to_datetime(g['일시']))
        s = s.interpolate(method='time', limit_direction='both').ffill().bfill()
        g['answer'] = s.values
        return g

    df_pred = df_pred.groupby('건물번호', group_keys=False).apply(_time_only)

    # 4) 타입/범위 마무리
    df_pred['answer'] = df_pred['answer'].astype(float).clip(lower=0)

    # 5) 제출 형태로 복원(샘플 순서 보장, 조인은 마지막에 딱 1번만)
    submission = sample[['num_date_time']].merge(
        df_pred[['num_date_time', 'answer']],
        on='num_date_time', how='left'
    )

    # 최종 검증(컬럼/길이/순서)
    assert list(submission.columns) == ['num_date_time', 'answer']
    assert len(submission) == len(sample), "행 수가 sample과 다릅니다."
    assert submission['num_date_time'].equals(sample['num_date_time']), "sample 순서와 다릅니다."

    # ===== 저장 & sMAPE 포함 파일명 =====
    val_str = f"{val_score:.4f}"
    out_path = os.path.join(path, f'ensemble_submission_feature_importance_val{val_str}_{timestamp}.csv')
    submission.to_csv(out_path, index=False)

    logger.info(f"결과 파일 저장 완료: {out_path}")
    logger.info(f"검증 sMAPE: {val_score:.6f}")
    print(f"[FINAL] saved: {out_path}")
    print(f"[FINAL] validation sMAPE: {val_score:.6f}")

    # (선택) 지표 로그 파일도 남기기
    metrics_path = os.path.join(path, f'metrics_{timestamp}.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"validation sMAPE: {val_score:.6f}\n")
        f.write(f"submission_path: {out_path}\n")
        f.write(f"rows: {len(submission)}\n")
        f.write(f"n_selected_features: {len(ensemble_model.features)}\n")
    logger.info(f"지표 저장: {metrics_path}")


if __name__ == "__main__":
    main()
