import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import holidays
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy.optimize import minimize
import optuna
import warnings
warnings.filterwarnings("ignore")
import os
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ----------------- 로깅 -----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------- 설정 -----------------
class Config:
    # (기존) 재현성/기본 랜덤 스테이트
    RANDOM_STATE = 190

    # === Optuna(탐색) ===
    OPTUNA_TRIALS = 100
    OPTUNA_SEED = 190

    # === CV/분할 ===
    CV_FOLDS = 4
    MIN_VAL_HOURS = 24
    INNER_VAL_RATIO = 0.2
    VAL_DATE = datetime(2024, 8, 18)

    # === 학습 반복 평균 ===
    TRAIN_REPEATS = 7
    TRAIN_SEED_BASE = 500

    # 앙상블 초기 가중치
    ENSEMBLE_WEIGHTS = {'xgb': 0.45, 'lgb': 0.35, 'cat': 0.2}

    # 증강(학습 데이터 전용)
    USE_AUGMENT = True
    AUG_N_COPIES = 4
    AUG_NOISE_STD = 0.05
    
    # 피처 중요도
    USE_FEATURE_IMPORTANCE = True
    IMPORTANCE_THRESHOLD = 0.2
    TOP_K_FEATURES = 70

# ----------------- 경로 -----------------
path = './keras/et/'

# ----------------- 지표 -----------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    eps = 1e-9
    denom = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

# ----------------- 데이터 로드 -----------------
def load_data():
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

# ----------------- 파생피처(기존) -----------------
def _add_time_features(df):
    df['year'] = df['일시'].dt.year
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    df['weekday'] = df['일시'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['is_workhour'] = ((df['hour'] >= 9) & (df['hour'] < 18)).astype(int)
    df['is_peak'] = df['hour'].isin([10, 11, 18, 19, 20]).astype(int)
    return df

def _add_holiday_features(df):
    kr_holidays = holidays.KR()
    d = df['일시'].dt.date
    df['is_holiday'] = d.map(lambda x: 1 if x in kr_holidays else 0)
    prev_dates = (df['일시'] - pd.Timedelta(days=1)).dt.date
    next_dates = (df['일시'] + pd.Timedelta(days=1)).dt.date
    df['is_prev_holiday'] = prev_dates.map(lambda x: 1 if x in kr_holidays else 0)
    df['is_next_holiday'] = next_dates.map(lambda x: 1 if x in kr_holidays else 0)
    df['is_workingday'] = ((df['is_holiday'] == 0) & (df['is_weekend'] == 0)).astype(int)
    return df

def _add_capacity_building_features(df):
    for c in ['ESS저장용량(kWh)', '태양광용량(kW)', 'PCS용량(kW)', '연면적(m2)', '냉방면적(m2)']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['has_pv']  = (df.get('태양광용량(kW)', 0).fillna(0) > 0).astype(int)
    df['has_ess'] = (df.get('ESS저장용량(kWh)', 0).fillna(0) > 0).astype(int)
    df['has_pcs'] = (df.get('PCS용량(kW)', 0).fillna(0) > 0).astype(int)
    eps = 1e-6
    if '연면적(m2)' in df.columns and '냉방면적(m2)' in df.columns:
        df['냉방면적비'] = (df['냉방면적(m2)'] / (df['연면적(m2)'] + eps)).clip(lower=0)
    cap_sum = df.get('태양광용량(kW)', 0).fillna(0).astype(float) \
              + df.get('PCS용량(kW)', 0).fillna(0).astype(float) \
              + df.get('ESS저장용량(kWh)', 0).fillna(0).astype(float)
    area = df.get('연면적(m2)', 0).fillna(0).astype(float) + 1e-3
    df['cap_per_area_log1p'] = np.log1p((cap_sum / area).values)
    return df

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
    tz_hours: int = 9
) -> pd.DataFrame:
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
    out['cloud_cover_rain'] = np.where(out['강수량(mm)'] <= 0, 0.0, (out['강수량(mm)']/(out['강수량(mm)']+2)).clip(0, 1))
    out['cloud_cover_humidity'] = out['cloudy_based_on_humidity']
    out['cloud_cover_combined'] = (0.5*out['cloud_cover_humidity'] + 0.3*out['cloud_cover_dewpoint'] + 0.2*out['cloud_cover_rain']).clip(0, 1)
    out['cloud_cover_weighted'] = out['cloud_cover_combined']

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

def _impute_target_timewise(df: pd.DataFrame, ref_df: pd.DataFrame,
                            target_col: str = '전력소비량(kWh)') -> pd.DataFrame:
    """
    같은 '건물번호×날짜' 내에서 ffill→bfill로 타깃 결측을 우선 메우고,
    아직 남은 NaN은 ref_df(참조 데이터; 보통 같은 split)에서
    ① 건물×요일×시간 중앙값 → ② 건물×시간 중앙값 → ③ 건물 중앙값 으로 채움.
    """
    out = df.copy()

    # 1) 동일일자 내 ffill → bfill
    out = out.sort_values(['건물번호', '일시'])
    out['__date'] = out['일시'].dt.date
    out[target_col] = (
        out
        .groupby(['건물번호', '__date'])[target_col]
        .transform(lambda s: s.ffill().bfill())
    )

    # 2) 백업용 통계(ref_df에서 산출)
    ref = ref_df.copy()
    ref['__hour'] = ref['일시'].dt.hour
    ref['__weekday'] = ref['일시'].dt.weekday

    # 건물×요일×시간
    m_bwh = (
        ref.groupby(['건물번호', '__weekday', '__hour'])[target_col]
           .median()
           .rename('m_bwh')
           .reset_index()
    )
    out['__hour'] = out['일시'].dt.hour
    out['__weekday'] = out['일시'].dt.weekday
    out = out.merge(m_bwh, left_on=['건물번호','__weekday','__hour'],
                    right_on=['건물번호','__weekday','__hour'], how='left')
    mask = out[target_col].isna()
    out.loc[mask, target_col] = out.loc[mask, 'm_bwh']
    out.drop(columns=['m_bwh'], inplace=True)

    # 건물×시간
    m_bh = (
        ref.groupby(['건물번호', '__hour'])[target_col]
           .median()
           .rename('m_bh')
           .reset_index()
    )
    out = out.merge(m_bh, on=['건물번호','__hour'], how='left')
    mask = out[target_col].isna()
    out.loc[mask, target_col] = out.loc[mask, 'm_bh']
    out.drop(columns=['m_bh'], inplace=True)

    # 건물
    m_b = (
        ref.groupby('건물번호')[target_col]
           .median()
           .rename('m_b')
           .reset_index()
    )
    out = out.merge(m_b, on='건물번호', how='left')
    mask = out[target_col].isna()
    out.loc[mask, target_col] = out.loc[mask, 'm_b']
    out.drop(columns=['m_b','__date','__hour','__weekday'], inplace=True)

    # 안전 처리
    out[target_col] = out[target_col].astype(float).clip(lower=0)
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

# ----------------- 전처리 메인 -----------------
def preprocess_data(train_df, test_df, building_df):
    logger.info("데이터 전처리 시작...")

    train_df['일시'] = pd.to_datetime(train_df['일시'])
    test_df['일시'] = pd.to_datetime(test_df['일시'])
    train_df = pd.merge(train_df, building_df, on='건물번호', how='left')
    test_df = pd.merge(test_df, building_df, on='건물번호', how='left')

    train_df = _add_time_features(train_df)
    test_df  = _add_time_features(test_df)

    train_df = _fill_weather_missing(train_df)
    test_df  = _fill_weather_missing(test_df)

    train_df = _add_holiday_features(train_df)
    test_df  = _add_holiday_features(test_df)

    train_df = _add_capacity_building_features(train_df)
    test_df  = _add_capacity_building_features(test_df)

    train_df = _add_degree_days(train_df)
    test_df  = _add_degree_days(test_df)

    train_df = _add_extra_weather_calendar_features_noaa(train_df)
    test_df  = _add_extra_weather_calendar_features_noaa(test_df)

    # LabelEncoding (유형)
    for col in ['건물유형']:
        if col in train_df.columns:
            le = LabelEncoder()
            combined = pd.concat([train_df[col], test_df[col]])
            le.fit(combined.astype(str))
            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))

    for df in (train_df, test_df):
        df['num_date_time'] = (
            df['건물번호'].astype(int).astype(str) + '_' +
            df['일시'].dt.strftime('%Y%m%d %H')
        )

    logger.info("데이터 전처리 완료")
    return train_df, test_df

# ----------------- 건물유형 기반 파생 (누수 방지) -----------------
def add_building_type_features(train_df: pd.DataFrame,
                               pred_df: pd.DataFrame,
                               ref_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ref_df(=학습 참조 데이터)에서만 집계한 유형별 통계/민감도를
    train_df / pred_df 에 merge하여 파생피처 생성.
    - type_hour_median / type_hour_rel / type_occ_prob
    - type_workingday_ratio / type_holiday_ratio
    - type_beta_cdd / type_beta_hdd / CDD_x_type_beta / HDD_x_type_beta
    - cap_per_area_rel_type
    """
    eps = 1e-8
    # --- 기본 키 보장 ---
    for df in (train_df, pred_df, ref_df):
        assert {'건물유형','hour','is_workingday','is_holiday','CDD','HDD','cap_per_area_log1p'}.issubset(df.columns), \
            "필수 컬럼 누락 (건물유형/hour/is_workingday/is_holiday/CDD/HDD/cap_per_area_log1p)"

    # ==== (1) 유형별 시간대 패턴 ====
    # type-hour 중앙값
    th = (ref_df.groupby(['건물유형','hour'])['전력소비량(kWh)']
                .median().rename('type_hour_median').reset_index())
    # 유형 전체 중앙값
    tmed = (ref_df.groupby('건물유형')['전력소비량(kWh)']
                 .median().rename('type_median').reset_index())
    th = th.merge(tmed, on='건물유형', how='left')
    th['type_hour_rel'] = th['type_hour_median'] / (th['type_median'] + eps)

    # 유형별 24시간 min-max로 점유/가동 proxy
    mm = th.groupby('건물유형')['type_hour_median'].agg(['min','max']).reset_index()
    th = th.merge(mm, on='건물유형', how='left')
    th['type_occ_prob'] = (th['type_hour_median'] - th['min']) / (th['max'] - th['min'] + eps)
    th = th.drop(columns=['min','max'])

    def _merge_th(df):
        out = df.merge(th[['건물유형','hour','type_hour_median','type_hour_rel','type_occ_prob']],
                       on=['건물유형','hour'], how='left')
        # fallback: 유형 중앙값으로 대체
        out = out.merge(tmed, on='건물유형', how='left')
        for col in ['type_hour_median','type_hour_rel','type_occ_prob']:
            if col == 'type_hour_median':
                out[col] = out[col].fillna(out['type_median'])
            else:
                out[col] = out[col].fillna(1.0 if col=='type_hour_rel' else 0.5)
        out = out.drop(columns=['type_median'])
        return out

    train_aug = _merge_th(train_df.copy())
    pred_aug  = _merge_th(pred_df.copy())

    # ==== (2) 근무일/휴일 비율 ====
    # workingday ratio = (WD median)/(~WD median)
    wd = (ref_df.groupby(['건물유형','is_workingday'])['전력소비량(kWh)']
               .median().unstack(fill_value=np.nan))
    wd = wd.rename(columns={0:'m_nonwd',1:'m_wd'}).reset_index()
    wd['type_workingday_ratio'] = (wd['m_wd'] + eps)/(wd['m_nonwd'] + eps)

    # holiday ratio = (Holiday median)/(Non-Holiday median)
    hd = (ref_df.groupby(['건물유형','is_holiday'])['전력소비량(kWh)']
               .median().unstack(fill_value=np.nan))
    hd = hd.rename(columns={0:'m_nonhol',1:'m_hol'}).reset_index()
    hd['type_holiday_ratio'] = (hd['m_hol'] + eps)/(hd['m_nonhol'] + eps)

    type_day_ratio = wd[['건물유형','type_workingday_ratio']]\
        .merge(hd[['건물유형','type_holiday_ratio']], on='건물유형', how='outer')
    type_day_ratio['type_workingday_ratio'] = type_day_ratio['type_workingday_ratio'].fillna(1.0)
    type_day_ratio['type_holiday_ratio'] = type_day_ratio['type_holiday_ratio'].fillna(1.0)

    train_aug = train_aug.merge(type_day_ratio, on='건물유형', how='left')
    pred_aug  = pred_aug.merge(type_day_ratio,  on='건물유형', how='left')
    for col in ['type_workingday_ratio','type_holiday_ratio']:
        train_aug[col] = train_aug[col].fillna(1.0)
        pred_aug[col]  = pred_aug[col].fillna(1.0)

    # ==== (3) 유형별 CDD/HDD 민감도(선형회귀 기울기) ====
    betas = []
    for tval, g in ref_df.groupby('건물유형'):
        y = g['전력소비량(kWh)'].astype(float).values
        X = np.c_[g['CDD'].astype(float).values,
                  g['HDD'].astype(float).values,
                  np.ones(len(g))]
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)  # [beta_cdd, beta_hdd, intercept]
            betas.append((tval, float(coef[0]), float(coef[1])))
        except Exception:
            betas.append((tval, 0.0, 0.0))

    beta_df = pd.DataFrame(betas, columns=['건물유형','type_beta_cdd','type_beta_hdd'])
    # 정규화(0~1)
    for col in ['type_beta_cdd','type_beta_hdd']:
        mn = beta_df[col].min()
        mx = beta_df[col].max()
        if np.isclose(mx - mn, 0):
            beta_df[col+'_norm'] = 0.5
        else:
            beta_df[col+'_norm'] = (beta_df[col] - mn)/(mx - mn)

    train_aug = train_aug.merge(beta_df[['건물유형','type_beta_cdd','type_beta_hdd','type_beta_cdd_norm','type_beta_hdd_norm']],
                                on='건물유형', how='left')
    pred_aug  = pred_aug.merge(beta_df[['건물유형','type_beta_cdd','type_beta_hdd','type_beta_cdd_norm','type_beta_hdd_norm']],
                                on='건물유형', how='left')
    for col in ['type_beta_cdd','type_beta_hdd','type_beta_cdd_norm','type_beta_hdd_norm']:
        train_aug[col] = train_aug[col].fillna(train_aug[col].median())
        pred_aug[col]  = pred_aug[col].fillna(train_aug[col].median())

    # 상호작용
    train_aug['CDD_x_type_beta'] = train_aug['CDD'] * train_aug['type_beta_cdd_norm']
    train_aug['HDD_x_type_beta'] = train_aug['HDD'] * train_aug['type_beta_hdd_norm']
    pred_aug['CDD_x_type_beta']  = pred_aug['CDD']  * pred_aug['type_beta_cdd_norm']
    pred_aug['HDD_x_type_beta']  = pred_aug['HDD']  * pred_aug['type_beta_hdd_norm']

    # ==== (4) 유형 기준 설비/면적 상대치 ====
    # cap_per_area_log1p 를 선형 스케일로 되돌려서 유형 중앙값과 비율 계산
    ref_df['_cap_lin'] = np.expm1(ref_df['cap_per_area_log1p'].astype(float))
    med_cap = (ref_df.groupby('건물유형')['_cap_lin'].median()
               .rename('type_cap_lin_median').reset_index())

    def _attach_cap_rel(df):
        out = df.merge(med_cap, on='건물유형', how='left')
        out['_cap_lin_row'] = np.expm1(out['cap_per_area_log1p'].astype(float))
        out['cap_per_area_rel_type'] = (out['_cap_lin_row'] + eps)/(out['type_cap_lin_median'] + eps)
        return out.drop(columns=['_cap_lin_row','type_cap_lin_median'])

    train_aug = _attach_cap_rel(train_aug)
    pred_aug  = _attach_cap_rel(pred_aug)

    return train_aug, pred_aug

# ----------------- 피처 중요도 계산/관리 -----------------
class FeatureImportanceManager:
    def __init__(self):
        self.feature_importances = {}
        self.selected_features = None
        self.importance_scores = {}
    
    def calculate_feature_importance(self, X_train, y_train, feature_names, method='combined'):
        importances = {}
        if method in ['xgb', 'combined']:
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=600, random_state=Config.RANDOM_STATE,
                    n_jobs=-1, eval_metric="rmse", tree_method="hist",
                )
                xgb_model.fit(X_train, y_train, verbose=False)
                importances['xgb'] = xgb_model.feature_importances_
            except Exception as e:
                logger.warning(f"XGBoost 중요도 계산 실패: {e}")
                importances['xgb'] = np.ones(len(feature_names)) / len(feature_names)
        if method in ['lgb', 'combined']:
            try:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=600, random_state=Config.RANDOM_STATE,
                    n_jobs=-1, verbose=-1
                )
                lgb_model.fit(X_train, y_train)
                importances['lgb'] = lgb_model.feature_importances_
            except Exception as e:
                logger.warning(f"LightGBM 중요도 계산 실패: {e}")
                importances['lgb'] = np.ones(len(feature_names)) / len(feature_names)
        if method in ['rf', 'combined']:
            try:
                rf_model = RandomForestRegressor(
                    n_estimators=600, random_state=Config.RANDOM_STATE, n_jobs=-1
                )
                rf_model.fit(X_train, y_train)
                importances['rf'] = rf_model.feature_importances_
            except Exception as e:
                logger.warning(f"RandomForest 중요도 계산 실패: {e}")
                importances['rf'] = np.ones(len(feature_names)) / len(feature_names)

        if method == 'combined':
            combined_importance = np.mean(list(importances.values()), axis=0)
        else:
            combined_importance = importances[method]
        return dict(zip(feature_names, combined_importance))
    
    def select_features_by_importance(self, feature_importance_dict, threshold=None, top_k=None):
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        if top_k:
            selected = [feat for feat, imp in sorted_features[:top_k]]
        elif threshold:
            selected = [feat for feat, imp in sorted_features if imp >= threshold]
        else:
            selected = [feat for feat, imp in sorted_features]
        logger.info(f"피처 선택 완료: {len(selected)}개 / 총 {len(feature_importance_dict)}개")
        return selected
    
    def plot_feature_importance(self, feature_importance_dict, top_n=20, save_path=None):
        try:
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_features)
            plt.figure(figsize=(12, 8))
            sns.barplot(x=list(importances), y=list(features))
            plt.title(f'Top {top_n} Feature Importances')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"피처 중요도 플롯 저장: {save_path}")
            plt.show()
            return sorted_features
        except Exception as e:
            logger.warning(f"피처 중요도 시각화 실패: {e}")
            return list(feature_importance_dict.items())

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
            '날씨불안정지수'
        ]

        # === 신규: 건물유형 기반 피처 ===
        type_feats = [
            'type_hour_median','type_hour_rel','type_occ_prob',
            'type_workingday_ratio','type_holiday_ratio',
            'type_beta_cdd','type_beta_hdd','type_beta_cdd_norm','type_beta_hdd_norm',
            'CDD_x_type_beta','HDD_x_type_beta',
            'cap_per_area_rel_type'
        ]

        self.base_features = base_core + base_extra + type_feats
        self.features = self.base_features.copy()
        
        self.best_params = {}
        self.models = {}
        self.ensemble_weights = Config.ENSEMBLE_WEIGHTS.copy()
        self.weight_optimizer = EnsembleWeightOptimizer()
        self.feature_manager = FeatureImportanceManager()

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
            'cloud_okta','cloud_blocking_factor',
            # 새 연속형
            'type_hour_median','type_hour_rel','type_occ_prob',
            'type_workingday_ratio','type_holiday_ratio',
            'type_beta_cdd','type_beta_hdd','type_beta_cdd_norm','type_beta_hdd_norm',
            'CDD_x_type_beta','HDD_x_type_beta',
            'cap_per_area_rel_type'
        ]

    def analyze_feature_importance(self, train_df, method='combined'):
        logger.info("피처 중요도 분석 시작...")
        available_features = [f for f in self.base_features if f in train_df.columns]
        X = train_df[available_features].fillna(0).values
        y = train_df['전력소비량(kWh)'].values
        feature_importance_dict = self.feature_manager.calculate_feature_importance(
            X, y, available_features, method=method
        )
        plot_path = os.path.join(path, 'feature_importance.png')
        self.feature_manager.plot_feature_importance(feature_importance_dict, top_n=20, save_path=plot_path)
        if Config.USE_FEATURE_IMPORTANCE:
            selected_features = self.feature_manager.select_features_by_importance(
                feature_importance_dict,
                threshold=Config.IMPORTANCE_THRESHOLD,
                top_k=Config.TOP_K_FEATURES
            )
            self.features = selected_features
            logger.info(f"중요도 기반 피처 선택: {len(selected_features)}개")
        else:
            self.features = available_features
            logger.info(f"모든 피처 사용: {len(available_features)}개")
        self.feature_manager.importance_scores = feature_importance_dict
        self.feature_manager.selected_features = self.features
        return feature_importance_dict

    # --------- 증강 ---------
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

    # --------- Optuna ---------
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
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'random_state': Config.RANDOM_STATE,
                'n_jobs': -1
            }
            model = xgb.XGBRegressor(**params, tree_method="hist")
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      eval_metric='rmse', verbose=False, early_stopping_rounds=30)
            y_pred = model.predict(X_val)
            return smape(y_val, y_pred)

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=Config.OPTUNA_SEED))
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
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'random_state': Config.RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse',
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            y_pred = model.predict(X_val)
            return smape(y_val, y_pred)

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=Config.OPTUNA_SEED))
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
            model = cb.CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      early_stopping_rounds=30, verbose=False)
            y_pred = model.predict(X_val)
            return smape(y_val, y_pred)

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=Config.OPTUNA_SEED))
        study.optimize(objective, n_trials=Config.OPTUNA_TRIALS)
        return study.best_params

    # --------- 건물 단위 학습/예측 ---------
    def train_building_ensemble(self, building_train, building_pred, building_num,
                                has_label_in_pred=False, optimize_params=True):
        if building_train.empty or building_pred.empty:
            return np.full(len(building_pred), np.nan)

        # 내부 홀드아웃
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

        # 증강
        if Config.USE_AUGMENT:
            train_df_aug = self._augment_train_df(train_df_use, n_copies=Config.AUG_N_COPIES, noise_std=Config.AUG_NOISE_STD)
        else:
            train_df_aug = train_df_use

        X_train = train_df_aug[self.features].fillna(0).values
        y_train = train_df_aug['전력소비량(kWh)'].values
        X_val   = val_df_use[self.features].fillna(0).values
        y_val   = val_df_use['전력소비량(kWh)'].values
        X_pred  = building_pred[self.features].fillna(0).values

        # 파라미터 탐색(최초 1회)
        if optimize_params and building_num not in self.best_params:
            logger.info(f"건물 {building_num} 하이퍼파라미터 최적화 시작...")
            try:
                self.best_params[building_num] = {
                    'xgb': self.optimize_xgb_params(X_train, y_train, X_val, y_val),
                    'lgb': self.optimize_lgb_params(X_train, y_train, X_val, y_val),
                    'cat': self.optimize_cat_params(X_train, y_train, X_val, y_val)
                }
                logger.info(f"건물 {building_num} 최적화 완료")
            except Exception as e:
                logger.warning(f"건물 {building_num} 최적화 실패: {e}. 기본 파라미터 사용")
                self.best_params[building_num] = {}

        predictions = {}

        # XGB 반복 평균
        try:
            base_params = self.best_params.get(building_num, {}).get('xgb', {
                'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 1000,
                'random_state': Config.TRAIN_SEED_BASE, 'n_jobs': -1, 'verbose': -1
            })
            xgb_preds = []
            for r in range(Config.TRAIN_REPEATS):
                p = base_params.copy(); p['random_state'] = Config.TRAIN_SEED_BASE + r
                m = xgb.XGBRegressor(**p, eval_metric="rmse", tree_method="hist")
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=30)
                xgb_preds.append(m.predict(X_pred))
            predictions['xgb'] = np.mean(np.vstack(xgb_preds), axis=0)
        except Exception as e:
            logger.warning(f"xgb 훈련 실패: {e}")

        # LGB 반복 평균
        try:
            base_params = self.best_params.get(building_num, {}).get('lgb', {
                'learning_rate': 0.01, 'num_leaves': 31, 'n_estimators': 1000,
                'random_state': Config.TRAIN_SEED_BASE, 'n_jobs': -1, 'verbose': -1
            })
            lgb_preds = []
            for r in range(Config.TRAIN_REPEATS):
                p = base_params.copy(); p['random_state'] = Config.TRAIN_SEED_BASE + r
                m = lgb.LGBMRegressor(**p)
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse",
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                best_iter = getattr(m, "best_iteration_", None)
                lgb_preds.append(m.predict(X_pred, num_iteration=best_iter) if best_iter is not None else m.predict(X_pred))
            predictions['lgb'] = np.mean(np.vstack(lgb_preds), axis=0)
        except Exception as e:
            logger.warning(f"LightGBM 훈련 실패: {e}")

        # CAT 반복 평균
        try:
            base_params = self.best_params.get(building_num, {}).get('cat', {
                'learning_rate': 0.01, 'depth': 6, 'iterations': 1000,
                'random_seed': Config.TRAIN_SEED_BASE, 'thread_count': -1, 'verbose': False
            })
            cat_preds = []
            for r in range(Config.TRAIN_REPEATS):
                p = base_params.copy(); p['random_seed'] = Config.TRAIN_SEED_BASE + r
                m = cb.CatBoostRegressor(**p)
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
                cat_preds.append(m.predict(X_pred))
            predictions['cat'] = np.mean(np.vstack(cat_preds), axis=0)
        except Exception as e:
            logger.warning(f"CatBoost 훈련 실패: {e}")

        ensemble_pred = (
            self.ensemble_weights['xgb'] * predictions['xgb'] +
            self.ensemble_weights['lgb'] * predictions['lgb'] +
            self.ensemble_weights['cat'] * predictions['cat']
        )
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

                try:
                    m_xgb = xgb.XGBRegressor(n_estimators=600, random_state=Config.RANDOM_STATE,
                                             n_jobs=-1, eval_metric="rmse", tree_method="hist")
                    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=30)
                    preds_dict['xgb'].extend(m_xgb.predict(X_va))
                except:
                    preds_dict['xgb'].extend([y_tr.mean()] * len(va_idx))

                try:
                    m_lgb = lgb.LGBMRegressor(n_estimators=600, random_state=Config.RANDOM_STATE,
                                              n_jobs=-1, verbose=-1)
                    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    preds_dict['lgb'].extend(m_lgb.predict(X_va))
                except:
                    preds_dict['lgb'].extend([y_tr.mean()] * len(va_idx))

                try:
                    m_cat = cb.CatBoostRegressor(iterations=600, random_seed=Config.RANDOM_STATE,
                                                 thread_count=-1, verbose=False)
                    m_cat.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                              early_stopping_rounds=30, verbose=False)
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

    def save_params(self, filepath):
        save_data = {
            'best_params': self.best_params, 
            'ensemble_weights': self.ensemble_weights,
            'selected_features': self.features,
            'feature_importances': self.feature_manager.importance_scores
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"파라미터/가중치/피처 저장: {filepath}")

    def load_params(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            if isinstance(save_data, dict) and 'best_params' in save_data:
                self.best_params = save_data['best_params']
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
            logger.info("저장된 파라미터 없음. 새로 최적화 진행.")
            return False

# ----------------- 메인 -----------------
def main():
    logger.info("피처 중요도 기반 앙상블 모델 훈련 시작...")

    # 1) 로드
    train_df, test_df, building_df = load_data()

    # 2) 전처리
    train_df_processed, test_df_processed = preprocess_data(train_df, test_df, building_df)

    # 3) 모델
    ensemble_model = OptimizedEnsembleModel()

    # 파라미터 로드 시도
    param_path = os.path.join(path, f'best_params_with_features_{timestamp}.pkl')
    use_saved_params = ensemble_model.load_params(param_path)

    # 4) 홀드아웃 검증(누수 방지: train_val을 ref로 사용)
    logger.info("검증 데이터셋 성능 평가 시작...")
    train_val = train_df_processed[train_df_processed['일시'] < Config.VAL_DATE].copy()
    val_df   = train_df_processed[train_df_processed['일시'] >= Config.VAL_DATE].copy()

    train_val_type, val_with_type = add_building_type_features(
        train_val, val_df, ref_df=train_val
    )

    val_pred = ensemble_model.train_and_predict(
        train_val_type, val_with_type,
        optimize_params=not use_saved_params,
        optimize_weights=not use_saved_params
    )

    logger.info("\n" + ensemble_model.get_feature_importance_summary())

    if not use_saved_params:
        ensemble_model.save_params(param_path)

    y_true_val = val_df['전력소비량(kWh)'].to_numpy(dtype=np.float64)
    val_score = smape(y_true_val, val_pred)
    logger.info(f"검증 데이터셋 sMAPE: {val_score:.4f}")

    # 5) 최종 예측(제출) - ref=전체 train
    logger.info("전체 학습 데이터로 최종 예측 시작...")
    train_full_with_type, test_with_type = add_building_type_features(
        train_df_processed, test_df_processed, ref_df=train_df_processed
    )

    final_pred = ensemble_model.train_and_predict(
        train_full_with_type, test_with_type,
        optimize_params=False, optimize_weights=False
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

    if not test_df_processed['num_date_time'].is_unique:
        dup_cnt = int(test_df_processed['num_date_time'].duplicated(keep=False).sum())
        raise ValueError(f"[제출 키 중복] num_date_time 중복 {dup_cnt}개. 전처리/병합 로직 확인 필요.")

    sample['num_date_time'] = sample['num_date_time'].astype(str)
    test_df_processed['num_date_time'] = test_df_processed['num_date_time'].astype(str)

    pred_map = pd.DataFrame({
        'num_date_time': test_df_processed['num_date_time'].values,
        'answer': final_pred.astype(float)
    })

    submission = sample[['num_date_time']].merge(pred_map, on='num_date_time', how='left')

    # (기본) 결측/타입/범위 처리 — 필요 시 ffill/bfill 방식으로 교체 가능
    fill_val = float(train_df_processed['전력소비량(kWh)'].mean())
    submission['answer'] = submission['answer'].astype(float).fillna(fill_val).clip(lower=0)

    assert list(submission.columns) == ['num_date_time', 'answer']
    assert len(submission) == len(sample), "행 수가 sample과 다릅니다."
    assert submission['num_date_time'].equals(sample['num_date_time']), "sample 순서와 다릅니다."

    val_str = f"{val_score:.4f}"
    out_path = os.path.join(path, f'ensemble_submission_feature_importance_val{val_str}_{timestamp}.csv')
    submission.to_csv(out_path, index=False)

    logger.info(f"결과 파일 저장 완료: {out_path}")
    logger.info(f"검증 sMAPE: {val_score:.6f}")
    print(f"[FINAL] saved: {out_path}")
    print(f"[FINAL] validation sMAPE: {val_score:.6f}")

    metrics_path = os.path.join(path, f'metrics_{timestamp}.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"validation sMAPE: {val_score:.6f}\n")
        f.write(f"submission_path: {out_path}\n")
        f.write(f"rows: {len(submission)}\n")
        f.write(f"n_selected_features: {len(ensemble_model.features)}\n")
    logger.info(f"지표 저장: {metrics_path}")

if __name__ == "__main__":
    main()
