"""
Dashboard Streamlit – Treino de MLP (Keras) com seleção de variáveis via CHECKBOX.

Como rodar
1) venv (opcional)
   py -m venv .venv
   .venv\Scripts\activate
2) dependências
   pip install streamlit tensorflow==2.17.0 pandas scikit-learn openpyxl matplotlib joblib
3) executar
   streamlit run app.py
"""

import io
import re
import json
import traceback
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import threading
import queue
import time


import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ===============================
# Config / utilitários
# ===============================

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TS_FMT = "%Y%m%d_%H%M%S"
_timestamp_re = re.compile(r"(\d{8}_\d{6})")

def _ts() -> str:
    return datetime.now().strftime(TS_FMT)

def _extract_ts(name: str) -> Optional[str]:
    m = _timestamp_re.search(name)
    return m.group(1) if m else None

def _read_any(uploaded) -> pd.DataFrame:
    """Lê CSV/XLSX e detecta automaticamente se a primeira linha é cabeçalho."""
    
    name = uploaded.name.lower() if hasattr(uploaded, "name") else str(uploaded).lower()

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded, header=None)
    else:
        df = pd.read_csv(uploaded, header=None)

    # verifica se a primeira linha parece cabeçalho (strings)
    first_row = df.iloc[0]

    is_header = all(isinstance(v, str) for v in first_row)

    if is_header:
        df.columns = first_row
        df = df.drop(index=0).reset_index(drop=True)
    else:
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]

    return df

#    def _try(header):
#        if name.endswith((".xlsx", ".xls")):
#            return pd.read_excel(uploaded, header=header)
#        return pd.read_csv(uploaded, header=header)

#    try:
#        return _try(0)
#    except Exception:
#       if hasattr(uploaded, "seek"):
#           uploaded.seek(0)
#        return _try(None)

def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda s: pd.to_numeric(s, errors="coerce"))



def _max_consecutive_nan(s: pd.Series) -> int:
    """Retorna o maior gap consecutivo de NaN em uma série."""
    mask = s.isna().to_numpy()
    if not mask.any():
        return 0
    max_run = run = 0
    for is_missing in mask:
        if is_missing:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo de valores ausentes por coluna."""
    n = max(len(df), 1)
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append({
            "coluna": str(col),
            "ausentes": missing,
            "percentual_ausente": float(missing / n),
            "maior_gap_consecutivo": _max_consecutive_nan(df[col]),
        })
    return pd.DataFrame(rows)


def adaptive_impute_numeric_df(
    df: pd.DataFrame,
    *,
    short_gap_limit: int = 3,
    complex_missing_threshold: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Pipeline adaptativo para valores ausentes em dados numéricos.

    Estratégia:
    1) Converte as variáveis para formato numérico, preservando valores ausentes.
    2) Remove colunas totalmente vazias, pois não há informação para imputar.
    3) Interpola gaps curtos respeitando a ordem original das linhas.
    4) Usa MICE/IterativeImputer quando a ausência restante é mais complexa.
       Se falhar, usa KNNImputer.
    5) Aplica mediana como fallback final para qualquer resíduo.

    Observação: a função não remove linhas como regra padrão. Linhas só deixam de existir
    se a base não tiver informação suficiente para treinar depois da imputação.
    """
    df_num = coerce_numeric_df(df).replace([np.inf, -np.inf], np.nan)
    report_before = _missing_report(df_num)
    notes: List[str] = []

    if df_num.empty:
        raise ValueError("A base selecionada está vazia.")

    all_nan_cols = [c for c in df_num.columns if df_num[c].isna().all()]
    if all_nan_cols:
        notes.append(
            "Colunas totalmente vazias removidas antes da imputação: "
            + ", ".join(map(str, all_nan_cols))
        )
        df_num = df_num.drop(columns=all_nan_cols)

    if df_num.empty:
        raise ValueError("Todas as variáveis selecionadas estão vazias ou ausentes; não há dados suficientes para imputação.")

    total_missing_before = int(df_num.isna().sum().sum())
    if total_missing_before == 0:
        report = report_before.copy()
        report["metodo_aplicado"] = "sem valores ausentes"
        return df_num.reset_index(drop=True), report, ["Nenhum valor ausente foi encontrado nas variáveis selecionadas."]

    # 1) Interpolação temporal/ordenada para gaps curtos.
    df_work = df_num.copy()
    cols_with_short_gaps = []
    for col in df_work.columns:
        if df_work[col].isna().any() and _max_consecutive_nan(df_work[col]) <= short_gap_limit:
            cols_with_short_gaps.append(col)

    if cols_with_short_gaps:
        df_work[cols_with_short_gaps] = (
            df_work[cols_with_short_gaps]
            .interpolate(method="linear", limit=short_gap_limit, limit_direction="both")
        )
        notes.append(
            f"Interpolação linear aplicada em gaps curtos (até {short_gap_limit} linhas) "
            f"em {len(cols_with_short_gaps)} coluna(s)."
        )

    remaining_missing = int(df_work.isna().sum().sum())
    method_used = "interpolação + mediana"

    # 2) Casos restantes: MICE se a ausência for relevante/complexa; KNN como alternativa.
    if remaining_missing > 0:
        missing_ratio_remaining = float(remaining_missing / max(df_work.size, 1))
        try:
            if missing_ratio_remaining >= complex_missing_threshold and df_work.shape[1] >= 2:
                # Em versões recentes do scikit-learn, alguns estimadores baseados em árvores
                # podem falhar dentro do IterativeImputer com erros internos como
                # "NoneType object has no attribute pop". O estimador padrão do
                # IterativeImputer (BayesianRidge) é mais estável para uso geral no app.
                imputer = IterativeImputer(
                    estimator=None,
                    max_iter=12,
                    random_state=random_state,
                    initial_strategy="median",
                    skip_complete=True,
                    sample_posterior=False,
                )
                arr = imputer.fit_transform(df_work)
                method_used = "interpolação + MICE/IterativeImputer estável + mediana"
                notes.append("IterativeImputer aplicado aos valores ausentes restantes devido ao padrão de ausência observado.")
            elif df_work.shape[1] >= 2:
                n_neighbors = min(5, max(1, len(df_work) - 1))
                imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
                arr = imputer.fit_transform(df_work)
                method_used = "interpolação + KNNImputer + mediana"
                notes.append(f"KNNImputer aplicado aos valores ausentes restantes com {n_neighbors} vizinho(s).")
            else:
                arr = df_work.to_numpy(dtype=float)
                notes.append("Base com uma única coluna selecionada; utilizando mediana como estratégia de contingência.")
            df_work = pd.DataFrame(arr, columns=df_work.columns, index=df_work.index)
        except Exception as exc:
            notes.append(f"Imputação multivariada falhou ({exc}). Tentando KNN antes da mediana.")
            try:
                if df_work.shape[1] >= 2 and len(df_work) >= 2:
                    n_neighbors = min(5, max(1, len(df_work) - 1))
                    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
                    arr = imputer.fit_transform(df_work)
                    df_work = pd.DataFrame(arr, columns=df_work.columns, index=df_work.index)
                    method_used = "interpolação + KNNImputer fallback + mediana"
                    notes.append(f"KNNImputer fallback aplicado com {n_neighbors} vizinho(s).")
                else:
                    notes.append("Dados insuficientes para KNN; utilizando mediana como estratégia de contingência.")
            except Exception as exc_knn:
                notes.append(f"Imputação por KNN também falhou ({exc_knn}). Usando mediana como fallback final.")

    # 3) Fallback final por mediana. Se a mediana não existir, usa 0 apenas como último recurso.
    medians = df_work.median(numeric_only=True)
    df_work = df_work.fillna(medians).fillna(0.0)
    remaining_after = int(df_work.isna().sum().sum())

    report_after = _missing_report(df_work)
    report = report_before.merge(
        report_after[["coluna", "ausentes"]].rename(columns={"ausentes": "ausentes_apos"}),
        on="coluna",
        how="left",
    )
    report["ausentes_apos"] = report["ausentes_apos"].fillna(report["ausentes"]).astype(int)
    report["metodo_aplicado"] = method_used

    notes.append(
        f"Valores ausentes nas variáveis selecionadas: {total_missing_before} antes → {remaining_after} depois da imputação."
    )
    return df_work.reset_index(drop=True), report, notes


def _prepare_supervised_data_with_missingness(
    df_raw: pd.DataFrame,
    in_cols: List[str],
    out_cols: List[str],
    *,
    input_row_missing_limit: float = 0.80,
    input_col_missing_limit: float = 0.60,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, float], np.ndarray]:
    """
    Prepara dados supervisionados para bases com elevada proporção de valores ausentes.

    Estratégia:
    - y/saída nunca é imputada: linhas com alvo ausente são removidas.
    - quando a base apresenta elevada ausência, o controle de qualidade é ajustado automaticamente.
    - entradas com ausência extrema são removidas quando possível.
    - linhas com pouca informação são removidas quando há amostras suficientes.
    - adiciona indicadores binários "__was_missing" por coluna imputada.
    - adiciona "__missing_ratio" e "__row_quality" como features de qualidade.
    - remove features constantes/quase constantes após o pré-processamento.
    - gera sample_weight não linear: linhas mais completas têm influência bem maior no treino.
    """
    notes: List[str] = []
    selected_cols = list(dict.fromkeys(in_cols + out_cols))
    df_num = coerce_numeric_df(df_raw[selected_cols]).replace([np.inf, -np.inf], np.nan)

    if df_num.empty:
        raise ValueError("A base selecionada está vazia.")

    original_rows = int(len(df_num))
    original_input_missing_ratio = float(df_num[in_cols].isna().mean().mean()) if in_cols else 0.0
    original_selected_missing_ratio = float(df_num[selected_cols].isna().mean().mean()) if selected_cols else 0.0

    # Ajuste automático: quanto maior a ausência, mais restritivo o controle de qualidade.
    quality_regime = "normal"
    effective_row_limit = float(input_row_missing_limit)
    effective_col_limit = float(input_col_missing_limit)
    min_confidence_clip = 0.20

    if original_input_missing_ratio >= 0.35:
        quality_regime = "alta taxa de valores ausentes"
        effective_row_limit = min(effective_row_limit, 0.45)
        effective_col_limit = min(effective_col_limit, 0.50)
        min_confidence_clip = 0.08
    elif original_input_missing_ratio >= 0.20:
        quality_regime = "taxa moderada de valores ausentes"
        effective_row_limit = min(effective_row_limit, 0.60)
        effective_col_limit = min(effective_col_limit, 0.55)
        min_confidence_clip = 0.12

    notes.append(
        "Filtros adaptativos definidos como "
        f"amostra ≤ {effective_row_limit:.0%} de ausência, variável ≤ {effective_col_limit:.0%} de ausência "
        f"({quality_regime})."
    )

    # 1) Saída sem imputação: remove apenas linhas cujo alvo real está ausente.
    target_mask = df_num[out_cols].notna().all(axis=1)
    n_target_missing = int((~target_mask).sum())
    if n_target_missing > 0:
        notes.append(
            f"{n_target_missing} linha(s) removida(s) por ausência na variável de saída. "
            "A saída não foi imputada para evitar treino com rótulos artificiais."
        )

    df_obs = df_num.loc[target_mask].reset_index(drop=True)
    if len(df_obs) < 10:
        raise ValueError(
            "Poucas linhas com saída real disponível após remover alvos ausentes "
            "(precisa de pelo menos 10)."
        )

    X_raw = df_obs[in_cols].copy()
    y_df = df_obs[out_cols].copy()

    # 2) Mede confiabilidade por linha antes da imputação.
    row_missing_ratio = X_raw.isna().mean(axis=1).astype(float)

    # 3) Filtro rígido/adaptativo por linha. Só aplica se sobrar uma quantidade razoável.
    row_keep_mask = row_missing_ratio <= effective_row_limit
    n_rows_dropped = int((~row_keep_mask).sum())
    min_rows_after_filter = max(50, int(0.20 * len(X_raw)))
    applied_row_filter = False
    if n_rows_dropped > 0 and int(row_keep_mask.sum()) >= min_rows_after_filter:
        X_raw = X_raw.loc[row_keep_mask].reset_index(drop=True)
        y_df = y_df.loc[row_keep_mask].reset_index(drop=True)
        row_missing_ratio = row_missing_ratio.loc[row_keep_mask].reset_index(drop=True)
        applied_row_filter = True
        notes.append(
            f"{n_rows_dropped} linha(s) removida(s) por terem mais de "
            f"{effective_row_limit:.0%} das entradas ausentes."
        )
    elif n_rows_dropped > 0:
        row_missing_ratio = row_missing_ratio.reset_index(drop=True)
        notes.append(
            "Linhas com muitas entradas ausentes foram mantidas porque o filtro rígido "
            "reduziria demais a base disponível para treino."
        )
    else:
        row_missing_ratio = row_missing_ratio.reset_index(drop=True)

    # 4) Remoção adaptativa de variáveis com baixa completude.
    col_missing_ratio = X_raw.isna().mean(axis=0)
    high_missing_cols = [c for c, r in col_missing_ratio.items() if r > effective_col_limit]
    usable_cols = [c for c in X_raw.columns if c not in high_missing_cols]
    applied_col_filter = False
    if high_missing_cols and usable_cols:
        X_raw = X_raw[usable_cols]
        applied_col_filter = True
        notes.append(
            "Variável(is) de entrada removida(s) devido à elevada proporção de valores ausentes "
            f"(> {effective_col_limit:.0%} de ausência): " + ", ".join(map(str, high_missing_cols))
        )
    elif not usable_cols:
        notes.append(
            "Todas as variáveis de entrada ultrapassaram o limite de ausência; nenhuma variável foi removida para "
            "preservar ao menos uma variável de entrada."
        )

    # 5) Indicadores de ausência: informam à rede quais valores foram inventados pelo imputador.
    missing_indicator_cols = []
    indicator_frames = []
    for col in X_raw.columns:
        ratio = float(X_raw[col].isna().mean())
        if 0.0 < ratio < 1.0:
            indicator_name = f"{col}__was_missing"
            indicator_frames.append(X_raw[col].isna().astype(float).rename(indicator_name))
            missing_indicator_cols.append(indicator_name)

    # 6) Imputação adaptativa somente nas entradas.
    X_imputed, imputation_report, imputation_notes = adaptive_impute_numeric_df(
        X_raw,
        short_gap_limit=3,
        complex_missing_threshold=0.10,
        random_state=random_state,
    )
    notes.extend(imputation_notes)

    feature_parts = [X_imputed.reset_index(drop=True)]
    if indicator_frames:
        indicators = pd.concat(indicator_frames, axis=1).reset_index(drop=True)
        feature_parts.append(indicators)
        notes.append(
            f"{len(missing_indicator_cols)} coluna(s) indicadora(s) de ausência adicionada(s) às entradas."
        )

    # Features contínuas de qualidade da linha.
    row_quality = (1.0 - row_missing_ratio.astype(float)).clip(0.0, 1.0)
    feature_parts.append(row_missing_ratio.astype(float).rename("__missing_ratio").reset_index(drop=True).to_frame())
    feature_parts.append(row_quality.astype(float).rename("__row_quality").reset_index(drop=True).to_frame())
    notes.append("Features '__missing_ratio' e '__row_quality' adicionadas ao modelo.")

    X_final = pd.concat(feature_parts, axis=1)

    # 7) Remove features constantes/quase constantes, comuns depois de imputação pesada.
    constant_cols = []
    for col in list(X_final.columns):
        s = pd.to_numeric(X_final[col], errors="coerce")
        if s.nunique(dropna=False) <= 1 or float(s.std(skipna=True) or 0.0) < 1e-12:
            constant_cols.append(col)
    protected_quality_cols = {"__missing_ratio", "__row_quality"}
    constant_cols_to_drop = [c for c in constant_cols if c not in protected_quality_cols]
    if constant_cols_to_drop and len(constant_cols_to_drop) < X_final.shape[1]:
        X_final = X_final.drop(columns=constant_cols_to_drop)
        notes.append(
            f"{len(constant_cols_to_drop)} feature(s) constante(s)/quase constante(s) removida(s) após a imputação."
        )

    if X_final.empty or y_df.empty:
        raise ValueError("Não restaram dados suficientes para treinar após o pré-processamento.")

    # Pesos: penalização não linear. Amostras com maior ausência recebem menor peso no treinamento.
    sample_weight = np.power(row_quality.to_numpy(dtype=float), 2.0)
    sample_weight = np.clip(sample_weight, min_confidence_clip, 1.00).astype(np.float32)

    diagnostics = {
        "regime_qualidade_dados": quality_regime,
        "limite_nan_linha_aplicado": float(effective_row_limit),
        "limite_nan_coluna_aplicado": float(effective_col_limit),
        "linhas_originais": float(original_rows),
        "linhas_com_saida_ausente_removidas": float(n_target_missing),
        "linhas_por_excesso_nan_removidas": float(n_rows_dropped if applied_row_filter else 0),
        "linhas_finais_treino_validacao": float(len(X_final)),
        "percentual_nan_entradas_original": original_input_missing_ratio,
        "percentual_nan_variaveis_selecionadas_original": original_selected_missing_ratio,
        "percentual_medio_nan_linhas_usadas": float(row_missing_ratio.mean()) if len(row_missing_ratio) else 0.0,
        "qualidade_media_amostras": float(row_quality.mean()) if len(row_quality) else 0.0,
        "peso_medio_treino": float(sample_weight.mean()) if len(sample_weight) else 0.0,
        "colunas_entrada_originais": float(len(in_cols)),
        "colunas_entrada_removidas_por_nan": float(len(high_missing_cols) if applied_col_filter else 0),
        "features_constantes_removidas": float(len(constant_cols_to_drop)),
        "features_finais_do_modelo": float(X_final.shape[1]),
    }

    return X_final, y_df.reset_index(drop=True), imputation_report, notes, diagnostics, sample_weight

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    y_true, y_pred: (n, ) ou (n, k).
    Retorna métricas agregadas (média entre saídas quando k>1).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # garante 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    eps = 1e-9
    rmses, maes, r2s, mapes, smapes, wmapes, rs = [], [], [], [], [], [], []

    for j in range(y_true.shape[1]):
        yt = y_true[:, j].reshape(-1)
        yp = y_pred[:, j].reshape(-1)

        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))

        if yt.std() == 0 or yp.std() == 0:
            R = float("nan")
        else:
            R = float(np.corrcoef(yt, yp)[0, 1])

        mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))))
        smape = float(np.mean(2 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + eps)))
        wmape = float(np.sum(np.abs(yt - yp)) / (np.sum(np.abs(yt)) + eps))

        rmses.append(rmse); maes.append(mae); r2s.append(r2); rs.append(R)
        mapes.append(mape); smapes.append(smape); wmapes.append(wmape)

    return {
        "RMSE": float(np.mean(rmses)),
        "MAE": float(np.mean(maes)),
        "R2": float(np.mean(r2s)),
        "R": float(np.nanmean(rs)),
        "MAPE": float(np.mean(mapes)),
        "SMAPE": float(np.mean(smapes)),
        "WMAPE": float(np.mean(wmapes)),
    }


def _safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def _compute_feature_target_signal(X_df: pd.DataFrame, y_df: pd.DataFrame) -> Dict[str, float]:
    """Resumo simples de sinal preditivo linear após o pré-processamento."""
    try:
        y_ref = y_df.iloc[:, 0].astype(float) if y_df.shape[1] == 1 else y_df.mean(axis=1).astype(float)
        corrs = []
        for col in X_df.columns:
            xs = pd.to_numeric(X_df[col], errors="coerce")
            if xs.std(skipna=True) == 0 or y_ref.std(skipna=True) == 0:
                continue
            corr = xs.corr(y_ref)
            if pd.notna(corr) and np.isfinite(corr):
                corrs.append(abs(float(corr)))
        if not corrs:
            return {"max_abs_corr_feature_target": 0.0, "mean_abs_corr_feature_target": 0.0}
        return {
            "max_abs_corr_feature_target": float(np.max(corrs)),
            "mean_abs_corr_feature_target": float(np.mean(corrs)),
        }
    except Exception:
        return {"max_abs_corr_feature_target": 0.0, "mean_abs_corr_feature_target": 0.0}


def _build_modelability_assessment(metrics: Dict[str, float], diagnostics: Dict[str, float]) -> Dict[str, object]:
    r2 = _safe_float(metrics.get("R2"), 0.0)
    r = abs(_safe_float(metrics.get("R"), 0.0))
    nan_ratio = _safe_float(diagnostics.get("percentual_nan_entradas_original"), 0.0)
    confidence = _safe_float(diagnostics.get("qualidade_media_amostras"), 0.0)
    rows_final = _safe_float(diagnostics.get("linhas_finais_treino_validacao"), 0.0)
    rows_original = max(_safe_float(diagnostics.get("linhas_originais"), rows_final), 1.0)
    retained_ratio = rows_final / rows_original
    baseline_r2 = _safe_float(diagnostics.get("baseline_r2"), 0.0)
    improvement = r2 - baseline_r2
    max_corr = _safe_float(diagnostics.get("max_abs_corr_feature_target"), 0.0)

    score = 0.0
    score += np.clip((r2 + 0.10) / 1.10, 0.0, 1.0) * 35.0
    score += np.clip(r, 0.0, 1.0) * 20.0
    score += np.clip(confidence, 0.0, 1.0) * 15.0
    score += np.clip(1.0 - nan_ratio, 0.0, 1.0) * 10.0
    score += np.clip(retained_ratio, 0.0, 1.0) * 10.0
    score += np.clip((improvement + 0.05) / 0.30, 0.0, 1.0) * 5.0
    score += np.clip(max_corr / 0.60, 0.0, 1.0) * 5.0
    score = float(np.clip(score, 0.0, 100.0))

    if r2 < 0.05 or r < 0.25:
        classe = "Ruim"
        decisao = "Não recomendado para uso preditivo sem melhorar a base ou rever variáveis."
    elif score >= 90 and r2 >= 0.80 and r >= 0.90:
        classe = "Boa"
        decisao = "Modelo adequado para testes."
    elif score >= 50 and r2 >= 0.30:
        classe = "Moderado"
        decisao = "Modelo utilizável com cautela; recomenda-se melhorar cobertura e qualidade dos dados."
    else:
        classe = "Baixa/Moderado"
        decisao = "Resultado instável; recomenda-se validar com novos dados e/ou novas variáveis explicativas."

    fatores = []
    if nan_ratio >= 0.35:
        fatores.append("alta taxa de valores ausentes nas variáveis de entrada")
    if retained_ratio < 0.50:
        fatores.append("grande perda de linhas após filtros")
    if confidence < 0.75:
        fatores.append("qualidade média reduzida após o pré-processamento")
    if max_corr < 0.15:
        fatores.append("baixa associação linear entre variáveis de entrada e variável de saída")
    if r2 <= baseline_r2 + 0.02:
        fatores.append("ganho pequeno em relação ao baseline pela média")
    if not fatores:
        fatores.append("dados com sinal preditivo consistente")

    return {
        "score": score,
        "classe": classe,
        "decisao": decisao,
        "fatores": fatores,
    }

def _build_model(n_features: int, n_outputs: int, n_neurons: int, learning_rate: float) -> keras.Model:
    """MLP com regularização para bases com valores imputados e possíveis outliers."""
    inputs = keras.Input(shape=(n_features,), name="features")

    x = layers.Dense(
        max(8, n_neurons),
        kernel_regularizer=keras.regularizers.l2(1e-5),
        name="dense_1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.Dropout(0.08, name="dropout_1")(x)

    x = layers.Dense(
        max(8, n_neurons // 2),
        kernel_regularizer=keras.regularizers.l2(1e-5),
        name="dense_2",
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)

    outputs = layers.Dense(n_outputs, name="output")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_regressor_robusto")
    opt = keras.optimizers.Adam(learning_rate=float(learning_rate), clipnorm=1.0)

    # Huber é menos sensível a outliers e a ruídos criados por imputação do que MSE puro.
    model.compile(optimizer=opt, loss=keras.losses.Huber(delta=1.0), metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    return model


# ===============================
# Estado
# ===============================

def _ensure_state():
    st.session_state.setdefault("page", "setup")  # setup | results
    st.session_state.setdefault("df_raw", None)
    st.session_state.setdefault("cols", [])
    st.session_state.setdefault("in_cols", [])
    st.session_state.setdefault("out_cols", [])
    st.session_state.setdefault("uploader_version", 0)
    st.session_state.setdefault("trained", False)
    st.session_state.setdefault("training", False)
    st.session_state.setdefault("train_requested", False)
    st.session_state.setdefault("run_training_now", False)
    st.session_state.setdefault("artifacts", {})  # paths + bytes
    st.session_state.setdefault("metrics", None)
    st.session_state.setdefault("val_table", None)
    st.session_state.setdefault("history", None)
    st.session_state.setdefault("imputation_report", None)
    st.session_state.setdefault("imputation_notes", None)
    st.session_state.setdefault("data_diagnostics", None)
    st.session_state.setdefault("modelability_assessment", None)
    st.session_state.setdefault("train_epoch_current", 0)
    st.session_state.setdefault("train_epoch_total", 0)
    st.session_state.setdefault("train_loss", None)
    st.session_state.setdefault("train_val_loss", None)
    st.session_state.setdefault("train_note", "")
    st.session_state.setdefault("training_error", None)
    st.session_state.setdefault("train_cancel_requested", False)
    st.session_state.setdefault("train_thread", None)
    st.session_state.setdefault("train_queue", None)
    st.session_state.setdefault("train_cancel_event", None)

def _go(page: str):
    st.session_state["page"] = page

def _set_in(col: str):
    # Exclusividade: se marcou Entrada, desmarca Saída
    out_key = f"out__{col}"
    if st.session_state.get(f"in__{col}", False):
        st.session_state[out_key] = False

def _set_out(col: str):
    # Se marcou Saída, desmarca Entrada
    in_key = f"in__{col}"
    if st.session_state.get(f"out__{col}", False):
        st.session_state[in_key] = False

        # NOVO: garante apenas uma saída selecionada
        for key in list(st.session_state.keys()):
            if key.startswith("out__") and key != f"out__{col}":
                st.session_state[key] = False


# ===============================
# Estilo (mockup "cards")
# ===============================

st.set_page_config(page_title="Treinamento de Modelo", layout="wide")

st.markdown(
    """
    <style>
      /* tira elementos do Streamlit */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* força visual claro mesmo com tema dark do usuário */
      html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        background: #eef1f6 !important;
        color: #111827 !important;
      }

      /* centraliza e controla largura do conteúdo */
      .block-container {
        max-width: 1120px !important;
        padding-top: 52px !important;
        padding-bottom: 52px !important;
      }

      /* títulos */
      .tcc-h1 {text-align:center; font-size: 52px; font-weight: 800; margin: 0 0 2px 0; color:#0f172a;}
      .tcc-h2 {text-align:center; font-size: 26px; font-weight: 500; margin: 0 0 28px 0; color:#6b7280;}

      /* aplica estilo de card direto nos containers com key */
      .st-key-uploader_card > div,
      .st-key-in_card > div,
      .st-key-out_card > div,
      .st-key-results_card > div {
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 18px;
        box-shadow: 0 16px 36px rgba(15,23,42,0.10);
        padding: 18px 18px 14px 18px;
      }


      .tcc-card-title {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        display:flex;
        align-items:center;
        gap:10px;
        margin: 2px 0 10px 0;
      }

      .tcc-col-title {font-size: 20px; font-weight: 800; letter-spacing: .04em; color:#0f172a; margin: 0 0 12px 0;}

      /* uploader como dropzone */
      .st-key-uploader_card div[data-testid="stFileUploader"]{
        border: 2px dashed rgba(15,23,42,0.22) !important;
        border-radius: 14px !important;
        padding: 18px !important;
        background: #f8fafc !important;
      }
      .st-key-uploader_card .stFileUploader label {display:none;}

      /* ===== UPLOADER MODERNO ===== */

        /* dropzone */
        .st-key-uploader_card div[data-testid="stFileUploader"]{
        border: 2px dashed rgba(15,23,42,0.25) !important;
        border-radius: 14px !important;
        padding: 22px !important;
        background: linear-gradient(180deg,#0b1220,#0f172a) !important;
    }

        /* texto drag and drop */
        .st-key-uploader_card div[data-testid="stFileUploader"] section,
        .st-key-uploader_card div[data-testid="stFileUploader"] section * {
            color: #ffffff !important;
            font-weight: 500;
        }

        /* botão Browse Files */
        .st-key-uploader_card div[data-testid="stFileUploader"] button {
            background: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid rgba(15,23,42,0.15) !important;
            border-radius: 10px !important;
            padding: 8px 18px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: all 0.15s ease;
        }

        /* hover do botão */
        .st-key-uploader_card div[data-testid="stFileUploader"] button:hover {
            background: #f8fafc !important;
            border: 1px solid rgba(15,23,42,0.25) !important;
            transform: translateY(-1px);
        }

        /* botão pressionado */
        .st-key-uploader_card div[data-testid="stFileUploader"] button:active {
            transform: translateY(0px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        }

        /* arquivo carregado */
        .st-key-uploader_card div[data-testid="stFileUploader"] ul,
        .st-key-uploader_card div[data-testid="stFileUploader"] li,
        .st-key-uploader_card div[data-testid="stFileUploader"] li * {
            color: #0f172a !important;
            font-weight: 500;
}

      /* === FileUploader (corrige contraste) ===
         - Texto "Drag and drop..." branco
         - Nome do arquivo + tamanho em preto
      */
      /* dropzone (texto de instrução) */
      .st-key-uploader_card div[data-testid="stFileUploader"] section,
      .st-key-uploader_card div[data-testid="stFileUploader"] section * {
        color: #ffffff !important;
      }
      .st-key-uploader_card div[data-testid="stFileUploader"] section {
        background: #0b1220 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
      }

      /* linha do arquivo enviado */
      .st-key-uploader_card div[data-testid="stFileUploader"] ul,
      .st-key-uploader_card div[data-testid="stFileUploader"] li,
      .st-key-uploader_card div[data-testid="stFileUploader"] li * {
        color: #ffffff !important;
        opacity: 1 !important;
      }

      /* garante que o "X" de remover fique visível */
      .st-key-uploader_card div[data-testid="stFileUploader"] button,
      .st-key-uploader_card div[data-testid="stFileUploader"] button * {
        color: #0f172a !important;
        opacity: 1 !important;
      }

      /* checkboxes */
      div[data-testid="stCheckbox"] label,
      div[data-testid="stCheckbox"] label * {
        font-size: 20px;
        color:#0f172a !important;
        opacity: 1 !important;
      }
      div[data-testid="stCheckbox"] {padding: 4px 0;}

      /* pílulas/brackets dos checkboxes (como no mockup) */
      .st-key-in_card div[data-testid="stCheckbox"],
      .st-key-out_card div[data-testid="stCheckbox"] {
        background: #ffffff;
        border-radius: 16px;
        border: 1px solid rgba(15,23,42,0.08);
        box-shadow: 0 14px 26px rgba(15,23,42,0.10);
        padding: 10px 14px;
        margin: 10px 0;
      }

      /* texto do label dentro da pílula (força visível) */
      .st-key-in_card div[data-testid="stCheckbox"] label,
      .st-key-in_card div[data-testid="stCheckbox"] label * ,
      .st-key-out_card div[data-testid="stCheckbox"] label,
      .st-key-out_card div[data-testid="stCheckbox"] label * {
        color:#0f172a !important;
        opacity: 1 !important;
        filter: none !important;
      }

      /* botão Treinar */
      .st-key-train_area div[data-testid="stButton"] > button {
        width: 440px;
        height: 78px;
        border-radius: 16px;
        border: none !important;
        background: linear-gradient(180deg, #4aa36a, #3c8b5a) !important;
        color: white !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        box-shadow: 0 18px 35px rgba(34,197,94,0.25);
      }
      .st-key-train_area div[data-testid="stButton"] > button:disabled {
        opacity: 0.55;
      }
      /* ===== MÉTRICAS ESTILO DASHBOARD ===== */

        /* container da métrica */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 12px;
            padding: 14px 18px;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        }

        /* título da métrica (RMSE, MAE, etc) */
        div[data-testid="stMetricLabel"] {
            font-size: 14px !important;
            font-weight: 600 !important;
            color: #0f172a !important;
            letter-spacing: 0.05em;
        }

        /* valor da métrica */
        div[data-testid="stMetricValue"] {
            font-size: 34px !important;
            font-weight: 800 !important;
            color: #0f172a !important;
        }

        /* remove fundo azul estranho */
        div[data-testid="stMetricValue"] > div {
            background: none !important;
        }

        /* ===== FORÇA COR DO TÍTULO DAS MÉTRICAS (LABEL) ===== */

        /* container do label */
        div[data-testid="stMetricLabel"]{
        color: #64748b !important;
        }

        /* QUALQUER elemento dentro do label (p, span, small, div etc.) */
        div[data-testid="stMetricLabel"] *{
        color: #64748b !important;
        opacity: 1 !important;
        filter: none !important;
        }

        /* (extra) algumas versões do Streamlit usam esse wrapper */
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] span,
        div[data-testid="stMetricLabel"] small{
        color: #64748b !important;
        opacity: 1 !important;
}

    

      /* ===== FIX DEFINITIVO: LABEL DAS MÉTRICAS (RESULTADOS) =====
         Algumas versões/temas aplicam cor no texto com alta especificidade e até text-fill.
         Forçamos somente dentro do card de resultados.
      */
      .st-key-results_card div[data-testid="stMetricLabel"],
      .st-key-results_card div[data-testid="stMetricLabel"] *,
      .st-key-results_card div[data-testid="stMetric"] [data-testid="stMetricLabel"],
      .st-key-results_card div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
        filter: none !important;
      }

      /* mantém os valores escuros e destacados */
      .st-key-results_card div[data-testid="stMetricValue"],
      .st-key-results_card div[data-testid="stMetricValue"] * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
      }

/* ================================
   RESULTADOS: DASHBOARD SaaS UPGRADE
   ================================ */
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
  background: radial-gradient(1200px 600px at 50% -10%, rgba(59,130,246,0.10), transparent 60%), #eef1f6 !important;
}
div[data-testid="stButton"] > button{
  background:#0b1220 !important;color:#fff !important;border-radius:12px !important;
  border:1px solid rgba(148,163,184,0.18)!important;padding:10px 18px!important;
  font-weight:700!important;font-size:15px!important;
  box-shadow:0 10px 22px rgba(15,23,42,.16);transition:all .15s ease;
}
div[data-testid="stButton"] > button:hover{
  background:#111b2e!important;transform:translateY(-1px);
  box-shadow:0 14px 28px rgba(15,23,42,.18);
}
div[data-testid="stDownloadButton"] > button{
  background:linear-gradient(180deg,#22c55e,#16a34a)!important;color:#fff!important;
  border:none!important;border-radius:12px!important;padding:10px 18px!important;
  font-weight:800!important;font-size:15px!important;
  box-shadow:0 14px 30px rgba(34,197,94,.22);
}
div[data-testid="stDownloadButton"] > button:hover{
  transform:translateY(-1px);box-shadow:0 18px 36px rgba(34,197,94,.26);
}
.st-key-results_card > div{
  background:rgba(255,255,255,.92)!important;backdrop-filter:blur(8px);
  border:1px solid rgba(15,23,42,.08);border-radius:18px!important;
  box-shadow:0 18px 42px rgba(15,23,42,.12)!important;
}


      /* ===== PARAMS PAGE ===== */
      .tcc-top-back-wrap {margin-top: 10px; margin-bottom: 10px;}
      .tcc-section-shell {
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(15,23,42,0.06);
        border-radius: 22px;
        box-shadow: 0 20px 45px rgba(15,23,42,0.10);
        padding: 14px 14px 18px 14px;
      }
      .tcc-param-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 20px;
        padding: 16px 18px 18px 18px;
        box-shadow: 0 10px 24px rgba(148,163,184,0.14);
        min-height: 118px;
        margin-bottom: 18px;
      }
      .tcc-param-label {
        display: inline-block;
        width: 100%;
        background: linear-gradient(180deg, rgba(243,246,251,0.98) 0%, rgba(236,241,248,0.98) 100%);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 14px;
        padding: 14px 18px;
        font-size: 15px;
        font-weight: 700;
        color:#1e3357;
        margin-bottom: 16px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
      }
      .tcc-helper {
        font-size: 14px; color:#64748b; margin-top: 8px;
      }
      .st-key-back_params div[data-testid="stButton"] > button {
        width: 170px; min-height: 62px;
        border-radius: 16px !important;
        background: linear-gradient(145deg,#071a3d,#153b7a) !important;
        color: #f8fbff !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        box-shadow: 0 14px 28px rgba(15,23,42,.18) !important;
        font-weight: 800 !important; font-size: 17px !important;
      }
      .st-key-back_params div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        background: linear-gradient(145deg,#0a2452,#1a4a97) !important;
      }
      .st-key-train_area div[data-testid="stButton"] > button {
        width: 360px;
        height: 72px;
        border-radius: 16px;
        border: none !important;
        background: linear-gradient(135deg, #8ed0a7 0%, #5cbe9b 55%, #3aa587 100%) !important;
        color: white !important;
        font-size: 22px !important;
        font-weight: 800 !important;
        box-shadow: 0 18px 35px rgba(16,185,129,0.22);
      }
      .st-key-train_area div[data-testid="stButton"] > button:hover:enabled {
        transform: translateY(-1px);
        box-shadow: 0 22px 40px rgba(16,185,129,0.26);
      }
      .st-key-train_area div[data-testid="stButton"] > button:disabled {
        background: linear-gradient(135deg, #bfdcc8 0%, #afcfbf 100%) !important;
        color: rgba(255,255,255,0.95) !important;
        box-shadow: none;
      }

      /* inputs numéricos e text_input */
      div[data-testid="stNumberInput"] label,
      div[data-testid="stTextInput"] label,
      div[data-testid="stSlider"] label {
        color:#1e3357 !important; font-weight: 700 !important;
      }
      div[data-testid="stNumberInput"] input,
      div[data-testid="stTextInput"] input {
        background: linear-gradient(180deg,#f8fbff,#f0f5fb) !important;
        color:#1e293b !important;
        border: 1px solid rgba(148,163,184,0.45) !important;
        border-radius: 14px !important;
        min-height: 54px !important;
        font-size: 18px !important;
        box-shadow: inset 0 1px 2px rgba(255,255,255,0.8), 0 2px 8px rgba(148,163,184,0.10);
      }
      div[data-testid="stNumberInput"] button {
        background: transparent !important;
        color:#64748b !important;
        border: none !important;
      }
      div[data-testid="stNumberInput"] button:hover {
        color:#1e3357 !important;
      }
      div[data-testid="stNumberInput"] input::placeholder,
      div[data-testid="stTextInput"] input::placeholder {
        color:#8aa0bf !important; opacity:1 !important;
      }

      /* slider */
      div[data-testid="stSlider"] > div[data-baseweb="slider"] > div div {
        box-shadow: none !important;
      }
      div[data-testid="stSlider"] [role="slider"] {
        background: linear-gradient(180deg,#5f6ed6,#4b52c5) !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 8px 18px rgba(79,70,229,0.28) !important;
      }
      div[data-testid="stSlider"] [data-testid="stTickBar"] {
        background: linear-gradient(90deg,#4991e6,#7b4dd9 55%, #cdd9ea) !important;
        height: 8px !important; border-radius: 999px !important;
      }
      .tcc-slider-value {
        display:inline-flex; align-items:center; justify-content:center;
        min-width: 52px; height: 42px; padding: 0 14px;
        background: linear-gradient(180deg,#5961d8,#4b52c5);
        color:#fff; border-radius: 14px; font-size: 18px; font-weight: 800;
        box-shadow: 0 12px 24px rgba(79,70,229,0.28);
        margin-bottom: 8px;
      }

      /* checkbox like image */
      .st-key-earlystop_card div[data-testid="stCheckbox"] {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(148,163,184,0.22);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 10px 24px rgba(148,163,184,0.14);
      }
      .st-key-earlystop_card div[data-testid="stCheckbox"] label,
      .st-key-earlystop_card div[data-testid="stCheckbox"] label * {
        font-size: 18px !important;
        font-weight: 700 !important;
        color:#1e3357 !important;
      }
      .st-key-earlystop_card input[type="checkbox"] {
        accent-color: #244c87;
      }

      /* caption info */
      .tcc-split-caption {
        font-size: 16px;
        color:#5b6d8a;
        font-weight: 500;
        margin-top: 8px;
      }

      /* ===== PARAMS PAGE LAYOUT FIX ===== */
      .st-key-param_shell,
      .st-key-param_shell > div {
        background: rgba(255,255,255,0.78) !important;
        border: 1px solid rgba(15,23,42,0.06) !important;
        border-radius: 24px !important;
        box-shadow: 0 20px 48px rgba(15,23,42,0.11) !important;
        padding: 26px 28px 28px 28px !important;
      }

      .st-key-card_nn,
      .st-key-card_epochs,
      .st-key-card_lr,
      .st-key-card_bs,
      .st-key-card_slider,
      .st-key-card_patience,
      .st-key-earlystop_card,
      .st-key-card_nn > div,
      .st-key-card_epochs > div,
      .st-key-card_lr > div,
      .st-key-card_bs > div,
      .st-key-card_slider > div,
      .st-key-card_patience > div,
      .st-key-earlystop_card > div {
        background: rgba(255,255,255,0.96) !important;
        border: 1px solid rgba(148,163,184,0.20) !important;
        border-radius: 20px !important;
        padding: 16px 16px 18px 16px !important;
        box-shadow: 0 10px 24px rgba(148,163,184,0.14) !important;
        margin-bottom: 18px !important;
      }

      .st-key-card_slider,
      .st-key-card_slider > div {
        min-height: 148px;
      }

      .st-key-card_nn .tcc-param-label,
      .st-key-card_epochs .tcc-param-label,
      .st-key-card_lr .tcc-param-label,
      .st-key-card_bs .tcc-param-label,
      .st-key-card_slider .tcc-param-label,
      .st-key-card_patience .tcc-param-label {
        display: block;
        width: 100%;
        background: linear-gradient(180deg,#edf2f8,#e8eef6) !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        color:#1e3357 !important;
        margin: 0 0 14px 0 !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.75) !important;
      }

      .st-key-card_nn div[data-testid="stNumberInput"],
      .st-key-card_epochs div[data-testid="stNumberInput"],
      .st-key-card_lr div[data-testid="stNumberInput"],
      .st-key-card_bs div[data-testid="stNumberInput"],
      .st-key-card_patience div[data-testid="stNumberInput"] {
        background: linear-gradient(180deg,#f8fbff,#eef4fb) !important;
        border: 1px solid rgba(148,163,184,0.24) !important;
        border-radius: 14px !important;
        padding: 10px 12px !important;
        margin-top: 0 !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.85), 0 4px 12px rgba(148,163,184,0.10) !important;
      }
      .st-key-card_nn div[data-testid="stNumberInput"] > div,
      .st-key-card_epochs div[data-testid="stNumberInput"] > div,
      .st-key-card_lr div[data-testid="stNumberInput"] > div,
      .st-key-card_bs div[data-testid="stNumberInput"] > div,
      .st-key-card_patience div[data-testid="stNumberInput"] > div {
        background: transparent !important;
        padding: 0 !important;
      }
      .st-key-card_nn div[data-testid="stNumberInput"] input,
      .st-key-card_epochs div[data-testid="stNumberInput"] input,
      .st-key-card_lr div[data-testid="stNumberInput"] input,
      .st-key-card_bs div[data-testid="stNumberInput"] input,
      .st-key-card_patience div[data-testid="stNumberInput"] input {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        min-height: 42px !important;
        font-size: 18px !important;
        color:#21324f !important;
      }

      .st-key-card_slider div[data-testid="stSlider"] {
        background: linear-gradient(180deg,#f8fbff,#eef4fb) !important;
        border: 1px solid rgba(148,163,184,0.20) !important;
        border-radius: 16px !important;
        padding: 18px 18px 12px 18px !important;
        margin-top: 0 !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.85), 0 4px 12px rgba(148,163,184,0.10) !important;
      }
      .st-key-card_slider div[data-testid="stSlider"] > div {
        padding-top: 6px !important;
        padding-bottom: 2px !important;
      }
      .st-key-card_slider [data-testid="stTickBar"] {
        background: linear-gradient(90deg,#4f9ef7 0%, #7257e3 55%, #d6deea 100%) !important;
        height: 10px !important;
        border-radius: 999px !important;
        margin-top: 6px !important;
      }
      .st-key-card_slider [data-baseweb="slider"] > div > div:first-child {
        height: 10px !important;
        border-radius: 999px !important;
        background: linear-gradient(90deg,#4f9ef7 0%, #7257e3 55%, #d6deea 100%) !important;
      }
      .st-key-card_slider [role="slider"] {
        width: 22px !important;
        height: 22px !important;
        border-radius: 999px !important;
        background: radial-gradient(circle at 30% 30%, #7faaff 0%, #5661dc 70%, #4a52c9 100%) !important;
        border: 3px solid #ffffff !important;
        box-shadow: 0 10px 18px rgba(79,70,229,0.24) !important;
      }
      .tcc-slider-value {
        display:inline-flex; align-items:center; justify-content:center;
        min-width: 54px; height: 42px; padding: 0 14px;
        background: linear-gradient(180deg,#5f69df,#4d56cc) !important;
        color:#fff !important; border-radius: 14px !important;
        font-size: 20px !important; font-weight: 800 !important;
        box-shadow: 0 12px 24px rgba(79,70,229,0.24) !important;
        margin-top: 10px !important;
      }

      .st-key-earlystop_card div[data-testid="stCheckbox"] {
        background: linear-gradient(180deg,#f8fbff,#eef4fb) !important;
        border: 1px solid rgba(148,163,184,0.20) !important;
        border-radius: 16px !important;
        padding: 14px 16px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.85), 0 4px 12px rgba(148,163,184,0.10) !important;
      }
      .st-key-earlystop_card div[data-testid="stCheckbox"] label,
      .st-key-earlystop_card div[data-testid="stCheckbox"] label * {
        font-size: 18px !important;
        font-weight: 700 !important;
        color:#1e3357 !important;
      }

      .tcc-split-caption {
        font-size: 16px;
        color:#5b6d8a;
        font-weight: 500;
        margin-top: 6px;
        margin-bottom: 12px;
      }

    /* esconde apenas o botão X da linha do arquivo carregado, sem afetar o Browse files */
        div[data-testid="stFileUploader"] ul button[aria-label*="Remove"],
        div[data-testid="stFileUploader"] ul button[title*="Remove"],
        div[data-testid="stFileUploader"] li button[aria-label*="Remove"],
        div[data-testid="stFileUploader"] li button[title*="Remove"],
        div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] {
            display: none !important;
        }


</style>
    """,
    unsafe_allow_html=True,
)

_ensure_state()


# ===============================
# Página 1 – Setup (mockup)
# ===============================

def _reset_dataset_state():
    """Limpa todos os dados carregados e seleções."""
    keys_to_clear = [
        "df_raw",
        "cols",
        "in_cols",
        "out_cols",
        "trained",
        "artifacts",
        "metrics",
        "val_table",
        "history",
        "imputation_report",
        "imputation_notes",
        "data_diagnostics",
        "modelability_assessment",
        "last_upload_name",
        "train_epoch_current",
        "train_epoch_total",
        "train_loss",
        "train_val_loss"
    ]

    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]

    # limpa checkboxes criados dinamicamente
    for k in list(st.session_state.keys()):
        if k.startswith("in__") or k.startswith("out__"):
            del st.session_state[k]

    st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1

def _scroll_top():
    """Força a página voltar ao topo após reset."""
    st.markdown(
        """
        <script>
        window.scrollTo(0,0);
        </script>
        """,
        unsafe_allow_html=True
    )


def render_setup():
    st.markdown('<div class="tcc-h1">Treinamento de Modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="tcc-h2">Configuração de variáveis</div>', unsafe_allow_html=True)

    # Upload (card)
    with st.container(key="uploader_card"):
        st.markdown('<div class="tcc-card-title">📄 Upload da Base (.csv)</div>', unsafe_allow_html=True)
        upl = st.file_uploader(
        "",
        type=["csv", "xlsx", "xls"],
        key=f"upl_main_{st.session_state['uploader_version']}"
        )

        loaded_name = st.session_state.get("last_upload_name")
        if st.session_state.get("df_raw") is not None and loaded_name:
            st.caption(f"Base em memória: {loaded_name}")
            if st.button("🗑 Remover base carregada", key="btn_clear_loaded_base", use_container_width=True):
                _reset_dataset_state()
                st.session_state["had_upload"] = False
                _scroll_top()
                st.rerun()

    # Leitura do arquivo (somente quando houver novo upload)
    if upl is not None:
        last_name = st.session_state.get("last_upload_name")
        if (st.session_state.get("df_raw") is None) or (last_name != getattr(upl, "name", None)):
            try:
                df = _read_any(upl)
                st.session_state["df_raw"] = df
                st.session_state["cols"] = list(df.columns)
                st.session_state["trained"] = False
                st.session_state["metrics"] = None
                st.session_state["val_table"] = None
                st.session_state["history"] = None
                st.session_state["imputation_report"] = None
                st.session_state["imputation_notes"] = None
                st.session_state["data_diagnostics"] = None
                st.session_state["modelability_assessment"] = None
                st.session_state["artifacts"] = {}
                st.session_state["last_upload_name"] = getattr(upl, "name", None)
                st.session_state["had_upload"] = True
                st.rerun()
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")
                st.session_state["df_raw"] = None
                st.session_state["cols"] = []
                st.session_state["had_upload"] = False

    cols = st.session_state.get("cols", [])
    df_for_labels: Optional[pd.DataFrame] = st.session_state.get("df_raw")

    def _pretty_label(col_name: str, idx: int) -> str:
        """Formato pedido: 'Coluna X: <primeiro objeto na primeira posição da coluna>'."""
        first_val = "—"
        if isinstance(df_for_labels, pd.DataFrame) and col_name in df_for_labels.columns and len(df_for_labels) > 0:
            try:
                # pega o primeiro valor não-nulo (se a primeira linha estiver vazia)
                s_col = df_for_labels[col_name]
                first_idx = s_col.first_valid_index()
                v = s_col.loc[first_idx] if first_idx is not None else s_col.iloc[0]
                if not pd.isna(v):
                    # deixa compacto e legível
                    s = str(v)
                    first_val = (s[:40] + "…") if len(s) > 40 else s
            except Exception:
                first_val = "—"
        return f"Coluna {idx + 1}: {first_val}"

    labels = {c: _pretty_label(c, i) for i, c in enumerate(cols)}

    # Entradas / Saídas (cards lado a lado)
    left, right = st.columns([1, 1], gap="large")

    def _chunks(seq: List[str], n: int = 3) -> List[List[str]]:
        return [seq[i:i+n] for i in range(0, len(seq), n)]

    with left:
        with st.container(key="in_card"):
            st.markdown('<div class="tcc-col-title">ENTRADAS</div>', unsafe_allow_html=True)
            if cols:
                for c in cols:
                    st.session_state.setdefault(f"in__{c}", False)
                    st.session_state.setdefault(f"out__{c}", False)
                for blk in _chunks(cols, 3):
                    for c in blk:
                        disabled = bool(st.session_state.get(f"out__{c}", False))
                        st.checkbox(labels.get(c, str(c)), key=f"in__{c}", disabled=disabled, on_change=_set_in, args=(c,))
                    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            else:
                st.caption("Faça upload de um .csv/.xlsx para listar as variáveis.")

    with right:
        with st.container(key="out_card"):
            st.markdown('<div class="tcc-col-title">SAÍDAS</div>', unsafe_allow_html=True)
            if cols:
                for blk in _chunks(cols, 3):
                    for c in blk:
                        disabled = bool(st.session_state.get(f"in__{c}", False))
                        st.checkbox(labels.get(c, str(c)), key=f"out__{c}", disabled=disabled, on_change=_set_out, args=(c,))
                    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
            else:
                st.caption("Faça upload de um .csv/.xlsx para listar as variáveis.")

    # Seleções consolidadas
    if cols:
        st.session_state["in_cols"] = [c for c in cols if st.session_state.get(f"in__{c}", False)]
        st.session_state["out_cols"] = [c for c in cols if st.session_state.get(f"out__{c}", False)]

    # Botão para próxima etapa
    st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
    can_continue = bool(st.session_state.get("df_raw") is not None) and len(st.session_state.get("in_cols", [])) >= 1 and len(st.session_state.get("out_cols", [])) >= 1
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        with st.container(key="train_area"):
            if st.button("⚙️  Definir Parâmetros", type="primary", disabled=not can_continue):
                _go("params")
                st.rerun()


# ===============================
# Página 2 – Parâmetros
# ===============================

st.markdown("""
<style>
/* ===== PARAMS PAGE REWORK ===== */
.st-key-back_params button {
  background: transparent !important;
  color: #17345f !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  min-height: auto !important;
  font-weight: 700 !important;
}
            
.stNumberInput div[data-baseweb="input"] input {
    color: #e6edf7 !important;
    font-weight: 600 !important;
}

.st-key-back_params button:hover {
  background: transparent !important;
  color: #0f2747 !important;
}
.st-key-param_shell,
.st-key-param_shell > div {
  background: rgba(255,255,255,0.68) !important;
  border: 1px solid rgba(180,190,205,0.30) !important;
  border-radius: 24px !important;
  box-shadow: 0 12px 35px rgba(27,39,67,0.12) !important;
  padding: 16px !important;
}
.st-key-params_left_section,
.st-key-params_right_section,
.st-key-params_left_section > div,
.st-key-params_right_section > div {
  background: #f8fafc !important;
  border: 1px solid #dde4ee !important;
  border-radius: 20px !important;
  padding: 14px !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.65) !important;
  height: 100%;
}
.tcc-section-title {
  font-size: 1rem;
  font-weight: 800;
  color: #17345f;
  margin: 0 0 1rem 0;
  padding: 0.75rem 0.9rem;
  background: #e9eef5;
  border: 1px solid #d8e0eb;
  border-radius: 14px;
}
.st-key-card_nn,
.st-key-card_epochs,
.st-key-card_lr,
.st-key-card_bs,
.st-key-card_slider,
.st-key-card_patience,
.st-key-earlystop_card,
.st-key-card_nn > div,
.st-key-card_epochs > div,
.st-key-card_lr > div,
.st-key-card_bs > div,
.st-key-card_slider > div,
.st-key-card_patience > div,
.st-key-earlystop_card > div {
  background: #ffffff !important;
  border: 1px solid #e1e7f0 !important;
  border-radius: 16px !important;
  padding: 0.95rem !important;
  margin-bottom: 0.9rem !important;
  box-shadow: 0 6px 16px rgba(28, 41, 61, 0.06) !important;
  min-height: auto !important;
}
.st-key-card_nn .tcc-param-label,
.st-key-card_epochs .tcc-param-label,
.st-key-card_lr .tcc-param-label,
.st-key-card_bs .tcc-param-label,
.st-key-card_slider .tcc-param-label,
.st-key-card_patience .tcc-param-label {
  display: block;
  width: 100%;
  background: none !important;
  border: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
  font-size: 0.92rem !important;
  font-weight: 700 !important;
  color:#26466f !important;
  margin: 0 0 0.55rem 0 !important;
  box-shadow: none !important;
}
.tcc-param-hint {
  font-size: 0.80rem;
  color: #72829a;
  margin-top: 0.45rem;
}
.tcc-slider-result {
  margin-top: 0.65rem;
  padding: 0.6rem 0.8rem;
  border-radius: 12px;
  background: #eef3f9;
  border: 1px solid #d8e2ee;
  font-size: 0.92rem;
  font-weight: 700;
  color: #1e416d;
  text-align: center;
}
.st-key-card_slider div[data-testid="stSlider"] {
  background: linear-gradient(180deg,#f8fbff,#eef4fb) !important;
  border: 1px solid rgba(148,163,184,0.20) !important;
  border-radius: 16px !important;
  padding: 14px 16px 10px 16px !important;
  margin-top: 0 !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.85), 0 4px 12px rgba(148,163,184,0.10) !important;
}
.st-key-earlystop_card .tcc-param-hint { margin-top: 0.3rem; }
.st-key-train_area button {
  min-height: 3rem !important;
  border-radius: 14px !important;
  font-weight: 700 !important;
}

.tcc-train-live-card {
  margin-top: 0.95rem;
  padding: 0.95rem 1rem;
  background: #ffffff;
  border: 1px solid #dbe4ef;
  border-radius: 16px;
  box-shadow: 0 6px 16px rgba(28, 41, 61, 0.06);
}
.tcc-train-live-title {
  font-size: 0.9rem;
  font-weight: 800;
  color: #17345f;
  margin-bottom: 0.45rem;
}
.tcc-train-live-row {
  font-size: 0.92rem;
  color: #36537a;
  font-weight: 700;
  margin-top: 0.55rem;
}
.tcc-train-live-loss {
  font-size: 0.9rem;
  color: #5f7493;
  margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


def _cleanup_training_runtime():
    st.session_state["train_thread"] = None
    st.session_state["train_queue"] = None
    st.session_state["train_cancel_event"] = None
    st.session_state["run_training_now"] = False
    st.session_state["train_requested"] = False


def _drain_training_queue():
    q = st.session_state.get("train_queue")
    if q is None:
        return

    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break

        msg_type = msg.get("type")
        if msg_type == "progress":
            st.session_state["train_epoch_current"] = int(msg.get("epoch_current", 0))
            st.session_state["train_epoch_total"] = int(msg.get("epoch_total", 0))
            st.session_state["train_loss"] = msg.get("loss")
            st.session_state["train_val_loss"] = msg.get("val_loss")
            st.session_state["train_note"] = msg.get("note", "")
        elif msg_type == "done":
            st.session_state["training_error"] = None
            st.session_state["trained"] = True
            st.session_state["metrics"] = msg.get("metrics")
            st.session_state["val_table"] = msg.get("val_table")
            st.session_state["history"] = msg.get("history")
            st.session_state["imputation_report"] = msg.get("imputation_report")
            st.session_state["imputation_notes"] = msg.get("imputation_notes")
            st.session_state["data_diagnostics"] = msg.get("data_diagnostics")
            st.session_state["modelability_assessment"] = msg.get("modelability_assessment")
            st.session_state["artifacts"] = msg.get("artifacts")
            st.session_state["training"] = False
            st.session_state["train_note"] = msg.get("note", "Treinamento concluído.")
            _cleanup_training_runtime()
            _go("results")
        elif msg_type == "cancelled":
            st.session_state["training"] = False
            st.session_state["train_note"] = msg.get("note", "Treinamento cancelado.")
            st.session_state["history"] = msg.get("history")
            st.session_state["train_epoch_current"] = int(msg.get("epoch_current", 0))
            st.session_state["train_epoch_total"] = int(msg.get("epoch_total", 0))
            st.session_state["train_loss"] = msg.get("loss")
            st.session_state["train_val_loss"] = msg.get("val_loss")
            _cleanup_training_runtime()
        elif msg_type == "error":
            st.session_state["training"] = False
            st.session_state["training_error"] = msg.get("message", "Erro desconhecido durante o treino.")
            _cleanup_training_runtime()


def _render_live_training(progress_wrap, status_wrap):
    epoch_now = int(st.session_state.get("train_epoch_current", 0) or 0)
    epoch_total = int(st.session_state.get("train_epoch_total", 0) or 0)
    loss = st.session_state.get("train_loss")
    val_loss = st.session_state.get("train_val_loss")
    note = st.session_state.get("train_note", "")

    progress_value = 0.0
    if epoch_total > 0:
        progress_value = min(1.0, epoch_now / epoch_total)

    progress_wrap.progress(progress_value)

    loss_html = f'<div class="tcc-train-live-loss">Loss: <strong>{loss:.6f}</strong></div>' if loss is not None else ''
    val_loss_html = f'<div class="tcc-train-live-loss">Val loss: <strong>{val_loss:.6f}</strong></div>' if val_loss is not None else ''
    note_html = f'<div class="tcc-train-live-note">{note}</div>' if note else ''
    card_html = (
        f'<div class="tcc-train-live-card">'
        f'<div class="tcc-train-live-title">Progresso do treinamento</div>'
        f'<div class="tcc-train-live-row">Épocas: {epoch_now}/{epoch_total}</div>'
        f'{loss_html}'
        f'{val_loss_html}'
        f'{note_html}'
        f'</div>'
    )
    status_wrap.markdown(card_html, unsafe_allow_html=True)


def render_params():
    params = st.session_state.setdefault("train_params", {})

    st.markdown('<div class="tcc-h1">Parâmetros de Treinamento</div>', unsafe_allow_html=True)
    st.markdown('<div class="tcc-h2">Defina os hiperparâmetros antes de iniciar o treino</div>', unsafe_allow_html=True)

    with st.container(key="back_params"):
        if st.button("← Voltar para variáveis"):
            _go("setup")
            st.rerun()

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    with st.container(key="param_shell"):
        left, right = st.columns(2, gap="large")

        with left:
            with st.container(key="params_left_section"):
                st.markdown('<div class="tcc-section-title">Parâmetros principais</div>', unsafe_allow_html=True)

                with st.container(key="card_nn"):
                    st.markdown('<div class="tcc-param-label">Neurônios da camada 1</div>', unsafe_allow_html=True)
                    n_neurons = st.number_input(
                        "Quantidade de neurônios da camada 1",
                        min_value=1,
                        step=1,
                        value=int(params.get("n_neurons") or 10),
                        key="ui_n_neurons",
                        label_visibility="collapsed",
                        placeholder="Ex.: 10",
                    )
                    st.markdown('<div class="tcc-param-hint">Quantidade de neurônios usada na primeira camada densa.</div>', unsafe_allow_html=True)

                with st.container(key="card_epochs"):
                    st.markdown('<div class="tcc-param-label">Máximo de épocas</div>', unsafe_allow_html=True)
                    max_epochs = st.number_input(
                        "Quantidade máxima de épocas",
                        min_value=1,
                        step=1,
                        value=int(params.get("max_epochs") or 50),
                        key="ui_max_epochs",
                        label_visibility="collapsed",
                        placeholder="Ex.: 50",
                    )
                    st.markdown('<div class="tcc-param-hint">Número máximo de ciclos de treinamento.</div>', unsafe_allow_html=True)

                with st.container(key="card_lr"):
                    st.markdown('<div class="tcc-param-label">Learning rate</div>', unsafe_allow_html=True)
                    learning_rate = st.number_input(
                        "Learning Rate",
                        min_value=0.000001,
                        step=0.0001,
                        format="%.6f",
                        value=float(params.get("learning_rate") or 0.001),
                        key="ui_learning_rate",
                        label_visibility="collapsed",
                        placeholder="Ex.: 0.001000",
                    )
                    st.markdown('<div class="tcc-param-hint">Controla o tamanho do ajuste dos pesos a cada atualização.</div>', unsafe_allow_html=True)

                with st.container(key="card_bs"):
                    st.markdown('<div class="tcc-param-label">Batch size</div>', unsafe_allow_html=True)
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        step=1,
                        value=int(params.get("batch_size") or 32),
                        key="ui_batch_size",
                        label_visibility="collapsed",
                        placeholder="Ex.: 32",
                    )
                    st.markdown('<div class="tcc-param-hint">Quantidade de amostras processadas por lote.</div>', unsafe_allow_html=True)

        with right:
            with st.container(key="params_right_section"):
                st.markdown('<div class="tcc-section-title">Estratégias de treino</div>', unsafe_allow_html=True)

                with st.container(key="card_slider"):
                    st.markdown('<div class="tcc-param-label">Separação treino vs validação</div>', unsafe_allow_html=True)
                    train_ratio = st.slider(
                        "Separação TREINO vs VALIDAÇÃO (%)",
                        min_value=50,
                        max_value=95,
                        value=int(params.get("train_ratio") or 70),
                        key="ui_train_ratio",
                        label_visibility="collapsed",
                    )
                    st.markdown(
                        f'<div class="tcc-slider-result">Treino: {train_ratio}% &nbsp;&nbsp;•&nbsp;&nbsp; Validação: {100-train_ratio}%</div>',
                        unsafe_allow_html=True,
                    )

                with st.container(key="earlystop_card"):
                    use_early_stopping = st.checkbox(
                        "Habilitar EarlyStopping",
                        value=bool(params.get("use_early_stopping", False)),
                        key="ui_use_early_stopping",
                    )
                    st.markdown('<div class="tcc-param-hint">Interrompe o treino automaticamente quando não houver melhora.</div>', unsafe_allow_html=True)

                if use_early_stopping:
                    with st.container(key="card_patience"):
                        st.markdown('<div class="tcc-param-label">Épocas sem melhora</div>', unsafe_allow_html=True)
                        early_stopping_patience = st.number_input(
                            "Épocas do EarlyStopping",
                            min_value=1,
                            step=1,
                            value=int(params.get("early_stopping_patience") or 30),
                            key="ui_early_stopping_patience",
                            label_visibility="collapsed",
                            placeholder="Ex.: 30",
                        )
                        st.markdown('<div class="tcc-param-hint">Quantidade de épocas aguardadas antes de parar o treino.</div>', unsafe_allow_html=True)
                else:
                    early_stopping_patience = None

    params.update({
        "n_neurons": int(n_neurons),
        "max_epochs": int(max_epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "train_ratio": int(train_ratio),
        "use_early_stopping": bool(use_early_stopping),
        "early_stopping_patience": int(early_stopping_patience) if use_early_stopping else None,
    })

    has_required = all([
        params.get("n_neurons"),
        params.get("max_epochs"),
        params.get("learning_rate"),
        params.get("batch_size"),
        params.get("train_ratio"),
        (not params.get("use_early_stopping")) or params.get("early_stopping_patience"),
    ])

    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 1.4, 1.2])
    with c2:
        with st.container(key="train_area"):
            _render_training_panel(has_required=has_required, params=params)

    if st.session_state.get("training", False):
        time.sleep(3)
        st.rerun()



def _render_training_panel_impl(has_required, params):
    _drain_training_queue()

    if st.session_state.get("page") == "results":
        st.rerun()

    if st.session_state.get("training_error"):
        st.error(st.session_state.get("training_error"))

    btn_label = "⏳  Treinando..." if st.session_state.get("training", False) else "🚀 Treinar Modelo"
    if st.button(
        btn_label,
        type="primary",
        disabled=(not has_required) or st.session_state.get("training", False),
        key="btn_train_from_params",
        use_container_width=True,
    ):
        st.session_state["train_requested"] = True

    if st.session_state.get("training", False):
        if st.button("✕ Cancelar treino", key="btn_cancel_training", use_container_width=True):
            cancel_event = st.session_state.get("train_cancel_event")
            if cancel_event is not None and not cancel_event.is_set():
                cancel_event.set()
            st.session_state["train_cancel_requested"] = True
            st.session_state["train_note"] = "Cancelamento solicitado. Aguardando o fim da época atual..."

    live_progress = st.empty()
    live_status = st.empty()

    if (
        has_required
        and st.session_state.get("train_requested", False)
        and not st.session_state.get("training", False)
    ):
        df_raw_state = st.session_state.get("df_raw")
        in_cols_state = list(st.session_state.get("in_cols", []))
        out_cols_state = list(st.session_state.get("out_cols", []))

        if df_raw_state is None or not in_cols_state or not out_cols_state:
            st.session_state["train_requested"] = False
            st.session_state["training"] = False
            st.session_state["training_error"] = "Base ou variáveis não disponíveis. Recarregue a base e selecione as variáveis novamente."
        else:
            st.session_state["training"] = True
            st.session_state["train_requested"] = False
            st.session_state["training_error"] = None
            st.session_state["train_cancel_requested"] = False
            st.session_state["train_epoch_current"] = 0
            st.session_state["train_epoch_total"] = int(params.get("max_epochs") or 0)
            st.session_state["train_loss"] = None
            st.session_state["train_val_loss"] = None
            st.session_state["train_note"] = "Preparando dados..."

            train_queue = queue.Queue()
            cancel_event = threading.Event()
            st.session_state["train_queue"] = train_queue
            st.session_state["train_cancel_event"] = cancel_event

            df_raw_copy = df_raw_state.copy(deep=True)
            in_cols_copy = list(in_cols_state)
            out_cols_copy = list(out_cols_state)
            params_copy = dict(st.session_state.get("train_params", {}))

            train_thread = threading.Thread(
                target=_train_worker,
                args=(df_raw_copy, in_cols_copy, out_cols_copy, params_copy, train_queue, cancel_event),
                daemon=True,
            )
            st.session_state["train_thread"] = train_thread
            train_thread.start()

    _render_live_training(live_progress, live_status)


def _render_training_panel(has_required, params):
    return _render_training_panel_impl(has_required, params)



def _train_worker(df_raw, in_cols, out_cols, params, progress_queue, cancel_event):
    try:
        tf.keras.backend.clear_session()
        n_neurons = int(params.get("n_neurons") or 10)
        max_epochs = int(params.get("max_epochs") or 500)
        learning_rate = float(params.get("learning_rate") or 1e-3)
        batch_size = int(params.get("batch_size") or 32)
        train_ratio = float(params.get("train_ratio") or 70) / 100.0
        split = 1.0 - train_ratio
        #EarlyStopping passa a ser proteção obrigatória.
        use_early_stopping = True
        patience = int(params.get("early_stopping_patience") or 20)

        progress_queue.put({
            "type": "progress",
            "epoch_current": 0,
            "epoch_total": max_epochs,
            "loss": None,
            "val_loss": None,
            "note": "Preparando dados...",
        })

        X_df, y_df, imputation_report, imputation_notes, data_diagnostics, sample_weight = _prepare_supervised_data_with_missingness(
            df_raw,
            in_cols,
            out_cols,
            input_row_missing_limit=0.80,
            input_col_missing_limit=0.60,
            random_state=42,
        )

        signal_diag = _compute_feature_target_signal(X_df, y_df)
        data_diagnostics.update(signal_diag)

        X = X_df.to_numpy(dtype=np.float32)
        y = y_df.to_numpy(dtype=np.float32)

        if X.shape[0] < 10:
            progress_queue.put({
                "type": "error",
                "message": "Poucas linhas disponíveis após pré-processamento (precisa de pelo menos 10).",
            })
            return

        # Em bases pequenas ou muito incompletas, evita validação excessivamente grande.
        split = min(max(split, 0.05), 0.50)

        # Validação estratificada por qualidade da linha: evita que o conjunto de validação
        # fique artificialmente mais limpo ou mais sujo do que o treino.
        stratify_bins = None
        try:
            if len(sample_weight) >= 40 and len(np.unique(sample_weight)) > 3:
                stratify_bins = pd.qcut(sample_weight, q=4, labels=False, duplicates="drop")
                counts = pd.Series(stratify_bins).value_counts()
                if counts.min() < 2:
                    stratify_bins = None
        except Exception:
            stratify_bins = None

        X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
            X, y, sample_weight, test_size=split, random_state=42, shuffle=True, stratify=stratify_bins
        )

        # Baseline pela média do treino: referência essencial para avaliar se o modelo aprendeu algo.
        baseline_pred = np.tile(np.mean(y_train, axis=0, keepdims=True), (len(y_val), 1))
        baseline_metrics = _compute_metrics(y_val, baseline_pred)
        data_diagnostics.update({
            "baseline_rmse": float(baseline_metrics.get("RMSE", 0.0)),
            "baseline_mae": float(baseline_metrics.get("MAE", 0.0)),
            "baseline_r2": float(baseline_metrics.get("R2", 0.0)),
            "baseline_r": float(baseline_metrics.get("R", 0.0)) if not np.isnan(baseline_metrics.get("R", np.nan)) else 0.0,
        })

        # RobustScaler reduz o impacto de outliers e valores extremos criados pela imputação.
        x_scaler = RobustScaler()
        y_scaler = RobustScaler()
        X_train_s = x_scaler.fit_transform(X_train)
        X_val_s = x_scaler.transform(X_val)
        y_train_s = y_scaler.fit_transform(y_train)
        y_val_s = y_scaler.transform(y_val)

        model = _build_model(
            n_features=X_train_s.shape[1],
            n_outputs=y_train_s.shape[1],
            n_neurons=n_neurons,
            learning_rate=learning_rate,
        )

        history_dict = {"loss": [], "val_loss": []}
        trained_epochs = 0
        best_metric = float("inf")
        best_weights = None
        epochs_without_improve = 0
        stop_note = ""
        final_loss = None
        final_val_loss = None

        progress_queue.put({
            "type": "progress",
            "epoch_current": 0,
            "epoch_total": max_epochs,
            "loss": None,
            "val_loss": None,
            "note": "Treinando modelo...",
        })

        for epoch_idx in range(max_epochs):
            if cancel_event.is_set():
                stop_note = "Treinamento cancelado pelo usuário."
                progress_queue.put({
                    "type": "cancelled",
                    "epoch_current": trained_epochs,
                    "epoch_total": max_epochs,
                    "loss": final_loss,
                    "val_loss": final_val_loss,
                    "history": history_dict,
                    "note": stop_note,
                })
                return

            hist_epoch = model.fit(
                X_train_s,
                y_train_s,
                validation_data=(X_val_s, y_val_s, sw_val),
                epochs=1,
                batch_size=int(batch_size),
                sample_weight=sw_train,
                verbose=0,
            )

            trained_epochs = epoch_idx + 1
            loss = hist_epoch.history.get("loss", [None])[-1]
            val_loss = hist_epoch.history.get("val_loss", [None])[-1]

            final_loss = None if loss is None else float(loss)
            final_val_loss = None if val_loss is None else float(val_loss)

            history_dict["loss"].append(final_loss)
            history_dict["val_loss"].append(final_val_loss)

            progress_queue.put({
                "type": "progress",
                "epoch_current": trained_epochs,
                "epoch_total": max_epochs,
                "loss": final_loss,
                "val_loss": final_val_loss,
                "note": "Cancelamento solicitado. Aguardando o fim da época atual..." if cancel_event.is_set() else "Treinando modelo...",
            })

            if cancel_event.is_set():
                stop_note = "Treinamento cancelado pelo usuário."
                progress_queue.put({
                    "type": "cancelled",
                    "epoch_current": trained_epochs,
                    "epoch_total": max_epochs,
                    "loss": final_loss,
                    "val_loss": final_val_loss,
                    "history": history_dict,
                    "note": stop_note,
                })
                return

            metric_now = final_val_loss if final_val_loss is not None else final_loss
            min_delta = 1e-6
            if metric_now is not None and metric_now < (best_metric - min_delta):
                best_metric = metric_now
                best_weights = model.get_weights()
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

                # ReduceLROnPlateau manual: suaviza treino em bases ruidosas.
                if epochs_without_improve > 0 and epochs_without_improve % max(5, int(patience // 2)) == 0:
                    try:
                        current_lr = float(keras.backend.get_value(model.optimizer.learning_rate))
                        new_lr = max(current_lr * 0.5, 1e-6)
                        if hasattr(model.optimizer.learning_rate, "assign"):
                            model.optimizer.learning_rate.assign(new_lr)
                        else:
                            keras.backend.set_value(model.optimizer.learning_rate, new_lr)
                    except Exception:
                        pass

                if epochs_without_improve >= int(patience):
                    stop_note = f"EarlyStopping acionado na época {trained_epochs}."
                    break

        if best_weights is not None:
            model.set_weights(best_weights)

        y_pred_val_s = model.predict(X_val_s, verbose=0)
        y_pred_val = y_scaler.inverse_transform(y_pred_val_s)
        y_val_orig = y_scaler.inverse_transform(y_val_s)

        metrics = _compute_metrics(y_val_orig, y_pred_val)
        data_diagnostics.update({
            "ganho_r2_vs_baseline": float(metrics.get("R2", 0.0) - data_diagnostics.get("baseline_r2", 0.0)),
            "ganho_rmse_vs_baseline": float(data_diagnostics.get("baseline_rmse", 0.0) - metrics.get("RMSE", 0.0)),
        })
        modelability_assessment = _build_modelability_assessment(metrics, data_diagnostics)
        data_diagnostics["modelability_score"] = float(modelability_assessment["score"])
        data_diagnostics["modelability_class"] = str(modelability_assessment["classe"])
        data_diagnostics["modelability_decision"] = str(modelability_assessment["decisao"])

        eps = 1e-9
        rel_err = np.mean(np.abs(y_pred_val - y_val_orig) / (np.abs(y_val_orig) + eps), axis=1)

        val_table = pd.DataFrame({
            "idx": np.arange(len(rel_err)),
            "erro_relativo": rel_err.astype(float),
            "peso_validacao": sw_val.astype(float),
            "qualidade_aproximada": np.sqrt(np.clip(sw_val.astype(float), 0.0, 1.0)),
        })
        if y_val_orig.shape[1] == 1:
            val_table["y_true"] = y_val_orig.reshape(-1).astype(float)
            val_table["y_pred"] = y_pred_val.reshape(-1).astype(float)

        ts = _ts()
        model_name = f"model_{ts}.keras"
        x_name = f"x_scaler_{ts}.pkl"
        y_name = f"y_scaler_{ts}.pkl"
        metadata_name = f"preprocess_metadata_{ts}.json"

        model_path = MODELS_DIR / model_name
        x_path = MODELS_DIR / x_name
        y_path = MODELS_DIR / y_name
        metadata_path = MODELS_DIR / metadata_name

        # Salva uma cópia limpa do modelo, sem o estado interno do otimizador.
        # Isso evita falhas ocasionais de serialização em algumas combinações de TensorFlow/Keras
        # sem afetar o uso do arquivo para inferência com model.predict().
        inference_model = keras.models.clone_model(model)
        inference_model.set_weights(model.get_weights())
        inference_model.save(model_path)

        with open(x_path, "wb") as fx:
            joblib.dump(x_scaler, fx)
        with open(y_path, "wb") as fy:
            joblib.dump(y_scaler, fy)

        preprocess_metadata = {
            "metadata_version": "1.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "artifacts": {
                "model_file": model_name,
                "x_scaler_file": x_name,
                "y_scaler_file": y_name,
                "metadata_file": metadata_name,
            },
            "selected_columns": {
                "input_columns_original": [str(c) for c in in_cols],
                "output_columns": [str(c) for c in out_cols],
                "model_feature_columns_after_preprocessing": [str(c) for c in X_df.columns],
                "n_model_features": int(X_df.shape[1]),
                "n_outputs": int(y_df.shape[1]),
            },
            "preprocessing_pipeline": {
                "numeric_conversion": "pd.to_numeric(errors='coerce')",
                "infinite_values": "convertidos para NaN",
                "target_policy": "linhas com saída ausente são removidas; a saída não é imputada",
                "input_missing_policy": "imputação adaptativa nas entradas",
                "short_gap_interpolation": "interpolação linear para gaps curtos de até 3 linhas",
                "complex_missing_imputation": "IterativeImputer/MICE quando o padrão de ausência é complexo; KNN como alternativa; mediana como fallback final",
                "missing_indicators": "colunas '__was_missing' adicionadas para entradas imputadas quando aplicável",
                "quality_features": ["__missing_ratio", "__row_quality"],
                "sample_weight": "(1 - missing_ratio)^2, com limite inferior adaptativo",
                "x_scaler": "RobustScaler salvo em x_scaler_file",
                "y_scaler": "RobustScaler salvo em y_scaler_file",
            },
            "training_configuration": {
                "n_neurons_layer_1": int(n_neurons),
                "max_epochs": int(max_epochs),
                "trained_epochs": int(trained_epochs),
                "learning_rate_initial": float(learning_rate),
                "batch_size": int(batch_size),
                "train_ratio": float(train_ratio),
                "validation_ratio": float(split),
                "random_state": 42,
                "loss_function": "Huber(delta=1.0)",
                "optimizer": "Adam(clipnorm=1.0)",
                "early_stopping": {
                    "enabled": True,
                    "patience": int(patience),
                    "stop_note": stop_note or "não acionado",
                },
            },
            "metrics_validation": metrics,
            "baseline_validation": {
                "baseline_rmse": float(data_diagnostics.get("baseline_rmse", 0.0)),
                "baseline_mae": float(data_diagnostics.get("baseline_mae", 0.0)),
                "baseline_r2": float(data_diagnostics.get("baseline_r2", 0.0)),
                "baseline_r": float(data_diagnostics.get("baseline_r", 0.0)),
            },
            "data_diagnostics": data_diagnostics,
            "modelability_assessment": modelability_assessment,
            "imputation_notes": [str(n) for n in imputation_notes],
            "important_note": (
                "Este pacote contém o modelo, os scalers e os metadados de pré-processamento. "
                "Para prever fora deste app, aplique o mesmo pré-processamento, depois o x_scaler, o modelo .keras e, por fim, o y_scaler.inverse_transform()."
            ),
        }
        metadata_path.write_text(json.dumps(preprocess_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        model_bytes = model_path.read_bytes()
        metadata_bytes = metadata_path.read_bytes()

        package_name = f"modelo_completo_{ts}.zip"
        package_buffer = io.BytesIO()
        with zipfile.ZipFile(package_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(model_path, arcname=model_name)
            zf.write(x_path, arcname=x_name)
            zf.write(y_path, arcname=y_name)
            zf.write(metadata_path, arcname=metadata_name)
        package_bytes = package_buffer.getvalue()

        progress_queue.put({
            "type": "done",
            "metrics": metrics,
            "val_table": val_table,
            "history": history_dict,
            "imputation_report": imputation_report,
            "imputation_notes": imputation_notes,
            "data_diagnostics": data_diagnostics,
            "modelability_assessment": modelability_assessment,
            "artifacts": {
                "model_path": str(model_path),
                "x_scaler_path": str(x_path),
                "y_scaler_path": str(y_path),
                "model_bytes": model_bytes,
                "model_name": model_name,
                "metadata_bytes": metadata_bytes,
                "metadata_name": metadata_name,
                "preprocess_metadata_path": str(metadata_path),
                "package_bytes": package_bytes,
                "package_name": package_name,
            },
            "note": stop_note or "Treinamento concluído.",
        })
    except Exception as e:
        progress_queue.put({
            "type": "error",
            "message": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })


# ===============================
# Página 2 – Resultados
# ===============================

def render_results():
    st.markdown('<div class="tcc-h1">Resultados</div>', unsafe_allow_html=True)
    st.markdown('<div class="tcc-h2">Métricas, modelabilidade e diagnóstico da base</div>', unsafe_allow_html=True)

    top = st.columns([1, 1, 1])
    with top[0]:
        if st.button("◀ Voltar"):
            _go("setup")
            st.rerun()
    with top[2]:
        artifacts = st.session_state.get("artifacts", {})
        package_b = artifacts.get("package_bytes", None)
        package_fname = artifacts.get("package_name", "modelo_completo.zip")
        if package_b is not None:
            st.download_button(
                "Baixar modelo completo (.zip)",
                data=package_b,
                file_name=package_fname,
                mime="application/zip",
                use_container_width=True,
            )

    metrics = st.session_state.get("metrics", None)
    if metrics is None:
        st.warning("Nenhum resultado disponível. Treine um modelo primeiro.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.container(key="results_card"):
        # Métricas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"{metrics['RMSE']:.6f}")
        m2.metric("MAE", f"{metrics['MAE']:.6f}")
        m3.metric("R2", f"{metrics['R2']:.6f}")
        m4.metric("R", f"{metrics['R']:.6f}")

        d1, d2, d3 = st.columns(3)
        d1.metric("MAPE", f"{metrics['MAPE']:.6f}")
        d2.metric("SMAPE", f"{metrics['SMAPE']:.6f}")
        d3.metric("WMAPE", f"{metrics['WMAPE']:.6f}")

        modelability = st.session_state.get("modelability_assessment") or {}
        data_diagnostics_for_score = st.session_state.get("data_diagnostics") or {}
        if modelability:
            st.markdown("#### Avaliação de modelabilidade dos dados")
            score = float(modelability.get("score", 0.0))
            s1, s2, s3 = st.columns([1, 1.4, 2.2])
            s1.metric("Score", f"{score:.1f}/100")
            s2.metric("Classificação", str(modelability.get("classe", "—")))
            s3.info(str(modelability.get("decisao", "—")))
            st.progress(min(max(score / 100.0, 0.0), 1.0))

            fatores = modelability.get("fatores", [])
            if fatores:
                st.caption("Principais fatores detectados: " + "; ".join(map(str, fatores)) + ".")

        data_diagnostics = st.session_state.get("data_diagnostics")
        if data_diagnostics is not None:
            st.markdown("#### Diagnóstico da base de dados")
            cdiag1, cdiag2, cdiag3, cdiag4 = st.columns(4)
            cdiag1.metric("Valores ausentes nas entradas", f"{data_diagnostics.get('percentual_nan_entradas_original', 0.0):.1%}")
            cdiag2.metric("Linhas usadas", f"{int(data_diagnostics.get('linhas_finais_treino_validacao', 0))}")
            cdiag3.metric("Qualidade média", f"{data_diagnostics.get('qualidade_media_amostras', 0.0):.1%}")
            cdiag4.metric("Features finais", f"{int(data_diagnostics.get('features_finais_do_modelo', 0))}")

        hist = st.session_state.get("history", None)
        if hist and "loss" in hist and "val_loss" in hist:
            st.markdown("#### Curva de aprendizagem")
            fig = plt.figure()
            plt.plot(hist["loss"], label="Treinamento")
            plt.plot(hist["val_loss"], label="Validação")
            plt.legend()
            plt.xlabel("Época")
            plt.ylabel("Função de perda")
            plt.title("Evolução da perda durante o treinamento")
            st.pyplot(fig)


# ===============================
# Router
# ===============================

if st.session_state["page"] == "setup":
    render_setup()
elif st.session_state["page"] == "params":
    render_params()
else:
    render_results()
