import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import os
import joblib
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc

# --------------------------------------------------------------------------------
# 0. CLASS DEFINITION & UTILS
# --------------------------------------------------------------------------------
class RollingEnsembleClassifier:
    def __init__(self):
        self.estimators = []
        self.classes_ = None

    def add_model(self, scaler, model, period_name):
        self.estimators.append({
            'scaler': scaler,
            'model': model,
            'period': period_name
        })
        if self.classes_ is None:
            self.classes_ = model.classes_

    def predict_proba(self, X):
        if not self.estimators:
            raise ValueError("No models added.")
        
        avg_proba = None
        for item in self.estimators:
            scaler = item['scaler']
            model = item['model']
            
            # Feature Alignment
            if isinstance(X, pd.DataFrame) and hasattr(scaler, 'feature_names_in_'):
                 X_input = X.reindex(columns=scaler.feature_names_in_, fill_value=0)
            else:
                 X_input = X
            
            X_scaled = scaler.transform(X_input)
            proba = model.predict_proba(X_scaled)
            
            if avg_proba is None:
                avg_proba = proba
            else:
                avg_proba += proba
                
        avg_proba /= len(self.estimators)
        return avg_proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]



# ---------------------------------------------------------------------------------
# [전략 제안 딕셔너리 - 19개 지표 대응]
STRATEGIC_ADVICE = {
    "TQ": {"pos": "시장 가치(Tobin's Q)가 높아 미래 성장성에 대한 신뢰가 두텁습니다.", "neg": "자산 대비 시장 가치가 저평가되어 있습니다. IR 강화가 필요합니다."},
    "ROA": {"pos": "우수한 자산 효율성(ROA)이 ESG 경영의 토대가 됩니다.", "neg": "수익성 저하가 ESG 투자 여력을 제한하고 있습니다."},
    "SGR": {"pos": "견고한 매출 성장세가 기업 활력을 증명합니다.", "neg": "성장 정체 리스크가 존재합니다. 비즈니스 모델 전환을 검토하세요."},
    "LEV": {"pos": "안정적인 부채비율이 재무 리스크를 방어합니다.", "neg": "높은 부채비율이 재무 불안정성을 키우고 있습니다."},
    "A_SIZE": {"pos": "규모의 경제를 갖춘 대기업으로서 ESG 역량이 우수합니다.", "neg": "작은 자산 규모로 인한 ESG 관리 한계를 효율화로 극복해야 합니다."},
    "W_YEAR": {"pos": "높은 근속연수는 인적 자원의 안정성을 뜻합니다.", "neg": "짧은 근속연수는 인력 유출 리스크를 시사합니다."},
    "Fe_R": {"pos": "여성 직원 비율이 높아 다양성 측면에서 긍정적입니다.", "neg": "인력 구조의 다양성이 부족합니다. 채용 정책을 점검하세요."},
    "Re_R": {"pos": "높은 정규직 비율은 고용의 질이 우수함을 뜻합니다.", "neg": "비정규직 비중이 높아 고용 안정성 리스크가 있습니다."},
    "SA": {"pos": "우수한 임금 수준이 인재 확보 경쟁력을 높입니다.", "neg": "낮은 임금 수준은 인재 이탈 원인이 될 수 있습니다."},
    "Pay_Gap": {"pos": "낮은 임금 격차는 조직 내 형평성이 높음을 시사합니다.", "neg": "사내 임금 격차가 커 조직 결속력을 해칠 수 있습니다."},
    "FOR": {"pos": "높은 외국인 지분율이 경영 투명성을 보장합니다.", "neg": "외국인 투자자의 관심도가 낮습니다. 영문 공시를 확대하세요."},
    "MSE": {"pos": "적절한 대주주 지분율이 경영 안정성을 제공합니다.", "neg": "지분 구조가 지나치게 집중되어 이사회의 독립성이 우려됩니다."},
    "DIR_OUT": {"pos": "높은 사외이사 비율이 견제와 균형을 돕고 있습니다.", "neg": "사외이사 비중이 낮아 이사회의 독립성이 우려됩니다."},
    "DIR_FE": {"pos": "경영진 내 여성 비율이 높아 의사결정 다양성이 확보되었습니다.", "neg": "의사결정 기구의 성별 다양성이 부족합니다."},
    "SGAE_R": {"pos": "효율적인 판관비 관리가 수익성 개선으로 이어집니다.", "neg": "매출 대비 판관비 비중이 높아 운영 효율화가 시급합니다."},
    "DIV": {"pos": "주주 환원 정책이 우수하여 G등급에 긍정적입니다.", "neg": "적극적인 배당 정책으로 주주 신뢰를 회복하세요."},
    "DIV_enco": {"pos": "배당 실적이 주주 친화 경영을 증명합니다.", "neg": "배당 도입을 통해 지배구조 점수를 보완할 수 있습니다."},
    "DIR_FE_enco": {"pos": "여성 임원 선임은 거버넌스 선진화의 신호입니다.", "neg": "여성 임원 선임을 통해 이사회 다양성을 확보하세요."},
    "ESG_lag": {
        "pos": "과거의 우수한 ESG 경영 성과가 현재 등급을 견고하게 지지하고 있습니다.",
        "neg": "과거의 낮은 등급이 현재 평가에 하방 압력을 주고 있습니다. 구조적 혁신이 필요합니다."
    },
    "A_SIZE_FOR_inter": {
        "pos": "기업 규모와 외국인 투자자의 감시 체계가 시너지를 내어 지배구조 점수를 높이고 있습니다.",
        "neg": "자산 규모 대비 외국인 투자자의 긍정적 영향력이 충분히 발휘되지 못하고 있습니다."
    }

}


# [제안 생성 함수]
def make_shap_based_advice(tmp_df, model_in, advice_dict, top_k=3):
    results = {"pos": [], "neg": []}
    for _, row in tmp_df.iterrows():
        feat = str(row["feature"]).strip()
        if feat in advice_dict:
            val = model_in.iloc[0][feat] if feat in model_in.columns else None
            # SHAP 값이 양수면 pos, 음수면 neg
            if row["shap"] > 0:
                if len(results["pos"]) < top_k:
                    results["pos"].append({"feature": feat, "value": val, "text": advice_dict[feat]["pos"]})
            else:
                if len(results["neg"]) < top_k:
                    results["neg"].append({"feature": feat, "value": val, "text": advice_dict[feat]["neg"]})
    return results
# --------------------------------------------------------------------------------








# --------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="ESG Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #000000; }
    h1, h2, h3, h4, h5, h6 { color: #4B0082 !important; font-family: 'Helvetica Neue', sans-serif; }
    .home-title { text-align: center; color: #4B0082; font-size: 3.5rem; font-weight: 800; margin-top: 50px; }
    .home-subtitle { text-align: center; color: #DAA520; font-size: 1.8rem; font-weight: 500; margin-bottom: 20px; }
    .team-names { text-align: center; color: #333333; font-size: 1.2rem; margin-top: 10px; margin-bottom: 50px; }
    .info-box { padding: 20px; background-color: #f8f9fa; border-left: 5px solid #4B0082; border-radius: 5px; margin-bottom: 20px; }
    .warning-box { padding: 20px; background-color: #fff3cd; border-left: 5px solid #ffc107; border-radius: 5px; margin-bottom: 20px; }
    .stButton>button { background-color: #4B0082; color: #DAA520; font-weight: bold; width: 100%; }
    .metric-container { background-color: #F0F2F6; padding: 10px; border-radius: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

COLOR_MAIN = '#4B0082'
COLOR_ACCENT = '#DAA520'
COLOR_ALERT = '#FF4B4B'

# --------------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# --------------------------------------------------------------------------------
@st.cache_data
def load_data_basic():
    # 경로: data/fin/fin_total_all_years.csv
    file_path = os.path.join("data", "fin", "fin_total_all_years.csv")
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path)
    if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])
    return df.dropna().copy()

@st.cache_data
def load_data_advanced():
    # 경로: data/X_features_fin.csv
    file_path = os.path.join("data", "X_features_fin.csv")
    if not os.path.exists(file_path): return None, None, None, None

    df = pd.read_csv(file_path)
    if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])
    df = df.sort_values(by=['corp_code', 'year'])

    # 전처리: 산업군별 중앙값 대체
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['corp_code', 'stock_code', 'year', 'IND']
    targets = [c for c in numeric_cols if c not in exclude]

    if 'IND' in df.columns:
        df[targets] = df.groupby('IND')[targets].transform(lambda x: x.fillna(x.median()))
    df[targets] = df[targets].fillna(df[targets].median())

    # [수정] Target 생성: Shift 제거
    # 원본 데이터의 ESG 컬럼이 이미 T+1 시점의 정답 데이터라고 확인됨.
    df['Target_Grade'] = df['ESG'] 
    
    # Target이 있는 데이터만 사용
    df_fin = df.dropna(subset=['Target_Grade']).copy()
    
    # 검색용 데이터 (예측 대상 연도 = 재무년도 + 1)
    full_search = df.copy()
    full_search['year'] = full_search['year'] + 1 

    # X, y 생성
    drop_cols = ['corp_name', 'G', 'S', 'E', 'stock_code', 'corp_code', 'year', 'ESG', 'Target_Grade']
    X = df_fin.drop(columns=[c for c in drop_cols if c in df_fin.columns])
    if 'IND' in X.columns: X = pd.get_dummies(X, columns=['IND'], prefix='IND')
    y_cls = df_fin['Target_Grade']
    
    return X, y_cls, df_fin, full_search

# --------------------------------------------------------------------------------
# 3. MODEL LOADING
# --------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        # 파일 경로 확인
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 모델 로드
        models['reg_fin'] = joblib.load(os.path.join(base_dir, 'esg_model_regression_fin.pkl'))
        models['scaler_reg_fin'] = joblib.load(os.path.join(base_dir, 'esg_scaler_regression_fin.pkl'))
        models['cls_select'] = joblib.load(os.path.join(base_dir, 'esg_model_classifier_select.pkl'))
        models['final'] = joblib.load(os.path.join(base_dir, 'esg_model_classifier_final_depth7.pkl'))
        
        # 확장형 모델 로드
        models['ext_model'] = joblib.load(os.path.join(base_dir, 'xgb_model_ext.pkl'))
        models['ext_scaler'] = joblib.load(os.path.join(base_dir, 'scaler_ext.pkl'))

    except Exception as e:
        # 에러 발생 시 화면에 붉은 박스로 표시 (매우 중요!)
        st.error(f"모델 로딩 실패: {e}")
        st.info("requirements.txt의 scikit-learn 버전을 확인하거나, pkl 파일이 깃허브에 잘 올라갔는지 확인하세요.")
        
    return models

df_basic = load_data_basic()
X_adv, y_cls_adv, df_adv, full_search_df = load_data_advanced()
models = load_models()

# Label Encoder (D to S)
if y_cls_adv is not None:
    unique_classes = sorted(y_cls_adv.unique())
    le = LabelEncoder()
    le.fit(unique_classes)

# --------------------------------------------------------------------------------
# MAIN TABS
# --------------------------------------------------------------------------------
tab_home, tab_overview, tab_reg, tab_cls, tab_final, tab_pred = st.tabs([
    "HOME", "OVERVIEW", "REGRESSION", "CLASSIFICATION", "FINAL MODEL", "PREDICTOR"
])

# ==============================================================================
# TAB 1: HOME
# ==============================================================================
with tab_home:
    st.markdown('<div class="home-title">ESG Prediction Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-subtitle">Financial Data Based Forecasting</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="team-names">
            <b>Team Members</b><br>
            Park Hyun-woo | Min Sun-ah
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.info("💡 **Project Goal:** 재무 데이터를 활용하여 기업의 차년도(T+1) ESG 등급을 예측하고, 개선 가이드를 제공하는 모델링")

# ==============================================================================
# TAB 2: OVERVIEW
# ==============================================================================
with tab_overview:
    st.subheader("Features & Performance Overview")
    st.markdown("""
        <div class="info-box">
        <b>성과 요약:</b><br>
        기존 선행 연구(R² 0.225) 대비 우리 모델은 <b>R² 0.585</b>로 설명력을 대폭 개선하였으며,<br>
        분류 모델(XGBoost) 전환 후 <b>AUC 0.829</b>의 우수한 성능을 달성하였습니다.<br>
        * 논문 참조: 이재영, 차우창(2024) “머신러닝 모델을 활용한 ESG 활동과 기업 가치 분석”, 한국산업경영시스템학회지 47(4), 76-86.
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔹 Initial X_features")
        st.markdown("""
        | 변수명 | 설명 | 비고 |
        |---|---|---|
        | **A_SIZE** | log(총자산) | 기업 규모 |
        | **LEV** | 총부채 / 총자산 | 부채 비율 |
        | **TQ** | (시가총액+총부채)/총자산 | **Tobins'Q** |
        | **FOR** | 외국인 지분율 | 글로벌 모니터링 |
        | **MSE** | 주요주주 지분율 | 소유 구조 |
        | **ROA** | 총자산 수익률 | 수익성 |
        | **ADV** | 광고선전비 | 가시성 |
        | **SGR** | 매출액 성장률 | 성장성 |
        | **R&D** | 연구개발비 | 혁신성 |
        """)
        
    with col2:
        st.markdown("### 🔸 Advanced X_features")
        st.markdown("""
        | 변수명 | 설명 | 비고 |
        |---|---|---|
        | **SGAE_R** | 판관비 비율 | **ADV + R&D 결합** |
        | **Fe_R** | 남성 대비 여성 직원 비율 | 다양성 |
        | **Re_R** | 정규직 비율 | 고용 안정성 |
        | **SA** | log(1인당 평균임금) | 직원 처우 |
        | **Pay_Gap** | 남녀 임금 격차 | 공정성 |
        | **W_YEAR** | 평균 근속연수 | 조직 안정성 |
        | **DIV** | 주가 배당율 | 주주 환원 |
        | **DIR_FE** | 여성 임원 비율 | 이사회 다양성 |
        | **DIR_OUT** | 사외이사 비율 | 이사회 독립성 |
        """)

    with col3:
        st.markdown("### 🏆 Performance Milestone")
        st.write("") 
        milestone_df = pd.DataFrame({
            "Stage": ["Previous Research", "Initial Regression", 'Plus Regression', "Advanced Regression", 'Base Classification', "Final Classification"],
            "Metric": ["R² Score", "R² Score", 'R² Score', "R² Score", "ROC-AUC", "ROC-AUC"],
            "Score": [0.225, 0.440, 0.585, 0.664, 0.800, 0.829]
        })
        st.table(milestone_df.style.format({"Score": "{:.3f}"}).set_properties(**{'text-align': 'center', 'font-size': '16px'}))
        
    st.markdown("---")    
    st.markdown(f"""
    <div style="background-color: {COLOR_MAIN}; padding: 15px; border-radius: 5px; color: white; text-align: center; margin-top: 20px;">
    <b>"이전 연구에서 보였던 낮은 설명력을<br>최종 ROC_AUC 0.829로 크게 발전"</b>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 3: REGRESSION ANALYSIS
# ==============================================================================
with tab_reg:
    st.subheader("Regression Analysis Process")
    
    st.markdown("""
    <div class="info-box">
    <b>회귀분석 과정 요약:</b><br>
    초기 모델(Initial)의 한계를 극복하기 위해 파생변수 추가(Advanced) 및 롤링 윈도우 최적화를 수행하였습니다.<br>
    분석 결과, <b>최적의 Window Size는 3년(R² 0.585)</b>으로 도출되었으나, 여전히 존재하는 성능 한계(학습 곡선 정체)를 확인하고 분류 모델로의 전환을 결정하였습니다.
    </div>
    """, unsafe_allow_html=True)
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["1. Initial Model", "2. Feature Expanded", "3. Final Regression"])
    
    with sub_tab1:
        st.markdown("#### Initial Model Performance (Total vs Sector)")
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("**Correlation Matrix (Include ESG)**")
            if df_basic is not None:
                df_corr = df_basic.copy()
                grade_map = {'S': 7.0, 'A+': 6.0, 'A': 5.0, 'B+': 4.0, 'B': 3.0, 'C': 2.0, 'D': 1.0}
                if 'ESG' in df_corr.columns:
                    df_corr['ESG'] = df_corr['ESG'].map(grade_map)
                
                exclude_cols = ['corp_name', 'stock_code', 'corp_code', 'year']
                df_corr = df_corr.drop(columns=[c for c in exclude_cols if c in df_corr.columns])
                
                corr = df_corr.select_dtypes(include=[np.number]).corr()
                fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
                fig_heat.update_layout(height=700, font=dict(size=14))
                st.plotly_chart(fig_heat, use_container_width=True)
        with c2:
            st.markdown("**R² Score by Sector**")
            sectors = ['Total Model', 'E (Environment)', 'S (Social)', 'G (Governance)']
            scores = [0.440, 0.422, 0.450, 0.244]
            colors = [COLOR_MAIN, COLOR_MAIN, COLOR_MAIN, COLOR_ALERT]
            
            fig_bar = go.Figure(go.Bar(
                x=scores, y=sectors, orientation='h',
                text=scores, marker_color=colors, textposition='auto'
            ))
            fig_bar.update_layout(xaxis_range=[0, 0.6], height=500, font=dict(size=15))
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown(f"""
            <div class="warning-box">
            <b>📉 분석 결과:</b><br>
            G(지배구조) 분야는 정성적 요소가 강해 외형적 재무 지표만으로는 설명하기 어렵다는 한계 확인
            </div>
            """, unsafe_allow_html=True)

    with sub_tab2:
        st.markdown("#### Feature Expansion & Optimization")
        
        col_lc, col_rw = st.columns(2)
        with col_lc:
            st.markdown("**Learning Curve (R²)**")
            train_sizes = np.linspace(0.1, 1.0, 7)
            train_scores = [0.59, 0.588, 0.585, 0.585, 0.585, 0.585, 0.585]
            val_scores =   [-1.5, -0.8, -0.2, 0.2, 0.4, 0.5, 0.57]
            
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_scores, mode='lines+markers', name='Training Score', line=dict(color='red')))
            fig_lc.add_trace(go.Scatter(x=train_sizes, y=val_scores, mode='lines+markers', name='Validation Score', line=dict(color='green')))
            fig_lc.update_layout(yaxis_range=[-1.5, 1], xaxis_title="Training Samples", yaxis_title="R² Score")
            st.plotly_chart(fig_lc, use_container_width=True)
            
            st.info("📢 **회귀 Base Model의 학습 한계 확인 (데이터가 늘어도 성능 정체)**")
            
        with col_rw:
            st.markdown("**Rolling Window Performance**")
            windows = [2, 3, 4, 5]
            rw_scores = [0.386, 0.585, 0.572, 0.568]
            
            fig_rw = go.Figure()
            fig_rw.add_trace(go.Scatter(x=windows, y=rw_scores, mode='lines+markers', line=dict(width=3, color=COLOR_MAIN)))
            fig_rw.add_trace(go.Scatter(x=[3], y=[0.585], mode='markers', marker=dict(size=15, color='red'), name='Best'))
            fig_rw.add_annotation(x=3, y=0.585, text="Best: 0.585", showarrow=True, arrowhead=1)
            fig_rw.update_layout(xaxis_title="Window Size (Year)", yaxis_title="R²")
            st.plotly_chart(fig_rw, use_container_width=True)
            
            st.info("📢 **최적 Window Size 3년 (R² 0.585) 도출 **")

        st.markdown("---")
        st.markdown("**Model Performance Comparison (Optimization)**")
        sorted_models = ['DecisionTree', 'Linear', 'RandomForest', 'LightGBM', 'XGBoost']
        test_r2 = [0.302, 0.585, 0.606, 0.637, 0.664]
        train_r2 = [0.304, 0.580, 0.676, 0.755, 0.795]
        
        fig_ms = go.Figure()
        fig_ms.add_trace(go.Scatter(x=sorted_models, y=train_r2, mode='lines+markers', name='Train R²', line=dict(dash='dash', color='blue')))
        fig_ms.add_trace(go.Scatter(x=sorted_models, y=test_r2, mode='lines+markers', name='Test R²', line=dict(color='red', width=3)))
        fig_ms.update_layout(yaxis_range=[0.2, 0.9])
        st.plotly_chart(fig_ms, use_container_width=True)
        
        st.info("📢 **회귀 모델 중 최적의 모델(XGBoost) 확인**")

    with sub_tab3:
        st.markdown("#### Final Regression Model Limit")
        comp_df = pd.DataFrame({
            "Metric": ["Train R²", "Test R²", "Gap"],
            "Base Model (Plus)": ["0.580", "0.585", "-0.005"], 
            "Final Model (XGB)": ["0.795", "0.664", "0.131"]
        })
        st.table(comp_df.set_index("Metric"))
        
        st.markdown(f"""
        <div class="warning-box" style="text-align: center; font-size: 18px;">
        <b>"최종 회귀 모델 Test R² 0.664 달성 했으나, Train-Test 간 격차로 인한 과적합 우려와<br>
        회귀 모델의 성능 한계 도달로 분류 모델로의 전환 필요성 확인"</b>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 4: CLASSIFICATION ANALYSIS
# ==============================================================================
with tab_cls:
    st.subheader("Classification Model Comparison")
    st.markdown("""
    <div class="info-box">
    <b>모델 전환 전략:</b><br>
    회귀분석의 한계를 극복하기 위해 <b>다중 분류(Multi-Class Classification)</b>로 문제를 재정의하였습니다.<br>
    5개 알고리즘에 대해 <b>3년 롤링 윈도우를</b>을 적용하여 ROC-AUC를 비교 분석하였습니다.
    </div>
    """, unsafe_allow_html=True)
    
    sub_roc, sub_param = st.tabs(["ROC AUC", "Parameter Tuning"])
    
    # [4-1] ROC AUC (Smoothed Curves & New Models)
    with sub_roc:
        st.markdown("#### Multi-Model ROC Comparison (Macro-average)")
        
        # Hardcoded smoothed data points for 5 models
        # XGBoost (0.823) - Best
        fpr_xgb = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
        tpr_xgb = [0, 0.35, 0.55, 0.72, 0.83, 0.89, 0.95, 0.98, 1]
        
        # LGBM (0.819)
        fpr_lgbm = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
        tpr_lgbm = [0, 0.33, 0.53, 0.70, 0.81, 0.88, 0.94, 0.97, 1]
        
        # Random Forest (0.809)
        fpr_rf = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
        tpr_rf = [0, 0.30, 0.50, 0.68, 0.79, 0.86, 0.93, 0.96, 1]
        
        # SVM (0.805)
        fpr_svm = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
        tpr_svm = [0, 0.28, 0.48, 0.66, 0.78, 0.85, 0.92, 0.96, 1]
        
        # Logistic (0.800)
        fpr_log = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1]
        tpr_log = [0, 0.27, 0.47, 0.65, 0.77, 0.84, 0.91, 0.95, 1]

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), name='Random'))
        
        # Add traces with spline smoothing
        fig_roc.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, name='XGBoost (AUC = 0.823)', line=dict(color=COLOR_MAIN, width=4, shape='spline')))
        fig_roc.add_trace(go.Scatter(x=fpr_lgbm, y=tpr_lgbm, name='LGBM (AUC = 0.819)', line=dict(color=COLOR_ACCENT, width=2, shape='spline')))
        fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name='Random Forest (AUC = 0.809)', line=dict(color='green', width=2, shape='spline')))
        fig_roc.add_trace(go.Scatter(x=fpr_svm, y=tpr_svm, name='SVM (AUC = 0.805)', line=dict(color='orange', width=2, shape='spline')))
        fig_roc.add_trace(go.Scatter(x=fpr_log, y=tpr_log, name='Logistic (AUC = 0.800)', line=dict(color='blue', width=2, shape='spline')))
        
        fig_roc.update_layout(title="Multi-Model ROC Comparison", xaxis_title="FPR", yaxis_title="TPR", height=600)
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.success("✅ **XGBoost**가 AUC 및 안정성 측면에서 가장 우수한 성능을 보여 최종 모델로 선정")

    # [4-2] Parameter Tuning (Provided Data)
    with sub_param:
        st.markdown("#### Max Depth Tuning (Overfitting Check)")
        
        col_p1, col_p2 = st.columns(2)
        
        # Data
        depths = list(range(3, 21))
        # Left Chart: Test 2024
        test_auc_24 = [0.8028, 0.8161, 0.8231, 0.8265, 0.8290, 0.8308, 0.8311, 0.8305, 0.8312, 0.8317, 0.8300, 0.8297, 0.8299, 0.8303, 0.8309, 0.8309, 0.8320, 0.8312]
        
        # Right Chart: Test 2023 (Validation)
        test_auc_23 = [0.7781, 0.7816, 0.7835, 0.7863, 0.7873, 0.7829, 0.7831, 0.7797, 0.7811, 0.7798, 0.7810, 0.7795, 0.7806, 0.7800, 0.7782, 0.7812, 0.7794, 0.7787]
        
        with col_p1:
            fig_p1 = go.Figure()
            fig_p1.add_trace(go.Scatter(x=depths, y=test_auc_24, name="Test (Window Size)", line=dict(color='red')))
            fig_p1.add_trace(go.Scatter(x=[7], y=[0.8290], mode='markers', marker=dict(size=15, color='blue'), name='Slowing Point (Depth 7)'))
            fig_p1.add_trace(go.Scatter(x=[19], y=[0.8320], mode='markers', marker=dict(size=15, color='orange'), name='Best (Depth 19)'))
            fig_p1.update_layout(title="Finding Optimal Max_Depth (Current Year)", xaxis_title="Max Depth", yaxis_title="AUC")
            st.plotly_chart(fig_p1, use_container_width=True)
            
            st.info("📢 **Best Score는 19이지만, Depth 7부터 급격한 성장 완화 확인**")
            
        with col_p2:
            fig_p2 = go.Figure()
            fig_p2.add_trace(go.Scatter(x=depths, y=test_auc_23, name="Test (2023 Validation)", line=dict(color='green', width=3)))
            # Highlight 7
            fig_p2.add_trace(go.Scatter(x=[7], y=[0.7873], mode='markers', marker=dict(size=15, color='orange'), name='Best (Depth 7)'))
            fig_p2.update_layout(title="Cross-Validation Like Check (Past Year)", xaxis_title="Max Depth", yaxis_title="AUC")
            st.plotly_chart(fig_p2, use_container_width=True)

            st.info("📢 **Best Score 7로 성장곡선 교차검증 완료**")
            
        st.markdown(f"""
        <div style="text-align: center; background-color: {COLOR_MAIN}; color: white; padding: 10px; border-radius: 5px;">
        <b>최적 파라미터 (max_depth = 7) 도출: 과적합 방지 및 일반화 성능 확보</b>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 5: FINAL MODEL
# ==============================================================================
with tab_final:
    st.subheader("Final Model Analysis (XGBoost Depth 7)")
    st.markdown("""
    <div class="info-box">
    <b>최종 모델 선정 이유:</b><br>
    Depth 19 모델이 점수는 더 높았으나(0.832), 과거 데이터 검증 시 과적합이 확인되었습니다.<br>
    따라서 <b>일반화 성능이 검증된 Depth 7 (AUC 0.829)</b>을 최종 모델로 채택하였습니다.
    </div>
    """, unsafe_allow_html=True)
    
    if 'final' in models:
        final_model = models['final']
        last_model = final_model.estimators[-1]['model']
        last_scaler = final_model.estimators[-1]['scaler']
        
        sub_score, sub_shap = st.tabs(["Final Score & Importance", "SHAP Analysis"])
        
        with sub_score:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Feature Importance (All_Features)")
                if hasattr(last_scaler, 'feature_names_in_'):
                    feat_names = last_scaler.feature_names_in_
                else:
                    feat_names = X_adv.columns
                
                importances = last_model.feature_importances_
                fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
                fi_df = fi_df[~fi_df['Feature'].str.startswith('IND_')].sort_values(by='Importance', ascending=True).tail(20)
                
                fig_imp = px.bar(fi_df, x='Importance', y='Feature', orientation='h')
                fig_imp.update_traces(marker_color=COLOR_MAIN)
                fig_imp.update_layout(height=500)
                st.plotly_chart(fig_imp, use_container_width=True)
                
            with c2:
                st.markdown("#### Final Model ROC - AUC Curve")
                # Hard-coded from Image Data (Slightly better than XGBoost in Multi-model)
                fpr_final = [0, 0.05, 0.15, 0.3, 0.5, 0.8, 1]
                tpr_final = [0, 0.32, 0.58, 0.78, 0.90, 0.96, 1] # Slightly smoothed
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr_final, y=tpr_final, fill='tozeroy', 
                                             name='Macro AUC (0.829)', 
                                             line=dict(color=COLOR_MAIN, width=3, shape='spline')))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray')))
                fig_roc.update_layout(height=500, xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig_roc, use_container_width=True)

        with sub_shap:
            st.markdown("#### SHAP Beeswarm Analysis")
            st.info(
                    "SHAP_Analysis: AI가 왜 이런 결과를 냈는지, "
                    "어떤 항목이 결과에 가장 큰 영향을 줬는지를 보여줍니다.\n\n"
                    "📊 그래프 읽는 법\n"
                    "• 위에 있을수록 결과에 더 중요한 항목입니다\n"
                    "• 오른쪽일수록 결과를 높이는 영향, 왼쪽일수록 낮추는 영향입니다\n"
                    "• 점이 많을수록 해당 사례가 자주 나타난다는 의미입니다"
                    )

            
            try:
                X_sample = X_adv.sample(min(100, len(X_adv)))
                if hasattr(last_scaler, 'feature_names_in_'):
                    X_sample = X_sample.reindex(columns=last_scaler.feature_names_in_, fill_value=0)
                
                X_scaled = last_scaler.transform(X_sample)
                explainer = shap.TreeExplainer(last_model)
                shap_values = explainer.shap_values(X_scaled)
                
                non_ind_cols = [c for c in X_sample.columns if not c.startswith('IND_')]
                keep_idx = [X_sample.columns.get_loc(c) for c in non_ind_cols]
                
                X_vis = X_sample[non_ind_cols]
                
                col_s1, col_s2 = st.columns(2)
                
                with col_s1:
                    st.markdown("##### Grade 1 (Lowest) Drivers")
                    if isinstance(shap_values, list):
                        shap_v = shap_values[0][:, keep_idx]
                    else:
                        shap_v = shap_values[:, keep_idx, 0]
                        
                    plt.figure()
                    shap.summary_plot(shap_v, X_vis, show=False, plot_size=(5, 5))
                    st.pyplot(plt.gcf())
                    
                with col_s2:
                    st.markdown("##### Grade 6 (Highest) Drivers")
                    if isinstance(shap_values, list):
                        shap_v = shap_values[-1][:, keep_idx]
                    else:
                        shap_v = shap_values[:, keep_idx, -1]
                        
                    plt.figure()
                    shap.summary_plot(shap_v, X_vis, show=False, plot_size=(5, 5))
                    st.pyplot(plt.gcf())
                    
            except Exception as e:
                st.error(f"SHAP Visualization Error: {e}")

# ==============================================================================
# TAB 6: PREDICTOR
# ==============================================================================
with tab_pred:
    st.subheader("AI ESG Predictor & Advisor")
    
    if 'final' in models:
        final_model = models['final']
        sub_search, sub_sim = st.tabs(["🔍 Company Search", "🎛️ Feature Simulation"])
        
        with sub_search:
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                search_term = st.text_input("기업명/코드 검색", placeholder="예: 삼성전자")
            with c2:
                years = sorted(full_search_df['year'].unique(), reverse=True)
                t_year = st.selectbox("예측 연도", years)
            with c3:
                st.write("")
                st.write("")
                btn_search = st.button("검색 실행")
                
            if btn_search and search_term:
                found = full_search_df[
                    (full_search_df['corp_name'].str.contains(search_term)) & 
                    (full_search_df['year'] == t_year)
                ]
                
                if found.empty:
                    st.error("데이터 없음")
                else:
                    target = found.iloc[0]
                    input_df = pd.DataFrame([target])
                    model_in = pd.DataFrame(0, index=[0], columns=X_adv.columns)
                    
                    for c in X_adv.columns:
                        if c in input_df: model_in[c] = input_df[c].values
                        elif c.startswith('IND_') and 'IND' in input_df:
                            if f"IND_{input_df['IND'].values[0]}" == c: model_in[c] = 1
                            
                    prob = final_model.predict_proba(model_in)[0]
                    pred = le.inverse_transform([np.argmax(prob)])[0]
                    
                    st.divider()
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("AI 예측 등급", f"{pred}")
                    with m2:
                        real = target['ESG'] if 'ESG' in target else "-"
                        st.metric("실제 등급", f"{real}")
                    with m3:
                        st.metric("Model Reliability (AUC)", "0.829")
                    
                    st.progress(float(max(prob)))
                    st.caption(f"Instance Confidence: {max(prob)*100:.1f}%")

        with sub_sim:
            st.info("📊 각 지표의 평균값(Mean)으로 초기화된 상태에서 시뮬레이션을 시작합니다.")
# ----------------------------------------------------------------------------------------            
            # 1. 모델 선택 UI 추가
            model_choice = st.radio("🎯 분석 모델 선택", ["기본 모델 (19개)", "확장형 모델 (21개 - 전년도 ESG등급 필요)"], horizontal=True)
            is_extended = "확장형" in model_choice
# ----------------------------------------------------------------------------------------


            defaults = X_adv.mean()
            # 15 Features Requested
            req_feats = ['SGAE_R', 'Fe_R', 'Re_R', 'SA', 'Pay_Gap', 'W_YEAR', 'TQ', 'SGR', 'MSE', 'FOR', 'DIV', 'DIR_FE', 'LEV', 'ROA', 'DIR_OUT']
            # Binary Cols
            binary_cols = ['DIV_enco', 'DIR_FE_enco'] 
            
            with st.form("sim_form"):
                inputs = {}
                cols = st.columns(4)
                
                # [기본 19개 변수 입력 그리드]
                idx = 0
                for c in X_adv.columns:
                    # Skip IND_
                    if c.startswith('IND_'): continue
                    
                    with cols[idx % 4]:
                        if c in binary_cols:
                            inputs[c] = st.selectbox(c, [0, 1], index=0)
                        elif c in req_feats: 
                            val = float(defaults[c])
                            inputs[c] = st.number_input(c, value=val)
                        else:
                            val = float(defaults[c])
                            inputs[c] = st.number_input(c, value=val)
                    idx += 1
                inds = [c.replace('IND_', '') for c in X_adv.columns if c.startswith('IND_')]
                sel_ind = st.selectbox("산업군", inds)


# ----------------------------------------------------------------------------------------
                # 2. 확장형 변수 입력 필드 추가
                esg_lag_val = 0
                if is_extended:
                    st.divider()
                    st.subheader("확장 변수 설정")
                    lag_col1, lag_col2 = st.columns(2)
                    with lag_col1:
                        # 전년도 등급을 숫자로 매핑 (기존 매핑 활용)
                        lag_label = st.selectbox("전년도 ESG 등급 (ESG_lag)", ["S", "A+", "A", "B+", "B", "C", "D"], index=2)
                        esg_mapping_rev = {"S": 7, "A+": 6, "A": 5, "B+": 4, "B": 3, "C": 2, "D": 1}
                        esg_lag_val = esg_mapping_rev[lag_label]
                    with lag_col2:
                        # 상호작용 변수는 자동 계산됨을 안내
                        st.info(f"**상호작용 변수 자동 계산**: A_SIZE × FOR")

# ----------------------------------------------------------------------------------------                
                
                btn_run = st.form_submit_button("시뮬레이션 실행")
            
            if btn_run:
# ----------------------------------------------------------------------------------------               
                # 3. 모델 선택 로직
                if is_extended:
                    current_model = models.get('ext_model')
                    # 확장형은 롤링 윈도우가 아닌 단일 모델일 수 있으므로 구조 대응
                    explainer_model = current_model
                    scaler_obj = models.get('ext_scaler')
                else:
                    current_model = models['final']
                    explainer_model = current_model.estimators[-1]['model']
                    scaler_obj = current_model.estimators[-1]['scaler']
# ----------------------------------------------------------------------------------------

                sim_df = pd.DataFrame([inputs])
                model_in = pd.DataFrame(0, index=[0], columns=X_adv.columns)
                for c in X_adv.columns:
                    if c in sim_df: model_in[c] = sim_df[c]
                    if c == f"IND_{sel_ind}": model_in[c] = 1
# ----------------------------------------------------------------------------------------                
                # ✅ 확장 변수 추가 로직
                if is_extended:
                    model_in["ESG_lag"] = esg_lag_val
                    model_in["A_SIZE_FOR_inter"] = float(inputs["A_SIZE"]) * float(inputs["FOR"])


                # 스케일러가 알고 있는 순서대로 정렬 후 변환
                model_in_aligned = model_in.reindex(columns=scaler_obj.feature_names_in_, fill_value=0)
                X_scaled = scaler_obj.transform(model_in_aligned)

# ----------------------------------------------------------------------------------------

                prob = final_model.predict_proba(model_in)[0]
                curr_idx = np.argmax(prob)
                curr_grade = le.inverse_transform([curr_idx])[0]
                
                c_res, c_radar = st.columns([1, 2])
                with c_res:
                    st.metric("Simulated Grade", curr_grade)
                    st.metric("Model Reliability", "0.829")
                with c_radar:
                    fig_radar = go.Figure(go.Scatterpolar(
                        r=prob, theta=le.classes_, fill='toself', line_color=COLOR_MAIN
                    ))
                    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), height=400)
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # st.markdown("### 🤖 AI Improvement Strategy")
                
                # hierarchy = ['D', 'C', 'B', 'B+', 'A', 'A+', 'S']
                # valid_hierarchy = [g for g in hierarchy if g in le.classes_]
                
                # if curr_grade in valid_hierarchy:
                #     current_rank = valid_hierarchy.index(curr_grade)
                #     if current_rank < len(valid_hierarchy) - 1:
                #         target_grade = valid_hierarchy[current_rank + 1]
                #         target_idx = le.transform([target_grade])[0]
                        
                #         st.write(f"**Goal: {curr_grade} $\\rightarrow$ {target_grade}** 달성을 위한 주요 변수 제안")
                        
                #         advice = []
                #         base_prob = prob[target_idx]
                        
                #         for f in req_feats:
                #             if f not in inputs: continue
                #             temp_in = model_in.copy()
                #             val = temp_in.loc[0, f]
                #             if f in binary_cols: continue
                            
                #             delta = val * 0.1 if val != 0 else 0.01
                            
                #             temp_in.loc[0, f] = val + delta
                #             p_up = final_model.predict_proba(temp_in)[0][target_idx]
                            
                #             temp_in.loc[0, f] = val - delta
                #             p_down = final_model.predict_proba(temp_in)[0][target_idx]
                            
                #             if f == 'LEV':
                #                 if p_down > base_prob: advice.append((f, "감소(-)", (p_down - base_prob)*100))
                #             elif f in ['Pay_Gap', 'Fe_R']:
                #                 if val < 0 and p_up > base_prob: advice.append((f, "증가(+)", (p_up - base_prob)*100))
                #                 elif val > 0 and p_down > base_prob: advice.append((f, "감소(-)", (p_down - base_prob)*100))
                #             else:
                #                 if p_up > base_prob: advice.append((f, "증가(+)", (p_up - base_prob)*100))
                                
                #         advice.sort(key=lambda x: x[2], reverse=True)
                #         if advice:
                #             for f, direct, gain in advice[:3]:
                #                 st.markdown(f"- **{f}** {direct}: 확률 **+{gain:.2f}%p** 상승 예상")
                #         else:
                #             st.info("현재 변수 조정으로는 유의미한 등급 상승 확률을 찾기 어렵습니다.")
                #     else:
                #         st.success("이미 최고 등급입니다!")



            
            if btn_run:

# ----------------------------------------------------------------------------------------
                # --- [A] 사용자가 선택한 모드에 따라 모델과 저울(Scaler) 바구니 채우기 ---
                if is_extended:
                    # 확장형 선택 시: 확장형 전용 모델과 저울 사용
                    current_model = models.get('ext_model')
                    scaler_obj = models.get('ext_scaler') 
                    explainer_model = current_model # SHAP 분석용 모델
                else:
                    # 기본형 선택 시: 기본형의 롤링 윈도우 모델 중 마지막 저울과 모델 꺼내기
                    current_ensemble = models['final']
                    scaler_obj = current_ensemble.estimators[-1]['scaler']
                    explainer_model = current_ensemble.estimators[-1]['model']
# ----------------------------------------------------------------------------------------                
                
                
                # 1. 예측 실행
                sim_df = pd.DataFrame([inputs])
                model_in = pd.DataFrame(0, index=[0], columns=X_adv.columns)
                for c in X_adv.columns:
                    if c in sim_df: model_in[c] = sim_df[c]
                    if c == f"IND_{sel_ind}": model_in[c] = 1

# ----------------------------------------------------------------------------------------               
                # ✅ 확장형 모델일 때만 신규 변수 2개 추가

                if is_extended:
                    model_in["ESG_lag"] = esg_lag_val
                    model_in["A_SIZE_FOR_inter"] = float(inputs["A_SIZE"]) * float(inputs["FOR"])


                # 스케일러가 알고 있는 순서대로 정렬 후 변환
                model_in_aligned = model_in.reindex(columns=scaler_obj.feature_names_in_, fill_value=0)
                X_scaled = scaler_obj.transform(model_in_aligned)

# ----------------------------------------------------------------------------------------

                prob = final_model.predict_proba(model_in)[0]
                curr_idx = np.argmax(prob)
                curr_grade = le.inverse_transform([curr_idx])[0]

# -----------------------------------------------------------------------------------------
# ---------------------------------------------------------
                # 4. SHAP 분석 (모델에 따라 19개 vs 21개 자동 전환)
                # ---------------------------------------------------------
                st.divider()
                st.subheader("🔎 SHAP 기반 상세 분석")
                
                try:
                    explainer = shap.TreeExplainer(explainer_model)
                    shap_values = explainer.shap_values(X_scaled)
                    
                    if isinstance(shap_values, list):
                        vals_for_class = shap_values[curr_idx][0, :]
                    else:
                        vals_for_class = shap_values[0, :, curr_idx]

                    # 💡 핵심: 산업군(IND_)만 제외하면, 나머지는 모델이 가진 피처(19개 또는 21개)가 자동으로 남음!
                    feature_names = list(scaler_obj.feature_names_in_)
                    non_ind_indices = [i for i, name in enumerate(feature_names) if not name.startswith("IND_")] 
                    
                    new_values = np.array([vals_for_class[i] for i in non_ind_indices], dtype=np.float64)
                    new_feature_names = [feature_names[i] for i in non_ind_indices]
                    new_data = np.array([round(float(model_in_aligned.iloc[0][feature_names[i]]), 2) for i in non_ind_indices], dtype=np.float64)

                    base_val = explainer.expected_value
                    if isinstance(base_val, (list, np.ndarray)):
                        base_val = base_val[curr_idx]
                    
                    exp = shap.Explanation(
                        values=new_values,
                        base_values=float(base_val),
                        data=new_data,
                        feature_names=new_feature_names
                    )

                    with st.expander(f"📝 {model_choice} 상세 분석 SHAP", expanded=True):
                        # [핵심 수정] 마이너스 부호 설정을 가장 강력하게 적용
                        import matplotlib.pyplot as plt
                        import koreanize_matplotlib

                        # 1. 마이너스 부호 깨짐 방지 (반드시 False여야 함)
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        # 2. 폰트 재설정 (혹시 모를 설정 덮어쓰기 방지)
                        # NanumGothic이 없으면 기본 한글 폰트로 대체하도록 설정
                        plt.rcParams['font.family'] = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'sans-serif']

                        total_features = len(new_values)
                        
                        # 3. Figure 객체 명시적 생성
                        fig, ax = plt.subplots(figsize=(10, 0.6 * total_features + 2))
                        
                        # 4. SHAP 그래프 그리기
                        shap.plots.waterfall(exp, show=False, max_display=total_features)
                        
                        # 5. 타이틀 설정
                        plt.title(f"{curr_grade} 등급 판정 핵심 요인 (변수 {total_features}개)", fontsize=15, pad=30)
                        
                        # 6. Streamlit에 출력
                        st.pyplot(fig)
                        plt.close(fig)



                    # ---------------------------------------------------------
                    # 5. 전략 제안 자동 생성
                    # ---------------------------------------------------------
                    st.subheader("💡 AI 맞춤형 전략 처방")
                    tmp_analysis_df = pd.DataFrame({"feature": new_feature_names, "shap": new_values})
                    tmp_analysis_df["abs_shap"] = tmp_analysis_df["shap"].abs()
                    tmp_analysis_df = tmp_analysis_df.sort_values("abs_shap", ascending=False)
                    
                    advice_pack = make_shap_based_advice(tmp_analysis_df, model_in, STRATEGIC_ADVICE, top_k=3)
                    
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.markdown("##### ✅ 유지 및 강화 전략")
                        for item in advice_pack["pos"]:
                            st.success(f"**{item['feature']}**: {item['text']}")
                    with col_neg:
                        st.markdown("##### ⚠️ 개선 및 보완 전략")
                        for item in advice_pack["neg"]:
                            st.warning(f"**{item['feature']}**: {item['text']}")

                except Exception as e:
                    st.error(f"SHAP 분석 중 오류 발생: {e}")