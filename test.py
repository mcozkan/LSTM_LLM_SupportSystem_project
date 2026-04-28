import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import cohere

load_dotenv()

st.set_page_config(
    page_title="Sales Forecast Decision Support",
    page_icon="📈",
    layout="wide"
)

# =========================
# SESSION STATE
# =========================

if "decision_support_outputs" not in st.session_state:
    st.session_state.decision_support_outputs = {}

if "temperature_outputs" not in st.session_state:
    st.session_state.temperature_outputs = {}

if "final_decision_output" not in st.session_state:
    st.session_state.final_decision_output = None

if "human_decision_log" not in st.session_state:
    st.session_state.human_decision_log = None


st.title("📈 LSTM + LLM Sales Forecast Decision Support System")

st.markdown("""
Bu demo, LSTM tabanlı satış tahmin çıktısını sentetik olarak üretir.

LLM tahmin üretmez. Yalnızca LSTM tahminini yorumlar, riskleri değerlendirir,
insan onayı gerekip gerekmediğini açıklar ve karar destek katmanı olarak çalışır.
""")

st.sidebar.header("Experiment Settings")

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["GPT", "Gemini", "Cohere"]
)

run_mode = st.sidebar.radio(
    "Run Mode",
    ["Single Model", "Compare All Models"]
)

temperature = st.sidebar.selectbox(
    "Temperature",
    [0.0, 0.2, 0.7],
    index=1
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon Days",
    min_value=7,
    max_value=60,
    value=30,
    step=1
)

external_event = st.sidebar.selectbox(
    "External Event",
    [
        "None",
        "Special holiday expected",
        "Aggressive competitor campaign",
        "Sudden weather disruption",
        "Regional event increasing demand"
    ]
)

campaign_active = st.sidebar.checkbox(
    "Company campaign active",
    value=True
)

model_map = {
    "GPT": "gpt-4o-mini",
    "Gemini": "gemini-2.0-flash",
    "Cohere": "command-r-08-2024"
}

st.sidebar.divider()
st.sidebar.write(f"**Selected provider:** {provider}")
st.sidebar.write(f"**Selected model:** {model_map[provider]}")
st.sidebar.write(f"**Temperature:** {temperature}")

with st.sidebar.expander("Dependency check"):
    st.code(
        "pip install streamlit pandas numpy plotly python-dotenv openai google-generativeai cohere",
        language="bash"
    )


def generate_synthetic_lstm_forecast(
    days: int = 30,
    campaign: bool = True,
    external_event_name: str = "None"
) -> pd.DataFrame:
    np.random.seed(42)

    dates = pd.date_range(
        start=pd.Timestamp.today().normalize(),
        periods=days
    )

    base_sales = 1000
    trend = np.linspace(0, 120, days)
    seasonality = 100 * np.sin(np.arange(days) * 2 * np.pi / 7)
    noise = np.random.normal(0, 40, days)

    campaign_effect = 120 if campaign else 0
    lstm_forecast = base_sales + trend + seasonality + campaign_effect + noise

    risk_flag = ["Normal"] * days

    if external_event_name != "None":
        event_start = max(0, days // 2 - 2)
        event_end = min(days, event_start + 5)

        if external_event_name == "Special holiday expected":
            lstm_forecast[event_start:event_end] *= 1.12
            event_label = "Holiday demand risk"

        elif external_event_name == "Aggressive competitor campaign":
            lstm_forecast[event_start:event_end] *= 0.90
            event_label = "Competitor pressure risk"

        elif external_event_name == "Sudden weather disruption":
            lstm_forecast[event_start:event_end] *= 0.85
            event_label = "Weather disruption risk"

        elif external_event_name == "Regional event increasing demand":
            lstm_forecast[event_start:event_end] *= 1.15
            event_label = "Regional demand increase risk"

        else:
            event_label = "External event risk"

        for i in range(event_start, event_end):
            risk_flag[i] = event_label

    lower_bound = lstm_forecast * 0.90
    upper_bound = lstm_forecast * 1.10

    return pd.DataFrame({
        "date": dates,
        "lstm_forecast": lstm_forecast.round(0).astype(int),
        "lower_bound": lower_bound.round(0).astype(int),
        "upper_bound": upper_bound.round(0).astype(int),
        "risk_flag": risk_flag
    })


df = generate_synthetic_lstm_forecast(
    days=forecast_horizon,
    campaign=campaign_active,
    external_event_name=external_event
)

dataset_size = 95000
base_model_smape = 13.2
llm_adjusted_smape = 12.6

m1, m2, m3, m4 = st.columns(4)

m1.metric("Dataset Size", f"{dataset_size:,}")
m2.metric("Base Model", "LSTM Time Series")
m3.metric("Base SMAPE", base_model_smape)
m4.metric("Pilot LLM SMAPE", llm_adjusted_smape)

st.subheader("Synthetic LSTM Forecast Output")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["lstm_forecast"],
        mode="lines+markers",
        name="LSTM Forecast"
    )
)

fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["upper_bound"],
        mode="lines",
        name="Upper Bound",
        line=dict(dash="dash")
    )
)

fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["lower_bound"],
        mode="lines",
        name="Lower Bound",
        line=dict(dash="dash")
    )
)

risk_df = df[df["risk_flag"] != "Normal"]

if not risk_df.empty:
    fig.add_trace(
        go.Scatter(
            x=risk_df["date"],
            y=risk_df["lstm_forecast"],
            mode="markers",
            name="Risk Period",
            marker=dict(size=12, symbol="diamond")
        )
    )

fig.update_layout(
    title="Synthetic LSTM Sales Forecast",
    xaxis_title="Date",
    yaxis_title="Forecasted Sales",
    height=450
)

st.plotly_chart(fig, use_container_width=True)
st.dataframe(df, use_container_width=True)

context = {
    "task": "sales forecast adjustment support",
    "dataset_size": dataset_size,
    "base_model": "lstm_time_series",
    "forecast_horizon_days": forecast_horizon,
    "evaluation_metric": "smape",
    "base_model_smape": base_model_smape,
    "llm_adjusted_smape": llm_adjusted_smape,
    "constraints": {
        "human_approval_required": True,
        "explainability_required": True,
        "business_critical_decisions": True
    },
    "business_context": {
        "campaign_active": campaign_active,
        "external_event": external_event
    },
    "lstm_forecast_sample": df.head(10).to_dict(orient="records"),
    "risk_summary": {
        "risk_days_count": int((df["risk_flag"] != "Normal").sum()),
        "risk_types": list(df["risk_flag"].unique())
    }
}

with st.expander("View Context JSON"):
    st.json(context)

system_prompt = """
You are a decision support assistant for sales forecasting.

You are NOT a forecasting model.
You are NOT allowed to generate a new sales forecast from scratch.

Your task is to evaluate the LSTM model output using the provided context and determine whether the forecast may require human-reviewed adjustment.

Strict rules:
- Do not replace the LSTM model.
- Do not produce a new numerical forecast.
- Do not invent external events.
- Do not quantify adjustment values.
- Only suggest directional adjustment: Increase / Decrease / No Change / Review Required.
- Clearly state uncertainty.
- Explain why human approval is required.
- Avoid overconfident conclusions.
- Treat the LSTM as the primary forecasting model.
- Treat yourself only as a decision-support and explanation layer.
"""

user_prompt = f"""
Evaluate the following LSTM-based sales forecasting context.

Context:
{json.dumps(context, indent=2, default=str)}

Return your answer in this exact format:

Assessment:
Adjustment needed? Yes / No / Conditional

Adjustment direction:
Increase / Decrease / No Change / Review Required

Reasoning:
- Reason 1
- Reason 2
- Reason 3

Risks:
- Risk 1
- Risk 2

Uncertainty:
- Uncertainty 1
- Uncertainty 2

Human approval:
- Explain why human approval is required.

Final note:
- Explain whether the LLM is acting as a forecasting model or a decision-support layer.
"""

with st.expander("View System Prompt"):
    st.code(system_prompt, language="text")

with st.expander("View User Prompt"):
    st.code(user_prompt, language="text")


def get_api_key(selected_provider: str) -> str | None:
    api_key_map = {
        "GPT": "OPENAI_API_KEY",
        "Gemini": "GOOGLE_API_KEY",
        "Cohere": "COHERE_API_KEY"
    }

    return os.getenv(api_key_map[selected_provider])


def run_gpt(
    system_prompt_text: str,
    user_prompt_text: str,
    temp: float,
    api_key: str
) -> str:
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model_map["GPT"],
            temperature=temp,
            max_tokens=700,
            messages=[
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": user_prompt_text}
            ]
        )
    except Exception as exc:
        raise RuntimeError(
            "GPT isteği başarısız oldu. API key, model erişimi, quota veya bağlantıyı kontrol et."
        ) from exc

    return response.choices[0].message.content


def run_gemini(
    system_prompt_text: str,
    user_prompt_text: str,
    temp: float,
    api_key: str
) -> str:
    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name=model_map["Gemini"],
            system_instruction=system_prompt_text
        )

        response = model.generate_content(
            user_prompt_text,
            generation_config={
                "temperature": temp,
                "max_output_tokens": 700,
                "candidate_count": 1
            },
            request_options={
                "timeout": 90
            }
        )
    except Exception as exc:
        raise RuntimeError(
            "Gemini isteği başarısız oldu. Olası nedenler: timeout, rate limit, "
            "geçici servis yoğunluğu, API key/billing problemi veya model erişimi."
        ) from exc

    if response is None:
        raise RuntimeError("Gemini boş response döndürdü.")

    try:
        return response.text
    except Exception as exc:
        raise RuntimeError(
            "Gemini response.text okunamadı. Cevap safety filter, boş candidate "
            "veya tamamlanmamış response nedeniyle dönmemiş olabilir."
        ) from exc


def run_cohere(
    system_prompt_text: str,
    user_prompt_text: str,
    temp: float,
    api_key: str
) -> str:
    client = cohere.ClientV2(api_key=api_key)

    try:
        response = client.chat(
            model=model_map["Cohere"],
            temperature=temp,
            max_tokens=700,
            messages=[
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": user_prompt_text}
            ]
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cohere isteği başarısız oldu. Kullanılan model: {model_map['Cohere']}. "
            "API key, model erişimi, trial limit, rate limit veya quota durumunu kontrol et."
        ) from exc

    try:
        return response.message.content[0].text
    except Exception as exc:
        raise RuntimeError("Cohere cevap formatı beklenen yapıda değil.") from exc


def run_selected_provider(
    selected_provider: str,
    system_prompt_text: str,
    user_prompt_text: str,
    temp: float
) -> str:
    api_key = get_api_key(selected_provider)

    if not api_key:
        raise ValueError(
            f"{selected_provider} API key bulunamadı. .env dosyasını kontrol edin."
        )

    if selected_provider == "GPT":
        return run_gpt(system_prompt_text, user_prompt_text, temp, api_key)

    if selected_provider == "Gemini":
        return run_gemini(system_prompt_text, user_prompt_text, temp, api_key)

    if selected_provider == "Cohere":
        return run_cohere(system_prompt_text, user_prompt_text, temp, api_key)

    raise ValueError("Geçersiz provider seçimi.")


def render_provider_error(selected_provider: str, error: Exception) -> None:
    if isinstance(error, ValueError):
        st.warning(f"{selected_provider} çalıştırılamadı: {error}")
    else:
        st.error(f"{selected_provider} runtime hatası: {error}")


# =========================
# TABBED TEST UI
# =========================

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "LLM Decision Support",
    "Temperature Experiment",
    "Final Production Decision",
    "Human-in-the-Loop Approval"
])

# =========================
# TAB 1: LLM DECISION SUPPORT
# =========================

with tab1:
    st.subheader("LLM Decision Support Output")

    run_button = st.button("Run LLM Decision Support")

    if run_button:
        st.session_state.decision_support_outputs = {}

        providers_to_run = [provider] if run_mode == "Single Model" else ["GPT", "Gemini", "Cohere"]

        for selected_provider in providers_to_run:
            with st.spinner(f"{selected_provider} analiz ediyor..."):
                try:
                    output = run_selected_provider(
                        selected_provider=selected_provider,
                        system_prompt_text=system_prompt,
                        user_prompt_text=user_prompt,
                        temp=temperature
                    )

                    st.session_state.decision_support_outputs[selected_provider] = {
                        "model": model_map[selected_provider],
                        "temperature": temperature,
                        "output": output
                    }

                except Exception as e:
                    st.session_state.decision_support_outputs[selected_provider] = {
                        "model": model_map[selected_provider],
                        "temperature": temperature,
                        "error": str(e)
                    }

    if st.session_state.decision_support_outputs:
        for selected_provider, result in st.session_state.decision_support_outputs.items():
            st.markdown(f"## {selected_provider} Output")
            st.caption(
                f"Model: {result['model']} | Temperature: {result['temperature']}"
            )

            if "output" in result:
                st.markdown(result["output"])
            else:
                st.error(f"{selected_provider} runtime hatası: {result['error']}")
    else:
        st.info("Henüz LLM Decision Support analizi çalıştırılmadı.")

# =========================
# TAB 2: TEMPERATURE EXPERIMENT
# =========================

with tab2:
    st.subheader("Temperature Experiment")

    st.markdown("""
    Bu bölüm aynı provider, aynı context ve aynı user prompt ile farklı temperature değerlerini karşılaştırır.
    """)

    run_temp_experiment = st.button("Run Temperature Experiment: 0.0 vs 0.7")

    if run_temp_experiment:
        st.session_state.temperature_outputs = {}

        temp_values = [0.0, 0.7]

        for temp_value in temp_values:
            with st.spinner(f"{provider} temperature={temp_value} ile analiz ediyor..."):
                try:
                    output = run_selected_provider(
                        selected_provider=provider,
                        system_prompt_text=system_prompt,
                        user_prompt_text=user_prompt,
                        temp=temp_value
                    )

                    st.session_state.temperature_outputs[temp_value] = {
                        "provider": provider,
                        "model": model_map[provider],
                        "temperature": temp_value,
                        "output": output
                    }

                except Exception as e:
                    st.session_state.temperature_outputs[temp_value] = {
                        "provider": provider,
                        "model": model_map[provider],
                        "temperature": temp_value,
                        "error": str(e)
                    }

    if st.session_state.temperature_outputs:
        for temp_value, result in st.session_state.temperature_outputs.items():
            st.markdown(
                f"## {result['provider']} Output | Temperature: {result['temperature']}"
            )
            st.caption(f"Model: {result['model']}")

            if "output" in result:
                st.markdown(result["output"])
            else:
                st.error(
                    f"{result['provider']} temperature={result['temperature']} runtime hatası: {result['error']}"
                )
    else:
        st.info("Henüz Temperature Experiment çalıştırılmadı.")

# =========================
# TAB 3: FINAL PRODUCTION DECISION
# =========================

with tab3:
    st.subheader("Final Production Decision - Temperature 0.2")

    st.markdown("""
    Bu bölüm case study'nin final karar formatı için temperature=0.2 kullanır.
    """)

    final_prompt = f"""
Using the same context below, produce a final production decision.

Context:
{json.dumps(context, indent=2, default=str)}

Return your answer exactly in this format:

Recommendation:
Should LLM-supported adjustment be deployed to production?

Reasoning:
- Point 1
- Point 2

Risks:
- Risk 1
- Risk 2

Next actions:
- Action 1
- Action 2

Important:
- Do not generate a new sales forecast.
- Do not replace the LSTM model.
- The LLM must remain a decision-support layer.
- Human approval is required.
"""

    run_final_decision = st.button("Run Final Production Decision")

    if run_final_decision:
        st.session_state.final_decision_output = None

        with st.spinner(f"{provider} final production kararı üretiyor..."):
            try:
                final_output = run_selected_provider(
                    selected_provider=provider,
                    system_prompt_text=system_prompt,
                    user_prompt_text=final_prompt,
                    temp=0.2
                )

                st.session_state.final_decision_output = {
                    "provider": provider,
                    "model": model_map[provider],
                    "temperature": 0.2,
                    "output": final_output
                }

            except Exception as e:
                st.session_state.final_decision_output = {
                    "provider": provider,
                    "model": model_map[provider],
                    "temperature": 0.2,
                    "error": str(e)
                }

    if st.session_state.final_decision_output:
        result = st.session_state.final_decision_output

        st.markdown(
            f"## Final Decision | {result['provider']} | Temperature: {result['temperature']}"
        )
        st.caption(f"Model: {result['model']}")

        if "output" in result:
            st.markdown(result["output"])
        else:
            st.error(f"{result['provider']} runtime hatası: {result['error']}")
    else:
        st.info("Henüz Final Production Decision çalıştırılmadı.")

# =========================
# TAB 4: HUMAN-IN-THE-LOOP
# =========================

with tab4:
    st.subheader("Human-in-the-Loop Approval")

    human_decision = st.radio(
        "Final human decision",
        [
            "Approve LSTM forecast as-is",
            "Send forecast for manual review",
            "Reject LLM suggestion",
            "Request more data"
        ]
    )

    reviewer_comment = st.text_area("Reviewer Comment")

    if st.button("Save Human Decision"):
        decision_log = {
            "timestamp": datetime.now().isoformat(),
            "selected_provider": provider,
            "run_mode": run_mode,
            "temperature": temperature,
            "forecast_horizon_days": forecast_horizon,
            "campaign_active": campaign_active,
            "external_event": external_event,
            "base_model_smape": base_model_smape,
            "llm_adjusted_smape": llm_adjusted_smape,
            "human_decision": human_decision,
            "reviewer_comment": reviewer_comment
        }

        log_file = "human_decision_log.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision_log, ensure_ascii=False) + "\n")

        st.session_state.human_decision_log = decision_log

        st.success("Human decision logged successfully.")

    if st.session_state.human_decision_log:
        st.json(st.session_state.human_decision_log)