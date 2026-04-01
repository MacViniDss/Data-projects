import streamlit as st
import joblib
import pandas as pd

# Carregar modelo e scaler
@st.cache_resource
def load_model():
    modelo = joblib.load('modelo_fraude_rf.pkl')
    scaler = joblib.load('scaler.pkl')
    return modelo, scaler

modelo, scaler = load_model()

st.title("Detector de Fraudes em Cartão de Crédito")
st.markdown("### Preencha os dados da transação")

# Inputs
col1, col2 = st.columns(2)
with col1:
    time = st.number_input("Time (segundos)", value=0.0, step=1.0)
with col2:
    amount = st.number_input("Amount (R$)", value=100.0, step=0.01)

st.subheader("Features V1 até V28 (valores PCA)")
v_values = []
cols_v = st.columns(7)  # 28 features em 7 colunas bonitas

for i in range(1, 29):
    with cols_v[(i-1) % 7]:
        v = st.number_input(
            f"V{i}",
            value=0.0,
            step=0.01,
            key=f"v{i}",
            format="%.4f"
        )
        v_values.append(v)

# Botão para fazer a previsão
if st.button("🔍 Prever Fraude", type="primary"):
    # Monta o DataFrame com nomes em MINÚSCULO (igual ao treino)
    data = {
        'time': [time],
        **{f'v{i}': [v_values[i-1]] for i in range(1, 29)},
        'amount': [amount]
    }
    df_input = pd.DataFrame(data)

    # Aplica o scaler (exatamente como no treino)
    cols_to_scale = ['time', 'amount']
    df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])

    # Garante ordem exata das colunas
    df_input = df_input[modelo.feature_names_in_]

    # Faz a previsão
    prob = modelo.predict_proba(df_input)[0][1]
    pred = 1 if prob >= 0.5 else 0

    # Mostra resultado
    if pred == 1:
        st.error("🚨 **FRAUDE DETECTADA!**")
        st.metric("Probabilidade de fraude", f"{prob*100:.2f}%")
    else:
        st.success("✅ **Transação NORMAL**")
        st.metric("Probabilidade de fraude", f"{prob*100:.2f}%")

    st.caption(f"Score do modelo: {prob:.4f}")
