from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
modelo = joblib.load('modelo_fraude_rf.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df_input = pd.DataFrame([data])
    # Confirmando os Inputs
    print("✅ Colunas recebidas:", list(df_input.columns))          # debug
    print("✅ Colunas esperadas:", list(modelo.feature_names_in_))  # debug
    
    # Escalonamento
    cols_to_scale = ['time', 'amount']
    df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])
    
    # Garante ordem exata que o modelo viu no treino
    df_input = df_input[modelo.feature_names_in_]
    
    prob = modelo.predict_proba(df_input)[0][1]
    pred = 1 if prob >= 0.5 else 0
    
    return jsonify({
        'fraude': bool(pred),
        'probabilidade_fraude': float(prob),
        'mensagem': '🚨 ALERTA: Transação suspeita!' if pred == 1 else '✅ Transação normal'
    })
if __name__ == '__main__':
    app.run(debug=True, port=5000)
