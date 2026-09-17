import os
from openai import OpenAI
from src.predict import predict_email

# Inicializa o cliente OpenAI (pode utilizar uma chave em variável de ambiente)
# Em caso de testes sem chave ativa, o script possui modo de fallback simulado.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "SUA_CHAVE_AQUI"))

SYSTEM_PROMPT = """
Você é um Agente Analista do SOC (Security Operations Center) perito em Cibersegurança e Antifraude.
Sua missão é receber o texto de um e-mail analisado pelo classificador de Machine Learning e emitir um 
Relatório de Incidente de Cibersegurança estruturado em Markdown contendo:
1. Resumo Executivo da Ameaça
2. Vetores de Ataque Identificados (Engenharia Social, Urgência, Transferências, etc.)
3. Nível de Risco (Baixo, Médio, Alto, Crítico)
4. Ações Recomendadas de Mitigação para a Equipa de TI
"""

def generate_security_report(email_text: str) -> str:
    """Combina o modelo de ML com o Agente Generativo para auditar o e-mail."""
    # 1. Obter a classificação do modelo ML (Scikit-Learn)
    prediction = predict_email(email_text)
    prob_percent = prediction['fraud_probability'] * 100
    
    prompt = f"""
    --- DADOS DA ANÁLISE DE MACHINE LEARNING ---
    Status de Fraude: {prediction['is_fraud']}
    Probabilidade de Fraude: {prob_percent:.2f}%
    
    --- CONTEÚDO DO E-MAIL ---
    {email_text}
    
    Por favor, gere o relatório de cibersegurança com base nestes dados.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback de simulação caso a chave de API não esteja configurada no ambiente
        return f"""
# 🛡️ Relatório de Incidente de Cibersegurança (Simulação)

**Probabilidade de Fraude:** {prob_percent:.2f}%
**Classificação:** {'ALERTA DE FRAUDE' if prediction['is_fraud'] or prob_percent > 35 else 'NORMAL'}

## 1. Resumo Executivo
O sistema de deteção identificou padrões de linguagem suspeitos no e-mail analisado com um nível de confiança de {prob_percent:.2f}%.

## 2. Indicadores de Comprometimento (IoC)
* **Padrões Textuais:** Uso de palavras-chave associadas a transações urgentes/financeiras.
* **Modelo Utilizado:** Regressão Logística (TF-IDF 5000 features).

## 3. Ações Recomendadas
* Bloquear o remetente no Gateway de E-mail.
* Notificar a equipa de auditoria interna.

*(Nota: Adicione uma OPENAI_API_KEY válida para relatórios dinâmicos completos via LLM)*
"""

if __name__ == "__main__":
    test_email = "URGENT: Confidential wire transfer required immediately to proceed with transaction."
    print("--- A Executar Agente Generativo de Cibersegurança ---")
    report = generate_security_report(test_email)
    print(report)