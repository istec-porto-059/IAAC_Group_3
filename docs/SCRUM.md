Registos SCRUM - Módulo Email Fraud Detection

Daily Scrum Meetings
Aula: 21/09/2026 (Sprint 1)
    O que fiz: Estruturação inicial do pipeline de NLP para o dataset Enron.
    O que vou fazer: Limpeza de dados via Regex e tratamento de desequilíbrio de classes.
    Impedimentos: Alto volume de e-mails normais em relação ao número de fraudes (resolvido com Undersampling).
Aula: 23/09/2026 (Sprint 2)
    O que fiz: Implementação do data_prep.py e extração de features TF-IDF no features.py.
    O que vou fazer: Treino e avaliação dos modelos (Logistic Regression, Naive Bayes, Random Forest).
    Impedimentos: Ajuste dos imports do Python após reestruturação modular para a pasta src/modules/email/ (resolvido).

Sprint Review 23/09/2026
    Meta do Sprint: Entregar um pipeline completo de Data Preparation e um modelo funcional para deteção de fraudes em e-mail.
Entregáveis Concluídos:
    Tratamento de Dados: Amostragem balanceada (2.327 fraudes + 10.000 normais) e tokenização TF-IDF (5.000 palavras).
Modelo Selecionado:
    Regressão Logística com class_weight='balanced'.
Métrica Principal:
    Recall de 94.85% na classe de fraude (minimização de Falsos Negativos em cibersegurança).
Artefatos Gerados:
    Modelos exportados em models/model.pkl e models/vectorizer.pkl.

Sprint Retrospective (A cada 2 Aulas)
O que correu bem?
    A arquitetura modular (data_prep.py, features.py, train.py, predict.py) facilitou o isolamento e teste de erros. A escolha do Recall como métrica de decisão garantiu um modelo altamente sensível a ameaças.
O que pode ser melhorado?
    Melhorar a gestão do histórico de commits no Git para evitar resolução de conflitos de última hora. Automatizar a execução dos testes de inferência.
Ações de Melhoria para o Próximo Sprint:
    Manter o padrão de mensagens de commit (Conventional Commits). Consolidar a integração da LLM no agent.py para geração de relatórios de incidentes