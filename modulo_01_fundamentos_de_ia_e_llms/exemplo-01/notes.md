# 🎓 Exemplo 01 - E-commerce Recommendation System

Neste projeto, aplicamos IA em um sistema de e-commerce real para rastrear o comportamento do usuário e preparar recomendações.

## 🚀 O que foi trabalhado:
- **Arquitetura modular:** Separação de responsabilidades (MVC parcial) com `controller`, `view` e `service`.
- **Persistência de dados:** Uso do `sessionStorage` para simular uma base de dados local onde os produtos e carrinhos são salvos.
- **Preparação para Recomendações:**
    - Coleta de cliques e compras.
    - Estruturação do histórico do usuário para ser usado como base de treino pela IA futuramente.
- **TensorFlow.js (Integração):** Planejamento para um motor de recomendações baseado na similaridade de usuários.

## 💡 Insights:
- **Dados são o Combustível:** Antes de treinar a IA, é necessário saber *o que* coletar (cliques, tempo de visualização, compras).
- **Escalabilidade:** Organizar o código em classes facilita a integração da IA conforme o app cresce.
- **UX com IA:** Como a IA pode melhorar a jornada de compra oferecendo o produto certo na hora certa.
