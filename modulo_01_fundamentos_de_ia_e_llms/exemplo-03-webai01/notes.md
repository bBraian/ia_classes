# 🌐 Exemplo 03 - Web AI (Introdução aos Modelos Nativos do Navegador)

Neste projeto de Web AI, fomos além do treinamento de rede neural e usamos modelos de linguagem (LLM) integrados ao Google Chrome.

## 🚀 O que foi trabalhado:
- **LanguageModel API:** Acesso direto ao modelo Gemini Nano incorporado no navegador.
- **Prompt Streaming:** Em vez de esperar toda a resposta, usamos `promptStreaming` para exibir o texto conforme ele é gerado.
- **Configuração de Contexto (System Role):** Definimos o comportamento da IA (ex: "Você é um assistente claro e objetivo") usando `initialPrompts`.
- **Bibliotecas de Terceiros:** Integração com a biblioteca de Markdown para renderizar as respostas formatadas.

## 💡 Insights:
- **Privacidade total:** Diferente da OpenAI ou Anthropic, os dados do usuário não são enviados para um servidor; o processamento é local (Edge AI).
- **Latência:** Notamos a velocidade incrível da geração local sem delays de internet.
- **Browser-First IA:** A tendência do futuro: a web se torna uma plataforma capaz de rodar modelos generativos complexos nativamente.
