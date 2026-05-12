# 🌐 Exemplo 04 - Web AI (Ajustes Finos: Temperature & TopK)

Neste projeto, exploramos os parâmetros avançados de modelos de IA no navegador para controlar o comportamento da geração de texto.

## 🚀 O que foi trabalhado:
- **Temperature (Criatividade):**
    - `0.1 a 1.0`: Quanto maior, mais criativa e imprevisível a IA se torna.
    - `Baixo`: Respostas precisas e determinísticas.
- **TopK (Diversidade):**
    - Limita o número de palavras candidatas que a IA considera para gerar o próximo token.
    - Ajuda a manter a IA "focada" nos tópicos mais prováveis.
- **Gerenciamento de Download:**
    - Monitoramento do progresso (`downloadprogress`) e aviso ao usuário quando o modelo (Gemini Nano) precisa ser baixado.
- **AbortController:** Uso de sinais para interromper a geração a qualquer momento, economizando CPU.

## 💡 Insights:
- **Ajuste Fino:** Cada aplicação pede uma temperatura diferente; Chats precisam de mais criatividade (alto), enquanto extratores de dados ou assistentes técnicos pedem precisão (baixo).
- **Tratamento de Erros:** É vital verificar a disponibilidade do modelo (`LanguageModel.availability`) antes de iniciar.
- **Eficiência:** Destruir sessões antigas (`session.destroy`) para liberar memória da GPU/CPU no navegador.
