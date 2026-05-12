# 🎓 Aula 01 - Fundamentos do TensorFlow.js

Nesta aula, focamos na base teórica e prática de Redes Neurais usando JavaScript.

## 🚀 O que foi trabalhado:
- **Tensors (Tensores):** Entendemos que tudo para a IA são números organizados em tensores (2D neste caso).
- **Modelo Sequencial:** Criamos uma rede onde a informação flui em uma única direção (`tf.sequential`).
- **Camadas Densas (`tf.layers.dense`):**
    - `units: 80`: Definimos a complexidade da rede.
    - `activation: 'relu'`: Usada como um filtro para ativar neurônios com valores positivos.
- **Camada de Saída & Softmax:**
    - `activation: 'softmax'`: Transforma os valores de saída em probabilidades que somam 100%.
- **Treinamento (`model.fit`):**
    - `epochs: 100`: O modelo viu os mesmos dados 100 vezes para aprender os padrões.
    - `shuffle: true`: Evita que a rede decore a ordem dos dados.

## 💡 Insights:
- **Normalização:** Converter idade (0-100) e categorias (cores) para uma escala de 0 a 1 é vital.
- **One-Hot Encoding:** Representar categorias como vetores (ex: `[1, 0, 0]`) permite que a rede entenda classes distintas sem atribuir "peso" numérico maior a uma cidade ou cor.
