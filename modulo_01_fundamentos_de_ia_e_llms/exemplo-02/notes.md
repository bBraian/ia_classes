# 🦆 Exemplo 02 - Duck Hunt AI (Game Dev & Computer Vision)

Neste projeto de jogo, usamos Inteligência Artificial para reconhecimento de imagens em tempo real para automação.

## 🚀 O que foi trabalhado:
- **YOLOv5 (You Only Look Once):** Implementação de um dos modelos mais rápidos e precisos do mundo para detecção de múltiplos objetos.
- **Web Workers:** Uso de `worker.js` para garantir que o processamento pesado da IA não "congele" a animação de 60 FPS do jogo.
- **PixiJS & Green Sock (GSAP):** Renderização de alto desempenho e animações suaves integradas com a detecção de IA.
- **Bounding Boxes:** O modelo localiza os patos e retorna as coordenadas `[x, y, largura, altura]`.
- **Automação:** Programamos a lógica para que a IA "atire" automaticamente nos alvos assim que forem detectados.

## 💡 Insights:
- **Performance (60 FPS):** Em aplicações de tempo real, a separação da IA (em threads/workers) é obrigatória para evitar "lag".
- **Hardware Acceleration:** O TensorFlow.js tira proveito do WebGL/WebGPU para acelerar o cálculo matricial da rede neural.
- **Model Training:** Como o jogo tem elementos específicos, um modelo focado (YOLO) é muito superior a uma rede neural simples para detecção espacial.
