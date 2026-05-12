import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  //primeira camada da rede
  // entrada de 7 posições (idade normalizada + 3 cores + 3 localizadores)

  //  80 neuronios = aqui foi colocado tudo isso pois tem pouca base de treino
  // quanto mais neuronios, mais complexidade a rede pode aprender e consequentemente, mais processamento ela vai usar.

  // A ReLu age como um filtro. É como se ela deixasse somente os dados interessantes.
  // Se a informação que chegou nesse neuronio é positiva, passa para frente
  // se for 0 ou negativa, pode jogar fora, não vai servir para nada.
  model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }));

  // saída: 3 neuronios (um para cada categoria que é a saída)

  //Activation: softmax -> normaliza a saídaa em probabilidade.
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  //compilando o modelo
  // "Adam" (Adaptive Moment Estimation) é um treinador pessoal moderno para redes neurais.
  // Ajusta os pesos de forma eficiente e inteligente aprendendo com histórico de acetos e erros.

  //loss: categoricalCrossentropy: Compara o que o modelo "acha" (os scores de cada categoria) com a resposta certa.
  // ex: a categoria premium será sempre [1,0,0]

  // quanto mais distante da previsão do modelo da resposta correta, maior o erro (loss)

  //Exemplo clássico> classificação de imagens, recomendação, categorização de usuário
  model.compile({ optimizer: "adam", loss: "categoricalCrossentropy", metrics: ["accuracy"] });

  //treinamento do modelo
  /*
    verbose: desabilita o log interno
    epochs: quantidade de vezes que vai rodar com a base de dados passada
    shuffle: para embaralhar os dados (Importante para não "viciar" o modelo)
  */
  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    callbacks: { onEpochEnd: (epoch, log) => console.log(`Epoch: ${epoch}  loss: ${log.loss}`) },
  });

  return model;
}

async function predict(model, personTensorNormalized) {
  // transformar o array js para tensor (tfjs)
  const tfInput = tf.tensor2d(personTensorNormalized);

  //faz a predição (verot de 3 probrabilidades)
  const pred = model.predict(tfInput);
  const predArray = await pred.array();
  return predArray[0].map((prob, i) => ({ prob, i }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

const model = await trainModel(inputXs, outputYs);

const person = { nome: "Zé", idade: 28, cor: "verde", localizacao: "Curitiba" };

//transformar array e normalizar valores (array de arrays, cada item na lista representa 1 pessoa e cada item dentro da lista representa colunas de propriedades)
const personTensorNormalized = [
  [
    0.2, //idade normalizada
    0, //azul
    0, //vermelho
    1, //verde
    0, //SP
    0, //Rio
    1, //Curitiba
  ],
];

const predictions = await predict(model, personTensorNormalized);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labelsNomes[p.i]} (${(p.prob * 100).toFixed(2)}%)`)
  .join("\n");

console.log(results);
