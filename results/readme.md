# 📊 Análise de Resultados: Agrupamento Não Supervisionado

Esta pasta contém as visualizações geradas a partir dos vetores de características extraídos pela rede **VGG-16 pré-treinada na ImageNet**. O objetivo desta etapa experimental foi validar se uma Inteligência Artificial genérica conseguiria, por conta própria, diferenciar a morfologia de galáxias (Elípticas vs Espirais) baseando-se apenas em geometria e padrões visuais básicos, sem ter sido treinada especificamente com dados astronômicos.


## 1. Amostra Controlada: 28 Galáxias

Realizamos testes iniciais com um subconjunto pequeno.

### 📄 `pca_28_original.png`
Neste gráfico, utilizamos a técnica PCA para reduzir a dimensionalidade mantendo as três classes originais: Elípticas (E), Espirais (S) e Espirais Barradas (SB). Observa-se que as galáxias Elípticas tendem a ocupar uma região específica do espaço vetorial, enquanto as Espirais e Espirais Barradas aparecem misturadas. Isso sugere que a presença da "barra" central é um detalhe visual muito sutil para que a VGG-16 original consiga utilizar como critério forte de separação sem um treinamento prévio.

### 📄 `pca_28_espirais_juntas.png`
Aqui, simplificamos o problema unificando as classes S e SB em um único grupo de "Espirais", transformando a tarefa em uma classificação binária (Elípticas vs. Espirais). O resultado visual demonstra uma separação muito mais clara e definida. Isso indica que, em uma escala reduzida, a rede neural consegue distinguir com sucesso a diferença fundamental entre objetos difusos e arredondados (Elípticas) e objetos com estrutura de disco (Espirais).

### 📄 `pca_28_kmeans3.png`
Neste experimento, forçamos o algoritmo K-Means a encontrar automaticamente três grupos distintos nos dados. A expectativa era que ele redescobrisse as classes E, S e SB. No entanto, o algoritmo falhou em separar as Espirais Barradas das Normais, criando agrupamentos baseados em outras características. 


## 2. Dataset maior: ~10.000 Galáxias

Ao expandir o experimento para um volume maior de dados, a complexidade do problema se tornou evidente, revelando as limitações da abordagem não supervisionada.

### 📄 `pca_10k.png`
A visualização via PCA de todo o conjunto de dados resultou em uma grande "nuvem" com sobreposição significativa entre as classes. Como o PCA é uma técnica linear, ele não foi capaz de "desembolar" a complexidade dos dados astronômicos. A VGG-16 parece agrupar as imagens baseando-se em características genéricas de baixo nível, o que resulta em uma mistura onde não há fronteiras claras de decisão entre os tipos morfológicos.

### 📄 `tsne.png`
Utilizando o t-SNE, o resultado ainda apresentou uma mistura considerável das classes. Este é um resultado negativo muito importante: ele prova que os *embeddings* gerados por uma rede treinada em imagens do cotidiano (ImageNet) não são robustos o suficiente para a classificação astronômica direta. As características visuais que a rede prioriza não são as mesmas necessárias para distinguir morfologias galácticas complexas.


## 📝 Conclusão do Experimento

Os resultados visuais obtidos nesta etapa justificam a necessidade  de avançar para a etapa de **Treinamento Supervisionado** ou a utilização de modelos especializados (como o Zoobot). Ficou demonstrado que, sem ajustar os pesos da rede neural para o domínio específico da astronomia, a separação automática das classes baseada apenas em similaridade visual genérica é ineficaz para grandes volumes de dados.