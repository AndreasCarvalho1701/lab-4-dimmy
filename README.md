# Laboratório Técnico 04: O Transformer Completo "From Scratch"

**Aluno:** Andreas Carvalho  
**Disciplina:** Tópicos em Inteligência Artificial  
**Professor:** Dimmy Magalhães  

---

## Descrição

Este projeto implementa, em um único arquivo Python, a arquitetura completa de um **Transformer Encoder-Decoder**, reunindo os principais blocos estudados nos laboratórios anteriores.

A implementação contempla:

- **Scaled Dot-Product Attention**
- **Masked Self-Attention**
- **Cross-Attention**
- **Add & Norm**
- **Feed-Forward Network (FFN)**
- **Positional Encoding**
- **Empilhamento de camadas no Encoder e no Decoder**
- **Inferência auto-regressiva com laço `while`**

O objetivo do laboratório foi montar o fluxo completo do Transformer e realizar uma demonstração fim-a-fim com uma **toy sequence**, usando como entrada a frase **"Thinking Machines"**.

---

## Objetivos do projeto

- Integrar os blocos fundamentais do Transformer em uma arquitetura única
- Garantir o fluxo correto de tensores entre Encoder e Decoder
- Aplicar máscara causal no Decoder para impedir acesso a tokens futuros
- Utilizar Cross-Attention para conectar a saída do Encoder ao Decoder
- Implementar a geração de saída de forma auto-regressiva, iniciando com `<START>` e encerrando em `<EOS>`

---

## Estrutura implementada

### Encoder
Cada bloco do Encoder segue o fluxo:

1. Self-Attention  
2. Add & Norm  
3. Feed-Forward Network  
4. Add & Norm  

### Decoder
Cada bloco do Decoder segue o fluxo:

1. Masked Self-Attention  
2. Add & Norm  
3. Cross-Attention  
4. Add & Norm  
5. Feed-Forward Network  
6. Add & Norm  

### Inferência
A inferência foi implementada com um laço `while`, em que:

- o Decoder começa com o token `<START>`
- a cada iteração, o modelo prevê o próximo token com base nas probabilidades do Softmax
- o token previsto é concatenado à entrada do Decoder
- o processo continua até a geração do token `<EOS>`

---

## Tecnologias utilizadas

- Python
- PyTorch

---

## Arquivos do projeto

- `transformer.py` — implementação completa do Transformer
- `requirements.txt` — dependências necessárias
- `README.md` — documentação do projeto

---

## Execução

Para instalar as dependências:

`pip install -r requirements.txt`

Para executar o projeto:

`python transformer.py`

---

## Referências teóricas

A implementação foi baseada nos conceitos estudados em aula, nos laboratórios anteriores da disciplina e em referências clássicas sobre a arquitetura Transformer, especialmente:

- Vaswani, A. et al. *Attention Is All You Need*. NeurIPS, 2017.

Essas referências foram utilizadas para compreender a lógica matemática da atenção, do Encoder, do Decoder, da máscara causal e da geração auto-regressiva.

---

## Uso de Inteligência Artificial

**Partes geradas/complementadas com IA, revisadas por Andreas Carvalho.**

Durante o desenvolvimento deste projeto, utilizei ferramentas de IA, especialmente o ChatGPT, como apoio complementar em diferentes etapas do trabalho. O uso ocorreu principalmente para:

- esclarecimento de dúvidas conceituais sobre a arquitetura Transformer
- apoio no raciocínio lógico da implementação
- auxílio na organização da estrutura do código
- revisão de partes da lógica matemática do Encoder e do Decoder
- ajuda na compreensão e aplicação de funções e componentes do PyTorch
- suporte na construção e ajuste de trechos envolvendo atenção, máscaras, inferência auto-regressiva e manipulação de tensores

Além disso, também utilizei referências teóricas e artigos da área para compreender melhor os fundamentos da arquitetura Transformer e garantir maior consistência na implementação.

O uso de IA não substituiu o estudo do conteúdo. Toda a lógica, estrutura e implementação final foram revisadas, compreendidas e ajustadas antes da submissão.