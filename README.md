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

## Execução

Para instalar as dependências:

```bash
pip install -r requirements.txt

python transformer.py