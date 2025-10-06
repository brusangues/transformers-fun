
# transformers-fun
Treinamento de modelo GPT do zero com PyTorch, derivado do nanoGPT do Andrej Karpathy.  
Os dados utilizados são textos de alguns autores de horror na linguagem portuguesa.  
Referência principal: Andrej Karpathy's nanogpt-lecture  
[Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html)  
[nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)

# TODO
* Logar estatísticas sobre os pesos para debugar o treinamento das redes
* Redirect stdout para salvar logs de maneira mais fácil
* Adicionar mais autores
* salvar optimizador

# License

MIT

# Referências de parâmetros
## Attention is all you need
### Dataset
English-German  
4,5 milhões de pares de sentenças  
BPE  
37 mil tokens  

English-French  
36 milhões de sentenças  
word piece  
32 mil tokens  

batches de 25 mil de tokens  

### Hardware
8x P100 GPUs  
Cada passo de otimização levou 0.4 s.  
100 mil passos em 12h.  

### Parâmetros
Regular
- context_len: 512
- n_embd: 512
- n_feed_forward: 2048
- n_head: 8
- n_layer: 6
- dropout: 0.1
- 100 mil steps
- 65 milhões de parâmetros no paper  
- 23 milhões na minha arquitetura  

Big
- context_len: 1024
- n_embd: 1024
- n_feed_forward: 4096
- n_head: 16
- n_layer: 6
- dropout: 0.3
- 300 mil steps
- 213 milhões de parâmetros

### Optimizer
β1 = 0.9, β2 = 0.98 and ϵ = 10−9  
lrate = d
−0.5
model · min(step_num−0.5
, step_num · warmup_steps−1.5
)
warmup_steps = 4000