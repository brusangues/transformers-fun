
# transformers-fun
Treinamento de modelo GPT do zero com PyTorch, derivado do nanoGPT do Andrej Karpathy.  
Os dados utilizados são textos de alguns autores de horror na linguagem portuguesa.  
Referência principal: Andrej Karpathy's nanogpt-lecture  
[Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html)  
[nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py)

# TODO
* ~~Redirect stdout para salvar logs de maneira mais fácil~~
* ~~Salvar optimizador~~
* ~~Aumentar tokenizer de 10k para 50k~~
* ~~Acúmulo de gradiente~~
* ~~Attention is all you need learning rate~~
* Adicionar mais autores no dataset
* Treinar com as duas T4 no kaggle
* Early stopping
* Colocar no chatbot do telegram
* Classificação de autor ou tipo de texto
* Logar estatísticas sobre os pesos para debugar o treinamento das redes

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
We used the Adam optimizer with β1 = 0.9, β2 = 0.98 and ϵ = 1e−9.
We varied the learning rate over the course of training, according to the formula:

l_rate = d_model^(−0.5) · min(step_num^(−0.5), step_num · warmup_steps^(−1.5))

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_steps = 4000.