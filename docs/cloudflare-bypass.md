# Bypass de Cloudflare — método do pokescan-tcg (para Domínio Público BR)

> Documentação do mecanismo de acesso a sites protegidos por Cloudflare, extraído do
> repo `pokescan-tcg` (`crawler/scrapers.py`, branch hermes). Finalidade aqui:
> acessar programaticamente o **Domínio Público BR** (`dominiopublico.gov.br`), que é
> o acervo oficial do governo brasileiro com obras de domínio público — incluindo
> **traduções BR de autores estrangeiros** (Poe, Stoker, Doyle, Andersen…) — mas que
> está atrás do challenge "Just a moment..." do Cloudflare para acessos automatizados.

## Por que o bloqueio existe e por que o bypass funciona

O Cloudflare serve um challenge JS (`_cf_chl_opt` no HTML) para clientes que ele
suspeita não serem navegadores reais. Detecção clássica:

- **requests/urllib** → TLS fingerprint ≠ Chrome → challenge direto (HTTP 403).
- **Playwright/Chromium puro** → o CDP expõe flags (`navigator.webdriver=true`,
  headless detectável, ordem de injeção diferente) → challenge também.

O **`undetected_chromedriver`** patcheia o chromedriver para remover os marcadores
de automação e injeta os scripts na ordem que um navegador real carrega → o
Cloudflare não distingue do Chrome de um humano. Funciona com **janela visível**
(`headless=False`); headless é uma das primeiras coisas que o Cloudflare detecta.

## O mecanismo (fonte: pokescan-tcg `crawler/scrapers.py`)

Duas camadas, em ordem de sofisticação:

### Camada 1 — `cloudscraper` (HTTP puro, para proteções leves)

```python
import cloudscraper

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True},
    delay=10,          # pausa entre retries internos do challenge
    ssl_context=None,  # None = usa o TLS nativo do cloudscraper (impersonation)
)
resp = scraper.get(url)
# checar resp.status_code == 200; 403 = Cloudflare ainda bloqueando
```

`cloudscraper` imita o TLS fingerprint do Chrome no nível HTTP e resolve
challenges JS dos tipos mais simples (v1/v2 com cookieteste). É o caminho leve:
não abre navegador. No pokescan-tcg ficou só no `crawl_0.py` (legado) — para o
Liga Pokémon não basta.

### Camada 2 — `undetected_chromedriver` (navegador real; O caminho de produção)

```python
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

DRIVER = None

def get_driver():
    global DRIVER
    if DRIVER is None:
        DRIVER = uc.Chrome(
            headless=False,        # OBRIGATÓRIO: janela visível (headless é detectado)
            use_subprocess=True,   # chromedriver em subprocesso próprio
            version_main=150,      # pin da major version do Chrome instalado
        )
    return DRIVER

def selenium_get(url, retries=3):
    driver = get_driver()
    for attempt in range(retries):
        try:
            driver.get(url)
            # espera até 30s: sai quando o page_source contém o marcador de SUCESSO
            for _ in range(30):
                time.sleep(1)
                if 'cardsjson' in driver.page_source or 'card' in driver.page_source.lower():
                    break
            page_source = driver.page_source
            # Verifica: NÃO tem _cf_chl_opt (challenge) E tem marcador de conteúdo real
            if '_cf_chl_opt' not in page_source and 'cardsjson' in page_source:
                return page_source
            # ainda bloqueado → retry com backoff
            if attempt < retries - 1:
                time.sleep(5 + attempt * 3)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(3)
    raise Exception("Failed to load page after all retries")
```

Detalhes que importam:

| Parâmetro | Valor | Por quê |
|---|---|---|
| `headless` | `False` | headless é detectável; janela real passa |
| `use_subprocess` | `True` | isola o chromedriver |
| `version_main` | versão do Chrome local | mismatch chromedriver×Chrome quebra ou denuncia automação |
| Marcador de bloqueio | `_cf_chl_opt` no HTML | é a variável do script de challenge do Cloudflare |
| Marcador de sucesso | conteúdo real do site (`cardsjson`/`p1b`/`nPT` na Liga) | nunca confiar só em status 200 — o challenge responde 200 |
| Espera | poll de 1s até 30s | o challenge resolve sozinho em 3–10s numa sessão real |

No pokescan-tcg, `selenium_get` retorna um `Response` fake (`{'text': page_source,
'status_code': 200}`) para os callers ficarem agnósticos — padrão recomendado.

Uso real no repo: `crawler_liga.py`, `crawler_liga_bulk.py`,
`crawler_liga_snapshot.py`, `crawler_liga_hits.py`, `crawler_caixas.py`,
`baixar_sets_faltantes.py` — todos contra `ligapokemon.com.br`, que bloqueia o
Playwright com Cloudflare ("Just a moment...") mas passa com `get_driver()`.
Playwright ficou proibido lá para validar links (regra registrada na skill).

## Aplicação ao Domínio Público BR — fluxo VALIDADO (12/08/2026)

Alvo: `https://www.dominiopublico.gov.br/pesquisa/PesquisaObraForm.do`
(redireciona de http → https). Teste real com `undetected_chromedriver`
(chrome 150, `version_main=150`) — challenge limpa em **4–6s**.

Mecânica exata do site (medida no teste):

1. **Cloudflare challenge**: some o `_cf_chl_opt` do source em ~4–6s com o
   undetected_chromedriver (janela visível). Playwright/requests = bloqueados.
2. **Turnstile INVISÍVEL**: o submit do form é gated por
   `window.__cfRLUnblockHandlers` (o botão tem
   `onclick="if (!window.__cfRLUnblockHandlers) return false; return validar();"`).
   O Turnstile completa sozinho em 1–2s em navegador real — **esperar o handler
   existir antes de submeter** (poll via `execute_script`).
3. **Form** (nome `PesquisaObraActionForm`, `document.forms[1]`):
   - `no_autor` (text) / `ds_titulo` (text) — busca
   - **`co_midia` é OBRIGATÓRIO** (validação `validar()`: "Preencha o(s) Campo(s)
     Obrigatório(s)") — `2` = Texto, `3` = Som, `5` = Imagem, `6` = Video
   - `co_idioma`: `1` = Português, `2` = Inglês, `7` = Espanhol…
   - **Acentos importam na busca**: `Dracula` → 0 resultados, `Drácula` → 3
   - Botão `select_action` (class `myhidden` — oculto) → **click via JS**
     (`document.querySelector('input[name=select_action]').click()`), nunca
     `resetfull_action` (Reset)
   - Setar selects via JS direto (`co_midia.value='2'`) NÃO dispara o `onchange`
     (que clicaria `refresh_action` e limparia o form)
4. **Resultados**: página `ResultadoPesquisaObraForm.do`, 50 obras/página,
   paginação `?first=50&skip=N`. Cada obra: `DetalheObraForm.do?co_obra=<id>`.
5. **Download**: `DetalheObraDownload.do?select_action=&co_obra=<id>&co_midia=2`
   redireciona para o arquivo real — padrão **`/download/texto/<id>.pdf`**
   (ex: `gu000345.pdf`). O requests/urllib simples leva **403** (TLS fingerprint);
   o caminho que funciona: **`fetch()` dentro do driver** (mesma sessão/TLS do
   Chrome) → base64 → salvar:

```python
b64 = driver.execute_async_script("""
    var url = arguments[0], done = arguments[1];
    fetch(url).then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.arrayBuffer();
    }).then(function(buf) {
        var bytes = new Uint8Array(buf), binary = '';
        for (var i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        done(btoa(binary));
    }).catch(function(e) { done('ERR:' + e.message); });
""", pdf_url)
```

6. **Conteúdo real (medido)**: autor `Stoker` → 7 obras (Dracula ×2, Dracula's
   Guest, The Lady of Shroud, The Lair of the White Worm, The Man) — mas o
   "Dracula" baixado é o **Project Gutenberg em INGLÊS** (header do Gutenberg no
   PDF). Autor `Edgar Allan Poe` → 50+ obras na página 1 (EN/ES; títulos como
   "A Descent into the Maesltrom", "Annabel Lee", "El Cuervo", "El Corazón
   Delator"). **Traduções pt-BR de terror estrangeiro são escassas no acervo** —
   o DP BR tem originais EN/ES + autores BR; buscar com `co_idioma=1` + título
   acentuado e filtrar o diagnóstico (o notebook detecta EN/PT-PT/arcaico).
   PDFs do Gutenberg embutidos no DP BR são conteúdo público — ok, mas em EN
   não entram na base pt (o pipeline já filtra por variante).

Scripts de referência (teste validado): o fluxo completo está nos arquivos
`dpbr_*.py` usados na validação (busca → resultados → fetch → pdfplumber →
diagnóstico `arch_per_mil` + variante). Depois do teste: PDF em `data/<autor>/`
+ `playground_3.ipynb` → base vXX.

Dependências (no env `transformers-fun`):

```bash
conda install -c conda-forge undetected-chromedriver selenium cloudscraper
# ou: pip install undetected-chromedriver selenium cloudscraper
```

Requisito de sistema: Chrome/Chromium instalado; `version_main` deve bater com a
major version do Chrome local (`chrome://version`). No pokescan-tcg está fixado
em 150 (Chrome 150 instalado na máquina).

## Regras de uso (limite de conduta do projeto)

- O bypass serve para acessar conteúdo **legítimo**: o Domínio Público BR é o
  acervo oficial do governo brasileiro, com obras 100% de domínio público. Usar
  apenas para isso.
- **Nunca** usar o mecanismo (nem qualquer outro) para baixar material
  protegido por copyright — o mesmo vale para uploads "piratas" (LibGen,
  Z-Library, Le Livros/eLivros no Internet Archive): traduções BR modernas são
  protegidas por 70 anos pós-morte do tradutor e NÃO entram na base, mesmo que
  um site as hospede.
- Manter o ritmo humano (janela visível, polling de 1s, retries com backoff) —
  o objetivo é passar o challenge, não abusar do servidor.
