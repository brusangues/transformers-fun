# Crawlers — como funcionam (padrões reutilizáveis)

> Referência viva de como os crawlers foram construídos (**origem: `pokescan-tcg/crawler/`**),
> para reutilizar as técnicas em novos sites. No transformers-fun a pasta `crawler/`
> está vazia de propósito — é onde os novos crawlers de corpus vão entrar.

## Ideia central: fetch em CAMADAS + parse por regex

Todo crawler do projeto segue o mesmo esqueleto — **1) buscar** o HTML/JSON,
**2) extrair** com regex, **3) salvar** com cache idempotente. O que muda entre
sites é só a camada de fetch (quanto o site protege) e o parse.

```
buscar (camada certa)  →  extrair (regex/JSON)  →  salvar (idempotente + consolidate)
```

---

## Camadas de fetch — o coração (`scrapers.py`)

Escolhe-se a camada **mais leve que funciona** para o site. Tentativa em ordem:

| Camada | Técnica | Quando usar | Código |
|---|---|---|---|
| **0. requests** | `requests.Session` + User-Agent de Chrome | site SEM proteção | `crawler_pull_rates.py` (ThePriceDex) |
| **1. cloudscraper** | imita TLS fingerprint do Chrome no HTTP | Cloudflare leve / sem navegador | `scrapers.cloudscraper_get` |
| **2. undetected_chromedriver** | Chrome real via Selenium patcheado | Cloudflare "Just a moment..." / Turnstile | `scrapers.selenium_get` |

### Camada 2 — `scrapers.selenium_get` (a mais usada na Liga)

```python
# scrapers.py (pokescan-tcg/crawler)
import undetected_chromedriver as uc
DRIVER = None
def get_driver():
    global DRIVER
    if DRIVER is None:
        DRIVER = uc.Chrome(headless=False, use_subprocess=True, version_main=150)
    return DRIVER

def selenium_get(url, retries=3):
    driver = get_driver()
    for attempt in range(retries):
        driver.get(url)
        for _ in range(30):            # espera até 30s p/ o Cloudflare resolver
            time.sleep(1)
            if 'cardsjson' in driver.page_source or 'card' in driver.page_source.lower():
                break
        src = driver.page_source
        # BLOQUEADO = tem _cf_chl_opt (script do challenge); SUCESSO = conteúdo real
        if '_cf_chl_opt' not in src and ('cardsjson' in src or 'p1b' in src or 'nPT' in src):
            return type('Response', (), {'text': src, 'status_code': 200})()
        # ainda bloqueado → retry com backoff
        time.sleep(5 + attempt * 3)
    raise Exception("Failed to load page after all retries")
```

Pontos que SEMPRE importam:
- **`headless=False`** — janela visível; headless é detectado pelo Cloudflare.
- **`version_main=150`** — deve bater com a major do Chrome instalado.
- **Marcador de bloqueio `_cf_chl_opt`** no HTML = página ainda no challenge (o status 200 engana).
- **Marcador de sucesso = conteúdo real do site** (cardsjson/p1b/nPT na Liga) — nunca confiar no status.

### Multi-camadas num arquivo (`crawler_liga.py`)

```python
def crawl_all_modes():
    for name, fn in [('cloudscraper', crawl_via_cloudscraper),
                     ('selenium', crawl_via_selenium),
                     ('playwright'?, crawl_via_playwright)]:
        cards = fn()
        if cards: return cards
    return None  # fallback manual: imprime URL p/ abrir no navegador
```

`crawl_via_cloudscraper` → se falhar, `crawl_via_selenium` → se falhar, modo manual.
(Playwright tende a NÃO passar — CDP detectável; ficou praticamente só como fallback.)

---

## Parse — 2 padrões predominantes

### 1. JSON embutido no HTML (`var cardsjson = [...]`)
Site renderiza os dados num `var cardsjson`, e o crawler extrai com regex + `json.loads`:

```python
match = re.search(r'var cardsjson = (\[.*?\]);', html, re.DOTALL)
cards = json.loads(match.group(1)) if match else None
```
Padrão da **Liga** — usado em `crawler_liga`, `crawler_liga_bulk`, `crawler_liga_hits`,
`crawler_liga_snapshot`, `baixar_sets_faltantes`. Se o site expõe um estado JS/JSON
embutido, prefira SEMPRE isso ao invés de parsear HTML tag a tag.

### 2. Tabelas HTML por células (regex)
Crawler define um padrão de célula e varre sequências:

```python
CELL = r'<t[dh][^>]*>(?:<style>.*?</style>)?(?:<p[^>]*>)?([^<]*?)(?:</p>)?</t[dh]>'
cells = [c.strip() for c in re.findall(CELL, html)]
# depois percorre 'cells' procurando sequências significativas (ex: [Raridade, '1 in X packs', ...])
```
Padrão do **ThePriceDex** (`crawler_pull_rates.py`) — tabelas longas viram uma lista
plana de células e o parser busca *assinaturas* (ex: `^1 in [\d,.]+ packs$`, `^$.+`,
`^\d+ cards$`). Mais robusto que `findall` de `<tr>` quando a estrutura varia.

### Extra
- `crawl_0.py`: usa `cloudscraper_get` + `selenium_get` lado a lado (legado de testes).
- Nomes/links de páginas com `<a href=...co_obra=N>` — regex de âncora com o id.

---

## Cache & idempotência — a regra de ouro

**Nunca re-baixar o que já está em disco.** Todos os crawlers pulam o que existe:

```python
path = LIGA_DIR / f'set_{eid}.json'
if path.exists():
    cards = json.load(open(path))
    continue  # já temos — segue
```

- `crawler_liga_bulk`: grava `data/liga/set_<eid>.json` por set; re-run só baixa os faltantes.
- `crawler_liga_hits`: cache duplo — em memória (mesma execução) + em disco (JSON por dia).
- Wikisource (`ws_downloader*.py`): cache idempotente em `data/ws_<autor>/*.txt`.
- Isso permite **retomada**: se o processo morre, re-rodar continua de onde parou.

Corolário de salvamento: consolidar os N arquivos num CSV/JSON único no fim
(`consolidate()` em `crawler_liga_bulk`, com `data/liga/liga_all_cards.csv`).

---

## Rate-limit & backoff

- **`time.sleep(2)`** entre requests (Liga bulk) / **`time.sleep(1)`** (ThePriceDex).
- **Retries com backoff** no `selenium_get` (`5 + attempt*3`).
- **Honrar `Retry-After`** do servidor quando houver (lição do Wikisource: 400/429 com
  header Retry-After 2-19s; backoff cego é contraproducente quando o bucket se recupera).
- **NUNCA paralelizar request heavey contra um único servidor** sem saber os limites —
  o Wikisource deu 429 e backoff até 1024s quando paralelizado com 3 shards
  (lição registrada no `.hermes.md`). `parallel.py` existe mas é para workloads que
  toleram; para corpus, 1 processo + pacing ganha.

---

## Descoberta de IDs (`crawler_liga_bulk.discover_set_ids`)

Quando o site usa IDs numéricos por página, **escaneie faixas ao redor dos IDs já
conhecidos** (não o range inteiro):

```python
ranges = [range(max(1, kid - 5), kid + 6) for kid in known]
to_test = set().union(*ranges) - known
for eid in sorted(to_test):
    url = f'...?edid={eid} ed=POR'
    if cardsjson presente: discovered.add(eid)
    time.sleep(1)
```
Crawler testa, guarda `liga_set_ids.json`, e o próximo run só baixa o que falta.

---

## Paralelismo (`parallel.py`)

Worker pool (`multiprocessing.Pool`) com:
- **timeout** por job + **retry** automático (re-enfileira o job que estourou o tempo)
- **detecção de processo morto** (checa `mp.active_children()` vs os PIDs iniciados)
- **progress bar** (tqdm) + supressão de stdout dos workers (`io.capture_output`)
- variante `sequential_processing` com retry por re-append.

⚠️ Usar com critério — para scraping de um site só, a paralelização pode causar
rate-limit (ver seção anterior). Paralelize quando os alvos forem muitos e o servidor
aguentar (ou requests puros como ThePriceDex).

---

## Mapa dos crawlers existentes (o que reutilizar de cada)

| Arquivo | Site | Camada fetch | Parse | Padrão a reusar |
|---|---|---|---|---|
| `scrapers.py` | — | cloudscraper + undetected | — | **o núcleo de fetch** |
| `crawler_liga.py` | ligapokemon | multi-camada | cardsjson | fallback de camadas + salvar JSON/CSV |
| `crawler_liga_bulk.py` | ligapokemon | selenium_get | cardsjson | descoberta de IDs + idempotência + consolidate |
| `crawler_liga_hits.py` | ligapokemon | selenium_get | cardsjson | cache duplo (memória+disco) |
| `crawler_liga_snapshot.py` | ligapokemon | selenium_get | cardsjson | batch por set |
| `baixar_sets_faltantes.py` | ligapokemon | **get_driver** direto | cardsjson | navegação direta (edid=...) |
| `crawler_caixas.py` | ligapokemon | get_driver | cardsjson | pcode de produtos selados |
| `crawler_pull_rates.py` | thepricedex | **requests puro** | regex células | site SEM Cloudflare — requests+UA |
| `parallel.py` | — | — | — | pool com timeout/retry |
| `crawl_0.py` | ligapokemon | cloudscraper+selenium | cardsjson | legado de testes |

---

## Checklist para raspar um site novo

1. **Teste a proteção** (nesta ordem, no browser/curl):
   - `requests` + UA → 200 com conteúdo? → **camada 0**.
   - `cloudscraper` → consegue? → **camada 1**.
   - senão → **camada 2**: `selenium_get` / `undetected_chromedriver`. Ajuste o
     marcador de sucesso (o que identifica que a página carregou) e o de bloqueio.
2. **Descubra onde estão os dados**: JSON embutido (`var X = [...]` / `<script>`) é o
   melhor; senão regex de células/tabelas; senão anchors com IDs.
3. **Escreva o parse com uma amostra salva** (não fique batendo no site): salve o
   HTML 1× e desenvolva o regex contra o arquivo local.
4. **Aplique idempotência** (skip se arquivo existe) + `consolidate()` no fim.
5. **Pacing**: `time.sleep(1-2)` entre requests; respeite `Retry-After`; não paralelize
   sem testar antes.
6. **Salve com timestamp ou id** e cite o site no nome do arquivo/dir.

---

## Regras de conduta

- Uso do crawler **só para conteúdo legítimo** — o bypass de Cloudflare serve a
  acervos públicos/domínio público (ex.: Domínio Público BR, Wikisource), nunca para
  baixar material protegido por copyright (LibGen, Z-Library, LeLivros/eLivros = fora).
- Manter ritmo humano (janela visível, pacing, retries com backoff) — o objetivo é
  passar a proteção, não abusar do servidor.
- Documentar a fonte/proveniência no commit (não esconder de onde veio).
