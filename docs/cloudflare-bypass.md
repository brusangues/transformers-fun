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

## Aplicação ao Domínio Público BR

Alvo: `http://www.dominiopublico.gov.br/pesquisa/PesquisaObraForm.do`
(HTTP, não HTTPS — o site do governo). Bloqueio atual: Cloudflare "Just a moment..."
até para Chromium headless.

Roteiro com `selenium_get` adaptado:

1. `driver.get('http://www.dominiopublico.gov.br/pesquisa/PesquisaObraForm.do')`
2. Aguardar o `_cf_chl_opt` sumir do source (mesmo polling do `selenium_get`).
3. Preencher o form de busca via Selenium (campos: título/autor/idioma) e
   `submit()` — o site faz POST normal (não-JS) depois do challenge.
4. Na página de resultados, extrair os links de download — padrão conhecido:
   `http://www.dominiopublico.gov.br/download/texto/<id>.pdf` — via
   `driver.find_elements(By.CSS_SELECTOR, 'a[href*="/download/texto/"]')`.
5. Baixar os PDFs com requests COMUM (o download direto não passa pelo
   challenge, só o formulário de busca) — ou via `driver` se bloquear.
6. `data/<autor>/` + playground_3 (pipeline existente) → próxima base vXX.

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
