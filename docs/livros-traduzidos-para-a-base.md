# Livros traduzidos para aumentar a base

> Lista de obras **estrangeiras com tradução em português** que seriam boas para o
> corpus de terror/ficção — com a régua de direitos autorais aplicada (a mesma de
> sempre: **nada pirata**, LibGen/LeLivros/eLivros = fora). Tudo abaixo é de acesso
> **legítimo** (domínio público ou upload já verificado).
> Marcadores: `✓` = verificado nesta sessão | `?` = confirmar antes de baixar.

---

## A régua legal (por que a lista é do jeito que é)

Traduções têm **copyright próprio de 70 anos após a morte do tradutor**. Ou seja:

- **Traduções modernas (1956 em diante)** de Poe/Stoker/fantasia → **protegidas**.
  Mesmo que um site hospede (LibGen, Le Livros/eLivros, Z-Library, Internet Archive
  com uploads piratas) → **fora**. Não são opção legal.
- **Traduções do século XIX / início do XX (pré-1943/pré-1956)** → **domínio público**,
  legais. O custo: **ortografia arcaica** (pré-reforma de 1943) — o notebook já
  diagnostica (`arch_per_mil`) e a base já recebe autores com esse perfil (Alencar,
  Macedo, artur_azevedo, voltaire na v25).

**Conclusão prática:** traduções estrangeiras legais em pt moderno ≈ **não existem**.
As candidat = os clássicos PD antigos abaixo.

---

## Já na base (v25) — só para referência

| Obra/Autor | Tradução | Fonte | Observação |
|---|---|---|---|
| **Poe — Todos os Contos**, vol 1+2, etc. | modernas (Navras/Aleph) | `data/poe/` (locais) | 5 obras, ~2,8M chars — já entra |
| **Chambers — O Rei de Amarelo** | moderna (Clock Tower) | `data/chambers/` | 1 obra — já entra |
| **Voltaire — Cândido, Zadig, Micrômegas, O Ingênuo + 7** | antiga PD | DP BR (`dpbr_voltaire`) | 11 obras, ~1,15M chars — v25 |

> Voltaire é o único bloco de traduções estrangeiras PD já dentro da base (ficção
> filosófica completa em pt, arcaica mas ótima pro modelo).

---

## Candidatos novos RECOMENDADOS (traduções PD, legais)

### Grupo A — Ficção clássica com volume (boa relação custo/benefício)

| Obra | Autor | Tradução/época | Fonte de acesso | Prioridade |
|---|---|---|---|---|
| **Da Terra à Lua** ✓ | Júlio Verne | séc. XIX, PD | Gutenberg pt (id 28341) | ★★★ (prosa narrativa, ~80k chars) |
| **Robur, o Conquistador** ✓ | Júlio Verne | séc. XIX, PD | Gutenberg pt (id 62101) | ★★★ |
| **Os Trabalhadores do Mar** ✓ | Victor Hugo | séc. XIX, PD | Gutenberg pt (id 57895) | ★★★ (romance longo) |
| **Hamlet** ✓ | Shakespeare | séc. XIX, PD | Gutenberg pt (id 25667) | ★★ (teatro — variação de língua) |
| **Otelo** ✓ | Shakespeare | séc. XIX, PD | Gutenberg pt (id 28526) | ★★ |
| **Perolas e Diamantes** (contos) ✓ | Grimm | séc. XIX, PD | Gutenberg pt (id 30510) | ★★ (contos curtos) |
| **Um club da Má-Lingua** ✓ | Dostoiévski | séc. XIX, PD | Gutenberg pt (id 31657) | ★★ (conto; raro Dostoiévski pt PD) |

### Grupo B — Ficção filosófica / outras (poucas obras, mas legais; via DP BR)

| Obra | Autor | Qtd DP BR | Fonte | Prioridade |
|---|---|---|---|---|
| **Cervantes** (2 obras) ✓ | — | 2 | DP BR (`co_idioma=1`) | ★★ (verificar se é pt) |
| **Wilde** (2 obras) ✓ | — | 2 | DP BR | ★★ |
| **Verne** (2 obras) ✓ | — | 2 | DP BR | ★★ |
| **Dumas / Balzac / Shakespeare** ✓ | — | 2 cada | DP BR | ★★ |
| **Swift / Goethe / Kafka / Dante** ✓ | — | 1 cada | DP BR | ★ (Kafka = raro; confirmar) |

> No DP BR, buscar por AUTOR com `co_idioma=1` (Português) + `co_categoria=2`
> (Literatura) — a busca com acento importa. Onda de download: o builder já faz
> (busca → co_obra → fetch no driver → pdfplumber → `data/dpbr_<autor>/`).

### Grupo C — Wikisource pt (traduções soltas, legítimas)

| Obra | Autor | Observação |
|---|---|---|
| **O Corvo** (trad. Machado de Assis) | Poe | a mais famosa; curta (poesia) |
| Categoria "Traduções" do WS pt | — | `?` — probe ficou limitada pelo rate-limit; re-checar |

---

## O que NÃO dá para fazer (e por quê)

- **Bram Stoker — Drácula**: só existe em EN no DP BR (copy do Project Gutenberg) e em
  traduções BR modernas (copyright). **Sem versão pt PD acessível** — não entra.
- **Horror moderno traduzido (Stephen King pt, etc.)**: copyright óbvio. Não.
- **Qualquer upl pt "grátis" de tradução moderna** na Internet Archive / Le Livros:
  pirata, mesmo que o site deixe baixar. **Não.**
- **Poe "Histórias Extraordinárias" (tradução moderna)**: a que o DP BR/"gratuitas"
  oferecem em ES para o Poe, ou modernas protegidas — fora.

---

## Recomendação de execução (próximos passos)

1. **Grupo A** (Gutenberg pt — requests puro, sem Cloudflare): novo crawler leve
   `crawler/gutenberg.py` seguindo o checklist de `docs/crawlers-como-funcionam.md`
   (camada 0, parse de texto, salvar `data/gut_<autor>/<slug>.txt`). ~6-7 obras,
   algumas dezenas de MB de chars.
2. **Grupo B** (DP BR): rodar o builder existente com os autores → `data/dpbr_<autor>/`
   → conferir idioma/variante antes de entrar (o notebook já filtra EN e diagnostica
   arcaico).
3. Depois: **playground_5** de novo (autodetecta a última base) → **nova versão vXX**
   com a tabela de balanceamento.
4. Prévio/sanity: confirmar que cada obra retornou em **pt** (não ES/EN) e anotar o
   `arch_per_mil` (o Sherlock/Wilde em pt pode ser PT-PT europeu — decidir se entra).

> Regra de ouro mantida: fonte de acesso legítima, proveniência citada no commit e
> ortografia arcaica aceita (com diagnóstico) — nada escondido, nada pirata.
