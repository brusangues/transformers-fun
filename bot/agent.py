from .models_api import query_model
from .models_local import generate_sample
import json
import re

PROMPT = """
Você é um agente de perguntas e respostas com a seguinte ferramenta:

@modelo_local
Nome: modelo_local
Descrição:
Ferramenta de chamada de modelo local de geração de texto.
O modelo local é um gpt-2 pré-treinado sem fine-tuning.
Argumentos:
text - texto a ser completado pelo modelo local
n_tokens - número de tokens a serem gerados pelo modelo local (padrão: 100)

Se o usuário mencionar o @modelo_local, formate a resposta em um json contendo os argumentos para a chamada do mesmo, seguindo o exemplo abaixo:
```json
{{
    "modelo_local": {{
        "text": "Era uma vez...",
        "n_tokens": 50
    }}
}}
```
Do contrário, responda o usuário normalmente.

Pergunta do usuário:
{text}
"""


def answer_with_local_model(model_api, model_local, text):
    prompt_api = PROMPT.format(text=text)
    answer_api = query_model(model_api, prompt_api)[0]

    # Parse answer_api as json if possible
    text_local = None
    try:
        answer_clean = re.sub(r"```json(.*?)```", r"\1", answer_api, flags=re.DOTALL).strip()
        answer_json = json.loads(answer_clean)
        text_local = answer_json.get("modelo_local", {}).get("text", None)
        n_tokens = answer_json.get("modelo_local", {}).get("n_tokens", 100)
    except Exception as e:
        print("Failed to parse answer_api as json", e)
    
    if text_local is not None:
        answer_local = generate_sample(model_local[0], model_local[1], text_local, n_tokens)
        answer_local_formatted = "Resposta do modelo local:\n" + answer_local
        return answer_local_formatted
    else:
        return answer_api
