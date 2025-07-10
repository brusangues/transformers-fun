import re
from unidecode import unidecode

# Lendo e olhando para os caracteres do texto de Shakespeare
input_file = "data/shakespeare.txt"
print(f"{input_file=}")
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
print(f"{len(text)=} {len(chars)=}\n{chars=}")

# Obra de lovecraft em português
input_file = "data/lovecraft_ptbr.txt"
print(f"{input_file=}")
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
print(f"{len(text)=} {len(chars)=}\n{chars=}")

# Limpando trechos de início e fim do livro
# id_start = re.search("DAGON", text).span(0)[0]
# print(text[id_start : id_start + 1000])
# id_end = re.search("FONTES DOS TEXTOS", text).span(0)[0] - 1
# print(text[-1000 + id_end : id_end])
# text_cropped = text[id_start:id_end]


# Limpando algumas repetições de espaços, quebras de linha e tabulações
def clean_text(s):
    s = unidecode(s)
    s = re.sub("[&/]", " ", s)
    s = re.sub("\n+", "\n", s)
    s = re.sub("\t+", " ", s)
    s = re.sub(" +", " ", s)
    return s


text_clean = clean_text(text)
chars = sorted(list(set(text_clean)))
print(f"{len(text_clean)=} {len(chars)=}\n{chars=}")
print(f"{text_clean[:1000]=}")

# Salvando o texto limpo
output_file = "data/lovecraft_ptbr_clean.txt"
with open(output_file, "w") as f:
    f.write(text_clean)

print(f"\n\nTexto limpo salvo em {output_file}")
