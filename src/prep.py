import re
from unidecode import unidecode

chars_remove = ['^', '`', '~']

# Limpando algumas repetições de espaços, quebras de linha e tabulações
def clean_text(s):
    s = re.sub("[\n\t\xa0]+", "\n", s)

    s = unidecode(s)
    # print(f"{s=}")
    s = re.sub("[&@_~`^/\\\\|]", " ", s)
    # print(f"{s=}")
    s = re.sub("\s{4,}", "\n", s)
    # print(f"{s=}")
    # s = re.sub("\t+", " ", s)
    # print(f"{s=}")
    s = re.sub(" +", " ", s)
    # print(f"{s=}")
    s = s.strip()

    return s
