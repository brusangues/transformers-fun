import requests
from bs4 import BeautifulSoup
import re
from unidecode import unidecode

links = [
    "fiction/a.aspx"
    ,"fiction/as.aspx"
    ,"fiction/mm.aspx"
    ,"fiction/az.aspx"
    ,"fiction/bec.aspx"
    ,"fiction/bc.aspx"
    ,"fiction/bws.aspx"
    ,"fiction/b.aspx"
    ,"fiction/cc.aspx"
    ,"fiction/cdw.aspx"
    ,"fiction/cu.aspx"
    ,"fiction/c.aspx"
    ,"fiction/cb.aspx"
    ,"fiction/clc.aspx"
    ,"fiction/cs.aspx"
    ,"fiction/ca.aspx"
    ,"fiction/crc.aspx"
    ,"fiction/cy.aspx"
    ,"fiction/d.aspx"
    ,"fiction/ddb.aspx"
    ,"fiction/de.aspx"
    ,"fiction/dat.aspx"
    ,"fiction/di.aspx"
    ,"fiction/ds.aspx"
    ,"fiction/dq.aspx"
    ,"fiction/dwh.aspx"
    ,"fiction/dh.aspx"
    ,"fiction/ee.aspx"
    ,"fiction/ec.aspx"
    ,"fiction/eo.aspx"
    ,"fiction/faj.aspx"
    ,"fiction/f.aspx"
    ,"fiction/fb.aspx"
    ,"fiction/ge.aspx"
    ,"fiction/gm.aspx"
    ,"fiction/hd.aspx"
    ,"fiction/he.aspx"
    ,"fiction/hwr.aspx"
    ,"fiction/hn.aspx"
    ,"fiction/hwb.aspx"
    ,"fiction/hmb.aspx"
    ,"fiction/hrh.aspx"
    ,"fiction/hb.aspx"
    ,"fiction/hm.aspx"
    ,"fiction/h.aspx"
    ,"fiction/hy.aspx"
    ,"fiction/ibid.aspx"
    ,"fiction/iv.aspx"
    ,"fiction/iwe.aspx"
    ,"fiction/lt.aspx"
    ,"fiction/lgb.aspx"
    ,"fiction/ld.aspx"
    ,"fiction/lf.aspx"
    ,"fiction/ms.aspx"
    ,"fiction/mc.aspx"
    ,"fiction/m.aspx"
    ,"fiction/mb.aspx"
    ,"fiction/mo.aspx"
    ,"fiction/mez.aspx"
    ,"fiction/mys.aspx"
    ,"fiction/mg.aspx"
    ,"fiction/nc.aspx"
    ,"fiction/no.aspx"
    ,"fiction/n.aspx"
    ,"fiction/ob.aspx"
    ,"fiction/og.aspx"
    ,"fiction/oa.aspx"
    ,"fiction/o.aspx"
    ,"fiction/pm.aspx"
    ,"fiction/ph.aspx"
    ,"fiction/pg.aspx"
    ,"fiction/p.aspx"
    ,"fiction/qi.aspx"
    ,"fiction/rw.aspx"
    ,"fiction/rdsj.aspx"
    ,"fiction/sc.aspx"
    ,"fiction/sot.aspx"
    ,"fiction/soi.aspx"
    ,"fiction/soidd.aspx"
    ,"fiction/sh.aspx"
    ,"fiction/sk.aspx"
    ,"fiction/sm.aspx"
    ,"fiction/src.aspx"
    ,"fiction/shh.aspx"
    ,"fiction/s.aspx"
    ,"fiction/se.aspx"
    ,"fiction/te.aspx"
    ,"fiction/tom.aspx"
    ,"fiction/tm.aspx"
    ,"fiction/td.aspx"
    ,"fiction/tgsk.aspx"
    ,"fiction/tas.aspx"
    ,"fiction/t.aspx"
    ,"fiction/tjr.aspx"
    ,"fiction/trap.aspx"
    ,"fiction/tr.aspx"
    ,"fiction/th.aspx"
    ,"fiction/tbb.aspx"
    ,"fiction/up.aspx"
    ,"fiction/u.aspx"
    ,"fiction/vof.aspx"
    ,"fiction/wmb.aspx"
    ,"fiction/wid.aspx"
    ,"fiction/ws.aspx"
    ,"fiction/wd.aspx"
    ,"poetry/p061.aspx"
    ,"poetry/p286.aspx"
    ,"poetry/p337.aspx"
    ,"poetry/p122.aspx"
    ,"poetry/p253.aspx"
    ,"poetry/p209.aspx"
    ,"poetry/p470.aspx"
    ,"poetry/p367.aspx"
    ,"poetry/p445.aspx"
    ,"poetry/p473.aspx"
    ,"poetry/p395.aspx"
    ,"poetry/p441.aspx"
    ,"poetry/p432.aspx"
    ,"poetry/p184.aspx"
    ,"poetry/p163.aspx"
    ,"poetry/p336.aspx"
    ,"poetry/p168.aspx"
    ,"poetry/p093.aspx"
    ,"poetry/p265.aspx"
    ,"poetry/p289.aspx"
    ,"poetry/p100.aspx"
    ,"poetry/p267.aspx"
    ,"poetry/p181.aspx"
    ,"poetry/p340.aspx"
    ,"poetry/p131.aspx"
    ,"poetry/p353.aspx"
    ,"poetry/p068.aspx"
    ,"poetry/p287.aspx"
    ,"poetry/p355.aspx"
    ,"poetry/p121.aspx"
    ,"poetry/p190.aspx"
    ,"poetry/p109.aspx"
    ,"poetry/p196.aspx"
    ,"poetry/p051.aspx"
    ,"poetry/p285.aspx"
    ,"poetry/p095.aspx"
    ,"poetry/p104.aspx"
    ,"poetry/p085.aspx"
    ,"poetry/p004.aspx"
    ,"poetry/p245.aspx"
    ,"poetry/p139.aspx"
    ,"poetry/p170.aspx"
    ,"poetry/p078.aspx"
    ,"poetry/p124.aspx"
    ,"poetry/p342.aspx"
    ,"poetry/p187.aspx"
    ,"poetry/p052.aspx"
    ,"poetry/p228.aspx"
    ,"poetry/p281.aspx"
    ,"essays/ar.aspx"
    ,"essays/atr.aspx"
    ,"essays/cd.aspx"
    ,"essays/dp.aspx"
    ,"essays/lc.aspx"
    ,"essays/mr.aspx"
    ,"essays/nwwf.aspx"
    ,"essays/shil.aspx"
    ,"letters/1919-12-11-glm.aspx"
    ,"letters/1927-11-27-cas.aspx"
    ,"other/will-hpl.aspx"
    ,"other/will-aepg.aspx"
    ,"other/epm-ewl.aspx"
]
base = "https://www.hplovecraft.com/writings/texts/"
property_filter = {"face": "Arial,Sans-Serif"}

# Scrape lovecraft texts
lovecraft = []
for link in links:
    url = base + link
    print(url)
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content)
        # get text from elements according to property_filter
        texts = ["\nTITLE:\n"]
        for element in soup.find_all(attrs=property_filter):
            texts.append(element.get_text(separator=" "))
        text = "\n".join(texts)
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        text = re.sub("Return to “.*", "", text)
        text = re.sub("This page last revised .*", "", text)
    except Exception:
        print("Error")
        continue
    lovecraft.append(text)
lovecraft_joined = "".join(lovecraft)
print(lovecraft_joined[:30_000])
lovecraft_clean = unidecode(lovecraft_joined)

chars = sorted(list(set(lovecraft_clean)))
print(f"{chars=}")

out_file = 'lovecraft.txt'
with open(out_file, 'w') as f:
    f.write(lovecraft_clean)
