import re
import os

fd = open('input.txt', 'r', encoding='utf-8')
ic = fd.read()
fd.close()

fd = open('output.txt', 'r', encoding='utf-8')
oc = fd.read()
fd.close()

iq = re.split(r'Вопрос\s+\d+\.(?:\d+\.)*\s*', ic)[1:]
oq = re.split(r'Вопрос\s+\d+\.(?:\d+\.)*\s*', oc)[1:]

it = [[], [], [], [], [], [], []]
ot = [[], [], [], [], [], [], []]
ts = ["[s]", "[mu]", "[text]", "[number]", "[mat]", "[ord]", "[cloze]"]

for i in range(len(oq)):
    tag = oq[i].split('</p>')[0].split('<p>')[1]
    for j in range(len(ts)):
        if tag == ts[j]:
            it[j].append(iq[i])
            ot[j].append(oq[i])

n = max(list(map(len, it)))
for j in range(n):
    ifd = open(os.path.join('input', str(j + 1) + '.txt'), 'w', encoding='utf-8')
    ofd = open(os.path.join('output', str(j + 1) + '.txt'), 'w', encoding='utf-8')
    for i in range(len(it)):
        if len(it[i]) > j:
            ifd.write('Вопрос ' + str(i + 1) + '.\n')
            ifd.write(it[i][j])
            ofd.write('Вопрос ' + str(i + 1) + '.\n')
            ofd.write(ot[i][j])
    ifd.close()
    ofd.close()
