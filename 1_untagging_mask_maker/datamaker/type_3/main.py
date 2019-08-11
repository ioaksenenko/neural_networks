import re
from functools import reduce
import logging

logger = logging.getLogger("datamaker")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("stack.log", "w", encoding='utf-8')
formatter = logging.Formatter('%(filename)s[LINE:%(lineno)d]# %(levelname)s [%(asctime)s]  %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.propagate = False


def processing(fragments, patterns):
    logger.info('fragments: ' + str(fragments))
    logger.info('patterns: ' + str(patterns))
    res = []
    logger.info('res: ' + str(res))
    if len(patterns) > 0:
        next_patterns = patterns[:]
        pattern = next_patterns.pop(0)
        logger.info('pattern: ' + str(pattern))
        logger.info('next_patterns: ' + str(next_patterns))
        for i in range(len(fragments)):
            fragment = fragments[i]
            if fragment == '':
                res.append('')
                continue
            logger.info('fragment: ' + str(fragment))
            parts = re.findall(pattern[0], fragment)
            logger.info('parts: ' + str(parts))
            if len(parts) > 0:
                parts[0] = parts[0] if type(parts[0]) is not tuple else parts[0][0]
                next_fragments = re.split(re.escape(parts[0]), fragment)
                logger.info('next_fragments: ' + str(next_fragments))
                logger.info('-> 34')
                prev_res = processing(next_fragments, next_patterns)
                logger.info('<- 34')
                logger.info('prev_res: ' + str(prev_res))
                next_res = []
                for i in range(len(prev_res)):
                    if pattern[1] and i == 0 or not pattern[1] and i == len(prev_res) - 1:
                        next_res.append(prev_res[i])
                    elif pattern[1]:
                        next_res.append(parts[0] + prev_res[i])
                    elif not pattern[1]:
                        next_res.append(prev_res[i] + parts[0])
                logger.info('next_res: ' + str(next_res))
                res.append(''.join(next_res))
                logger.info('res: ' + str(res))
            else:
                if len(next_patterns) == 0:
                    res.append(reduce(lambda lhs, rhs: lhs + '_', '_' + fragment[1:]) if fragment != "" else "")
                    logger.info('res: ' + str(res))
                else:
                    logger.info('-> 62')
                    logger.info('res: ' + str(res))
                    res.append(''.join(processing(fragments[i:], next_patterns)))
                    logger.info('res: ' + str(res))
                    logger.info('<- 62')
    else:
        logger.info('len(patterns) = 0')
        for fragment in fragments:
            res.append(reduce(lambda lhs, rhs: lhs + '_', '_' + fragment[1:]) if fragment != "" else "")
        logger.info('res: ' + str(res))
    return res


def scq(sample):
    fragments = list(filter(None, re.split(r'<p>', sample)))
    patterns = [[r'</p>', False], [r'<s>', True], [r'</s>', False], [r'\d\.\s', True], [r';', False]]
    parts = re.findall(r'<p>', sample)
    res = processing(fragments, patterns)
    res = ''.join([parts[i] + res[i] for i in range(len(res))])
    return res


def mcq(sample):
    fragments = list(filter(None, re.split(r'<p>', sample)))
    patterns = [[r'</p>', False], [r'<s>', True], [r'</s>', False], [r'\d\.\s', True], [r';', False]]
    parts = re.findall(r'<p>', sample)
    res = processing(fragments, patterns)
    res = ''.join([parts[i] + res[i] for i in range(len(res))])
    return res


def tiq(sample):
    fragments = list(filter(None, re.split(r'<p>', sample)))
    patterns = [[r'</p>', False], [r'<u>', True], [r'</u>', False]]
    parts = re.findall(r'<p>', sample)
    res = processing(fragments, patterns)
    res = ''.join([parts[i] + res[i] for i in range(len(res))])
    return res


def niq(sample):
    fragments = list(filter(None, re.split(r'<p>', sample)))
    patterns = [[r'</p>', False], [r'<b><u>', True], [r'</u></b>', False]]
    parts = re.findall(r'<p>', sample)
    res = processing(fragments, patterns)
    res = ''.join([parts[i] + res[i] for i in range(len(res))])
    return res


def mq(sample):
    fragments = list(filter(None, re.split(r'<p>', sample)))
    patterns = [[r'</p>', False], [r'</span>,\s', False], [r'<span\sstyle=\'color:[a-z]+\'>', True], [r'</span>', False]]
    parts = re.findall(r'<p>', sample)
    res = processing(fragments, patterns)
    res = ''.join([parts[i] + res[i] for i in range(len(res))])
    return res


def oq(sample):
    fragments = list(filter(None, re.split(r'<p>', sample)))
    patterns = [[r'</p>', False], [r'</span>,\s', False], [r'<span\sstyle=\'color:[a-z]+\'>', True], [r'</span>', False]]
    parts = re.findall(r'<p>', sample)
    res = processing(fragments, patterns)
    res = ''.join([parts[i] + res[i] for i in range(len(res))])
    return res


def cq(sample):
    fragments = list(filter(None, re.split(r'<p>', sample)))
    patterns = [[r'</p>', False], [r'<s>', True], [r'</s>', False], [r'\d\.\s', True], [r';', False], [r'(<b><u>|<u>)', True], [r'(</u></b>|</u>)', False]]
    parts = re.findall(r'<p>', sample)
    res = processing(fragments, patterns)
    res = ''.join([parts[i] + res[i] for i in range(len(res))])
    return res


def test():
    scq_sample = "<p>Укажите административный центр Томской области:</p><p><s>1. Асино;</s></p><p><s>2. Северск;</s></p><p><s>3. Стрежевой;</s></p><p>4. Томск;</p>"
    mcq_sample = "<p>Укажите города Томской области:</p><p>1. Томск;</p><p><s>2. Кожевниково;</s></p><p>3. Северск;</p><p>4. Асино;</p><p><s>5. Юрга;</s></p>"
    tiq_sample = "<p>Какой город является административным центром Томской области?</p><p><u>Томск</u></p>"
    niq_sample = "<p>Укажите численность населения г. Томска в 2016 году. Ответ укажите в тыс. чел.</p><p><b><u>569 293</u></b></p>"
    mq_sample = "<p>Сопоставьте населённые пункты Томской области и год их основания.</p><p>Населённые пункты:</p><p><span style='color:red'>Томск</span>, <span style='color:orange'>Северск</span>, <span style='color:yellow'>Каргасок</span>, <span style='color:green'>Кривошеино</span></p><p>Года основания:</p><p><span style='color:yellow'>1640</span>, <span style='color:red'>1604</span>, <span style='color:green'>1826</span>, <span style='color:orange'>1949</span></p>"
    #mq_sample = "<p>Соотнесите способ предоставления и виды социальных услуг:</p><p>Способы предоставления социальных услуг:</p><p><span style='color:red'>организация посильной трудовой деятельности и поддержание активного образа жизни</span>, <span style='color:orange'>поддержание условий проживания в соответствии с гигиеническими требованиями</span>, <span style='color:yellow'>постоянный уход и наблюдение</span>, <span style='color:green'>услуги телефона доверия</span></p><p>Виды социальных услуг:</p><p><span style='color:orange'>услуги предоставляемые на дому</span>, <span style='color:yellow'>услуги предоставляемые в стационарных учреждениях</span>, <span style='color:red'>услуги предоставляемые в полустационарных учреждениях</span>, <span style='color:green'>услуги предоставляемые в рамках срочной социальной помощи</span></p>"
    oq_sample = "<p>Расставьте населённые пункты Томской области по порядку согласно году их основания.</p><p>Населённые пункты:</p><p><span style='color:yellow'>Кривошеино</span>, <span style='color:orange'>Каргасок</span>, <span style='color:red'>Томск</span>, <span style='color:green'>Северск</span></p>"
    cq_sample = "<p>Укажите административный центр Томской области:</p><p><s>1. Асино;</s></p><p><s>2. Северск;</s></p><p><s>3. Стрежевой;</s></p><p>4. Томск;</p><p>Он был основан в <b><u>1604</u></b> году.</p>"
    res = [scq(scq_sample), mcq(mcq_sample), tiq(tiq_sample), niq(niq_sample), mq(mq_sample), oq(oq_sample), cq(cq_sample)]
    #res = [mq(mq_sample)]
    fd = open('output.txt', 'w', encoding='utf-8')
    for el in res:
        fd.write(el + '\n')
    fd.close()


if __name__ == "__main__":
    #test()
    fd = open('input.txt', 'r', encoding='utf-8')
    fc = fd.read().replace('\n', '')
    fd.close()
    fd = open('output.txt', 'w', encoding='utf-8')
    qh = re.findall(r'Вопрос\s+[1-7]\.\s*', fc)
    qb = re.split(r'Вопрос\s+[1-7]\.\s*', fc)[1:]
    for i in range(len(qh)):
        if re.match(r'Вопрос\s+1\.\s*', qh[i]):
            fd.write('Вопрос 1.\n')
            fd.write(scq(qb[i]) + '\n')
        elif re.match(r'Вопрос\s+2\.\s*', qh[i]):
            fd.write('Вопрос 2.\n')
            fd.write(mcq(qb[i]) + '\n')
        elif re.match(r'Вопрос\s+3\.\s*', qh[i]):
            fd.write('Вопрос 3.\n')
            fd.write(tiq(qb[i]) + '\n')
        elif re.match(r'Вопрос\s+4\.\s*', qh[i]):
            fd.write('Вопрос 4.\n')
            fd.write(niq(qb[i]) + '\n')
        elif re.match(r'Вопрос\s+5\.\s*', qh[i]):
            fd.write('Вопрос 5.\n')
            fd.write(mq(qb[i]) + '\n')
        elif re.match(r'Вопрос\s+6\.\s*', qh[i]):
            fd.write('Вопрос 6.\n')
            fd.write(oq(qb[i]) + '\n')
        elif re.match(r'Вопрос\s+7\.\s*', qh[i]):
            fd.write('Вопрос 7.\n')
            fd.write(cq(qb[i]) + '\n')
        else:
            print("Undefined question type '" + qh[i] + "'!")