import re


def read_input(file_path='txt\\input.txt'):
    f = open(file_path, 'r', encoding='utf-8')
    fc = f.read()
    f.close()
    h = list(filter(lambda x: x != '', re.findall(r'Вопрос\s+\d+\.(?:\d+\.)*\s*', fc)))
    b = list(filter(lambda x: x != '', re.split(r'Вопрос\s+\d+\.(?:\d+\.)*\s*', fc)))
    res = []
    for i in range(len(h)):
        res.append({"header": h[i].replace('\n', ''), "body": b[i].replace('\n', '')})
    return res


def get_content(tmasks):
    res = []
    for tmask in tmasks:
        elements = list(set(re.findall(r'(\[/?[a-z]+(?:\s+[a-z]+=[a-z0-9-%.;)]+)*\])', tmask['body'])))
        fragments = [tmask['body']]
        for element in elements:
            res_fragments = []
            for fragment in fragments:
                res_fragments += fragment.split(element)
            fragments = res_fragments
        res.append({'header': tmask['header'], 'body': list(filter(lambda x : x != '', fragments))})
    return res


def get_text(source, masks, content):
    res = []
    for i in range(len(masks)):
        fragments = []
        pred_fragments = []
        fragment = ""
        pred_fragment = ""
        for j in range(len(masks[i]['body'])):
            if masks[i]['body'][j] == '_':
                fragment += source[i]['body'][j]
            else:
                if fragment != "":
                    fragments.append(fragment)
                    pred_fragments.append(pred_fragment)
                    fragment = ""
                    pred_fragment = masks[i]['body'][j]
                else:
                    pred_fragment += masks[i]['body'][j]
        res_fragments = []
        for j in range(len(fragments)):
            if content[i]['body'][j] == 'T':
                res_fragments.append(fragments[j])
            else:
                for k in range(len(pred_fragments)):
                    if content[i]['body'][j] in pred_fragments[k]:
                        res_fragments.append(fragments[k])
        res.append({"header": masks[i]['header'], "body": res_fragments})
    return res


def replace(tmasks, content, text):
    res = []
    for i in range(len(tmasks)):
        res_tmask = ""
        pointer = 0
        for j in range(len(content[i]['body'])):
            for k in range(pointer, len(tmasks[i]['body'])):
                if tmasks[i]['body'][k] != content[i]['body'][j]:
                    res_tmask += tmasks[i]['body'][k]
                else:
                    if tmasks[i]['body'][k-1] == ']' and tmasks[i]['body'][k+1] == '[':
                        res_tmask += text[i]['body'][j]
                        pointer = k + 1
                        break
                    else:
                        res_tmask += tmasks[i]['body'][k]
        for k in range(pointer, len(tmasks[i]['body'])):
            res_tmask += tmasks[i]['body'][k]
        res.append({'header': tmasks[i]['header'], 'body': res_tmask})
    return res


def write_output(outputs, file_path='txt\\output.txt'):
    f = open(file_path, 'w', encoding='utf-8')
    for output in outputs:
        f.write(output['header'] + '\n')
        f.write(output['body'] + '\n')
    f.close()


if __name__ == "__main__":
    source_dir = 'txt\\source.txt'
    masks_dir = 'txt\\masks.txt'
    tmasks_dir = 'txt\\tmasks.txt'
    output_dir = 'txt\\output.txt'

    source = read_input(source_dir)
    masks = read_input(masks_dir)
    tmasks = read_input(tmasks_dir)

    content = get_content(tmasks)
    text = get_text(source, masks, content)
    tmasks = replace(tmasks, content, text)
    write_output(tmasks, output_dir)
