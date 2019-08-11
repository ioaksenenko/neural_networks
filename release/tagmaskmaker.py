import json
import re


def read_input(file_path='txt\\input.txt'):
    f = open(file_path, 'r', encoding='utf-8')
    fc = f.read()
    f.close()
    h = list(filter(lambda x: x != '', re.findall(r'Вопрос\s+\d+\.(?:\d+\.)*\s*', fc)))
    b = list(filter(lambda x: x != '', re.split(r'Вопрос\s+\d+\.(?:\d+\.)*\s*', fc)))
    res = []
    for i in range(len(h)):
        res.append({"header": h[i].replace('\n', ''), "body": json.loads(b[i].replace('\n', ''))})
    return res


def read_rules(file_path='json\\rules.json'):
    f = open(file_path, 'r', encoding='utf-8')
    fc = f.read()
    f.close()
    res = json.loads(fc)
    return res


def tagging_mask_maker(inputs, rules):
    res = []
    for input in inputs:
        elements = parse(input["body"])
        tree = structure(elements, rules)
        res.append({"header": input["header"], "body": join(tree)})
    return res


def parse(elements):
    res = []
    for element in elements:
        m = re.match(r'\[[a-z]+(\s+(?P<attrs>.*))?\](?P<content>.*)\[/(?P<tag>.+)\]', element)
        if m is not None:
            res.append({'type': 'tag', 'name': m.group('tag'), 'content': m.group('content'), 'attrs': [m.group('attrs')] if m.group('attrs') is not None else []})
        else:
            m = re.match(r'(?P<attr>.+)=(?P<val>.+)', element)
            if m is not None:
                res.append({'type': 'attr', 'name': m.group('attr'), 'val': m.group('val')})
    return res


def structure(elements, rules):
    res = {"attrs": [], "content": []}
    rules = {"content": rules, "attributes": []}
    deep_structure(elements, 0, rules, res)
    return res


def deep_structure(elements, index, rules, res):
    while index < len(elements):
        if elements[index]['type'] == 'tag':
            if elements[index]['name'] in rules['content']:
                res['content'].append({
                    elements[index]['name']: {
                        'attrs': elements[index]['attrs'],
                        'content': [elements[index]['content']] if elements[index]['content'] is not '' else []
                    }
                })
                index = deep_structure(
                    elements,
                    index + 1,
                    rules['content'][elements[index]['name']],
                    res['content'][len(res['content']) - 1][elements[index]['name']]
                )
            else:
                return index
        elif elements[index]['type'] == 'attr':
            if elements[index]['name'] in rules['attributes']:
                res['attrs'].append(elements[index]['name'] + '=' + elements[index]['val'])
                index = deep_structure(elements, index + 1, rules, res)
            else:
                return index
    return index


def join(tree):
    res = ""
    for element in tree['content']:
        if type(element) is str:
            res += element
        else:
            for key, val in element.items():
                res += '[' + key + (' ' if len(val['attrs']) > 0 else '') + ' '.join(val['attrs']) + ']' + join(val) + '[/' + key + ']'
    return res


def write_output(outputs, file_path='txt\\output.txt'):
    f = open(file_path, 'w', encoding='utf-8')
    for output in outputs:
        f.write(output['header'] + '\n')
        f.write(output['body'] + '\n')
    f.close()


if __name__ == "__main__":
    input_dir = "txt\\input.txt"
    output_dir = "txt\\output.txt"
    rules_dir = "json\\rules.json"

    inputs = read_input(input_dir)
    rules = read_rules(rules_dir)
    outputs = tagging_mask_maker(inputs, rules)
    write_output(outputs, output_dir)
