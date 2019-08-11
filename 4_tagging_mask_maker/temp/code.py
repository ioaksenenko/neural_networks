def deep_structure(elements, index, rules, res, prev_rules=None, prev_res=None):
    if elements[index]['type'] == 'tag':
        if elements[index]['name'] in rules['content']:
            res['content'].append({elements[index]['name']: {'attrs': elements[index]['attrs'], 'content': [elements[index]['content']] if elements[index]['content'] is not '' else []}})
            if index + 1 < len(elements):
                deep_structure(elements, index + 1, rules['content'][elements[index]['name']], res['content'][len(res['content']) - 1][elements[index]['name']], rules, res)
                deep_structure(elements, index + 1, rules, res)
        else:
            return
    elif elements[index]['type'] == 'attr':
        if elements[index]['name'] in rules['attributes']:
            res['attrs'].append(elements[index]['name'] + '=' + elements[index]['val'])
            if index + 1 < len(elements):
                deep_structure(elements, index + 1, rules, res)
                if prev_rules is not None and prev_res is not None:
                    deep_structure(elements, index + 1, prev_rules, prev_res)
        else:
            return