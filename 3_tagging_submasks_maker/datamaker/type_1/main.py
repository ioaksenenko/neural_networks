import itertools
import json
import re


class Datamaker:

    def __init__(self):
        pass

    @staticmethod
    def single_choice_question_generate(n=1):
        inputs = []
        outputs = []
        for i in range(n):
            input = ['___T____', '________', '________', '<p>_</p>']
            output = ['[item]T[/item]', '[singlechoice][/singlechoice]', 'numbering=123.;', 'view=vertical']
            for _ in range(n):
                input.append('________')
            for j in range(n):
                fragment = '<p>' + str(j + 1) + '.' + ('=' if i == j else '~') + 'T;</p>'
                output.append('[choice value=1]T[/choice]' if i == j else '[choice]T[/choice]')
                for k in range(len(fragment)):
                    input[0] += '_'
                    input[1] += '_' if not (
                                fragment[k] in ['~', '='] or fragment[k] == 'T' and fragment[k - 1] in ['~', '=']) else fragment[k]
                    input[3] += '_' if not (fragment[k] in ['<', '/', 'p', '>']) else fragment[k]
                    for l in range(n):
                        if l == j:
                            input[l + 4] += '_' if not (
                                        fragment[k] in ['~', '='] or fragment[k] == 'T' and fragment[k - 1] in ['~', '=']) else fragment[k]
                        else:
                            input[l + 4] += '_'
                input[2] += '___' + str(j + 1) + '.' + '__;____'
            #inputs.append([re.sub(r'_+', '_', el) for el in input])
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def multi_choice_question_generate(n=1):
        inputs = []
        outputs = []
        products = list(itertools.product(["~T", "=T"], repeat=n))
        for product in products:
            input = ['___T____', '________', '________']
            output = ['[item]T[/item]', '[multichoice][/multichoice]', 'numbering=123.;']
            ra = int((1 / product.count('=T')) * 100) if product.count('=T') > 0 else 0
            wa = int((1 / product.count('~T')) * 100) if product.count('~T') > 0 else 0
            for _ in range(2 * len(product)):
                input.append('________')
            for j in range(len(product)):
                output.append('[choice]T[/choice]')
                output.append('value=-' + str(wa) + '%' if product[j] == '~T' else 'value=' + str(ra) + '%')
                fragment = '<p>' + str(j + 1) + '.' + product[j] + ';</p>'
                for k in range(len(fragment)):
                    input[0] += '_'
                    input[1] += '_' if not (fragment[k] in ['~', '='] or fragment[k] == 'T' and fragment[k - 1] in ['~', '=']) else fragment[k]
                    for l in range(len(product)):
                        if l == j:
                            input[2 * l + 3] += '_' if not (fragment[k] in ['~', '='] or fragment[k] == 'T' and fragment[k - 1] in ['~', '=']) else fragment[k]
                            for t in range(len(product)):
                                if product[t] == product[j]:
                                    input[2 * t + 4] += '_' if not (fragment[k] in ['~', '='] or fragment[k] == 'T' and fragment[k - 1] in ['~', '=']) else fragment[k]
                                else:
                                    input[2 * t + 4] += '_'
                        else:
                            input[2 * l + 3] += '_'
                input[2] += '___' + str(j + 1) + '.' + '__;____'
            if len(re.findall(r'=', input[1])) > 1:
                #inputs.append([re.sub(r'_+', '_', el) for el in input])
                inputs.append(input)
                outputs.append(output)
        return inputs, outputs

    @staticmethod
    def text_input_question_generate(n=1):
        inputs = []
        outputs = []
        input = ['___T____']
        output = ['[item]T[/item]']
        for _ in range(2 * n):
            input.append('________')
        for i in range(n):
            input[0] += '___________' + '________'
            output.append('[textinput][answer]T[/answer][/textinput]')
            output.append('T')
            for j in range(n):
                if j == i:
                    input[2 * j + 1] += '___{=T}____' + '________'
                    input[2 * j + 2] += '___________' + '___T____'
                else:
                    input[2 * j + 1] += '___________' + '________'
                    input[2 * j + 2] += '___________' + '________'
        #inputs.append([re.sub(r'_+', '_', el) for el in input])
        inputs.append(input)
        outputs.append(output)
        #inputs.append([re.sub(r'_+', '_', el) for el in input[:len(input) - 1]])
        inputs.append([el[:len(el) - 8] for el in input][:len(input) - 1])
        outputs.append(output[:len(output) - 1])
        #inputs.append(input[:len(input) - 1])
        #outputs.append(output[:len(output) - 1])
        return inputs, outputs

    @staticmethod
    def number_input_question_generate(n=1):
        inputs = []
        outputs = []
        input = ['___T____']
        output = ['[item]T[/item]']
        for _ in range(2 * n):
            input.append('________')
        for i in range(n):
            input[0] += '___________' + '________'
            output.append('[numberinput][answer]T[/answer][/numberinput]')
            output.append('T')
            for j in range(n):
                if j == i:
                    input[2 * j + 1] += '___{#T}____' + '________'
                    input[2 * j + 2] += '___________' + '___T____'
                else:
                    input[2 * j + 1] += '___________' + '________'
                    input[2 * j + 2] += '___________' + '________'
        #inputs.append([re.sub(r'_+', '_', el) for el in input])
        inputs.append(input)
        outputs.append(output)
        #inputs.append([re.sub(r'_+', '_', el) for el in input[0:len(input) - 1]])
        inputs.append([el[:len(el) - 8] for el in input][:len(input) - 1])
        outputs.append(output[:len(output) - 1])
        return inputs, outputs

    @staticmethod
    def matching_question_generate(n=1):
        inputs = []
        outputs = []
        iterable = [chr(i) for i in range(ord('а'), ord('я') + 1, 1)]
        permutations = itertools.permutations(iterable[:n], n)
        for permutation in permutations:
            input = ['___T____', '________', '________']
            output = ['[item]T[/item]', '[matching][/matching]', 'numbering=123)']
            for _ in range(len(permutation)):
                input.append('________')
            for j in range(len(permutation)):
                fragment = '<p>' + str(j + 1) + ')T</p>'
                for k in range(len(fragment)):
                    input[0] += '_'
                    input[1] += '_'
                    for l in range(3, len(input)):
                        input[l] += '_'
                input[2] += '___' + str(j + 1) + ')_____'
            for j in range(len(permutation)):
                for k in range(len(input)):
                    input[k] += '__________'
            for k in range(len(input)):
                input[k] += '_________'
            for j in range(len(permutation)):
                fragment = str(j + 1) + '-' + permutation[j] + (',' if j < len(permutation) - 1 else '')
                output.append('[subitem]' + str(j + 1) + '[answer]' + permutation[j] + '[/answer][/subitem]')
                for k in range(len(fragment)):
                    input[0] += '_'
                    input[1] += fragment[k]
                    input[2] += '_'
                    for l in range(3, len(input)):
                        if j == l - 3 and fragment[k] != ',':
                            input[l] += fragment[k]
                        else:
                            input[l] += '_'
            for k in range(len(input)):
                input[k] += '____'
            #inputs.append([re.sub(r'_+', '_', el) for el in input])
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def ordering_question_generate(n=1):
        inputs = []
        outputs = []
        iterable = [str(i + 1) for i in range(n)]
        permutations = itertools.permutations(iterable, n)
        for permutation in permutations:
            input = ['___T____', '________', '________']
            output = ['[item]T[/item]', '[ordering][/ordering]', 'numbering=123)']
            for _ in range(len(permutation)):
                input.append('________')
            for j in range(len(permutation)):
                fragment = '<p>' + str(j + 1) + ')T</p>'
                for k in range(len(fragment)):
                    input[0] += '_'
                    input[1] += '_'
                    for l in range(3, len(input)):
                        input[l] += '_'
                input[2] += '___' + str(j + 1) + ')_____'
            for k in range(len(input)):
                input[k] += '_________'
            for j in range(len(permutation)):
                fragment = permutation[j] + (',' if j < len(permutation) - 1 else '')
                output.append('[subitem]' + permutation[j] + '[/subitem]')
                for k in range(len(fragment)):
                    input[0] += '_'
                    input[1] += fragment[k]
                    input[2] += '_'
                    for l in range(3, len(input)):
                        if j == l - 3 and fragment[k] != ',':
                            input[l] += fragment[k]
                        else:
                            input[l] += '_'
            for k in range(len(input)):
                input[k] += '____'
            #inputs.append([re.sub(r'_+', '_', el) for el in input])
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def composite_question_generate(n=1, is_cloze=True):
        inputs = []
        outputs = []
        scq_inputs, scq_outputs = Datamaker.single_choice_question_generate(n)
        tiq_inputs, tiq_outputs = Datamaker.text_input_question_generate(n)
        niq_inputs, niq_outputs = Datamaker.number_input_question_generate(n)
        products = Datamaker.generate_products(scq_inputs, tiq_inputs)
        products += Datamaker.generate_products(scq_inputs, niq_inputs)
        products += Datamaker.generate_products(tiq_inputs, niq_inputs)
        products += Datamaker.generate_products(scq_inputs, tiq_inputs, niq_inputs)
        for product in products:
            for tuple in product:
                input = ""
                for element in tuple:
                    input += element
                inputs.append(re.sub(r'_+', '_', input))
        products = Datamaker.generate_products(scq_outputs, tiq_outputs)
        products += Datamaker.generate_products(scq_outputs, niq_outputs)
        products += Datamaker.generate_products(tiq_outputs, niq_outputs)
        products += Datamaker.generate_products(scq_outputs, tiq_outputs, niq_outputs)
        for product in products:
            for tuple in product:
                output = ("[cloze]_" if is_cloze else "")
                for element in tuple:
                    output += element
                outputs.append(re.sub(r'_+', '_', output) + ("[/cloze]" if is_cloze else ""))
        return inputs, outputs

    @staticmethod
    def generate_products(*iterables):
        res = []
        for iterable in iterables:
            products = []
            product = [iterable]
            for _ in range(len(iterable)):
                product.append(iterable)
                products.append(list(itertools.product(*product)))
            res += products
        products = []
        for iterable in iterables:
            product = []
            element = []
            for _ in iterable:
                element.append(json.dumps(iterable, ensure_ascii=False))
                product.append(element[:])
            products.append(product[:])
        products = list(itertools.product(*products))
        iterables = []
        for product in products:
            iterable = []
            for element in product:
                iterable += element
            iterables.append(iterable)
        for iterable in iterables:
            patterns = list(set(itertools.permutations(iterable, len(iterable))))
            for pattern in patterns:
                product = list(itertools.product(*list(map(json.loads, pattern))))
                res.append(product)
        return res

    @staticmethod
    def write_to_files(inputs, outputs, label):
        input_file = open('input.txt', 'a', encoding='utf-8')
        output_file = open('output.txt', 'a', encoding='utf-8')
        for i in range(len(inputs)):
            input_file.write(label + '\n')
            input_file.write(json.dumps(inputs[i], ensure_ascii=False) + '\n')
            output_file.write(label + '\n')
            output_file.write(json.dumps(outputs[i], ensure_ascii=False) + '\n')
        input_file.close()
        output_file.close()

    @staticmethod
    def clean_files():
        input_file = open('input.txt', 'w', encoding='utf-8')
        input_file.close()
        output_file = open('output.txt', 'w', encoding='utf-8')
        output_file.close()

    @staticmethod
    def select_sample(pattern, mask, question_type, input_file, output_file):
        lhs = re.findall(pattern, json.loads(mask)[1])
        if question_type == 'SCQ':
            inputs, outputs = Datamaker.single_choice_question_generate(len(lhs))
        elif question_type == 'MCQ':
            inputs, outputs = Datamaker.multi_choice_question_generate(len(lhs))
        elif question_type == 'TIQ':
            lhs = re.findall(pattern, mask)
            inputs, outputs = Datamaker.text_input_question_generate(len(lhs))
        elif question_type == 'NIQ':
            lhs = re.findall(pattern, mask)
            inputs, outputs = Datamaker.number_input_question_generate(len(lhs))
        elif question_type == 'MQ':
            inputs, outputs = Datamaker.matching_question_generate(len(lhs))
        elif question_type == 'OQ':
            inputs, outputs = Datamaker.ordering_question_generate(len(lhs))
        elif question_type == 'CQ':
            inputs, outputs = Datamaker.composite_question_generate(len(lhs))
        else:
            print("Unknown question type. There are 7 question types: SCQ, MCQ, TIQ, NIQ, MQ, OQ, CQ.")
            return
        question_numbers = {'SCQ': '1', 'MCQ': '2', 'TIQ': '3', 'NIQ': '4', 'MQ': '5', 'OQ': '6', 'CQ': '7'}
        for i in range(len(inputs)):
            lhs = re.sub('\s*', '', json.dumps(inputs[i], ensure_ascii=False))
            rhs = re.sub('\s*', '', mask)
            if lhs == rhs:
                input_file.write('Вопрос ' + question_numbers[question_type] + '.\n')
                input_file.write(json.dumps(inputs[i], ensure_ascii=False) + '\n')
                output_file.write('Вопрос ' + question_numbers[question_type] + '.\n')
                output_file.write(json.dumps(outputs[i], ensure_ascii=False) + '\n')
                break

    @staticmethod
    def make_test_dataset(file_name):
        file_descriptor = open(file_name, 'r', encoding='utf-8')
        old_data = file_descriptor.read()
        file_descriptor.close()
        input_file = open('input.txt', 'w', encoding='utf-8')
        output_file = open('output.txt', 'w', encoding='utf-8')
        masks = re.split(r'Вопрос\s+[1-7]\.\s*', old_data)[1:]
        types = re.findall(r'Вопрос\s+[1-7]\.\s*', old_data)
        for i in range(len(types)):
            if re.match(r'Вопрос\s+1\.\s*', types[i]):
                Datamaker.select_sample(r'[~=]', masks[i], 'SCQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+2\.\s*', types[i]):
                Datamaker.select_sample(r'[~=]', masks[i], 'MCQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+3\.\s*', types[i]):
                Datamaker.select_sample(r'{=T}', masks[i], 'TIQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+4\.\s*', types[i]):
                Datamaker.select_sample(r'{#T}', masks[i], 'NIQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+5\.\s*', types[i]):
                Datamaker.select_sample(r'\d+-[а-я]', masks[i], 'MQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+6\.\s*', types[i]):
                Datamaker.select_sample(r'\d+[,_]', masks[i], 'OQ', input_file, output_file)
                pass
        input_file.close()
        output_file.close()

    @staticmethod
    def make_train_dataset(n=1):
        Datamaker.clean_files()
        for i in range(n):
            if i > 0:
                inputs, outputs = Datamaker.single_choice_question_generate(i + 1)
                Datamaker.write_to_files(inputs, outputs, 'Вопрос 1.')
            inputs, outputs = Datamaker.multi_choice_question_generate(i + 1)
            Datamaker.write_to_files(inputs, outputs, 'Вопрос 2.')
            inputs, outputs = Datamaker.text_input_question_generate(i + 1)
            Datamaker.write_to_files(inputs, outputs, 'Вопрос 3.')
            inputs, outputs = Datamaker.number_input_question_generate(i + 1)
            Datamaker.write_to_files(inputs, outputs, 'Вопрос 4.')
            if i > 0:
                inputs, outputs = Datamaker.matching_question_generate(i + 1)
                Datamaker.write_to_files(inputs, outputs, 'Вопрос 5.')
                inputs, outputs = Datamaker.ordering_question_generate(i + 1)
                Datamaker.write_to_files(inputs, outputs, 'Вопрос 6.')
            # inputs, outputs = Datamaker.composite_question_generate(i+1)
            # Datamaker.write_to_files(inputs, outputs, 'Вопрос 7.')


if __name__ == "__main__":
    #"""
    inputs, outputs = Datamaker.ordering_question_generate(2)
    f = open('res.txt', 'w', encoding='utf-8')
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            f.write(inputs[i][j] + '\t\t' + outputs[i][j] + '\n')
        f.write('\n')
    #"""
    #Datamaker.make_train_dataset(6)
    #Datamaker.make_test_dataset('patterns.txt')