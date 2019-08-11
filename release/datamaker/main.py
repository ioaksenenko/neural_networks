import itertools
import json
import re
import math

from functools import reduce


class Datamaker:

    def __init__(self):
        pass

    @staticmethod
    def single_choice_question_generate(n=1):
        inputs = []
        outputs = []
        for i in range(n):
            input = '<p>_</p>'
            output = [[4], [4], [4], [1], [4], [4], [4], [4]]
            for j in range(n):
                input += '<p>' + str(j+1) + '.' + ('=' if i == j else '~') + '_;</p>'
                output += [[4], [4], [4]]
                for _ in str(j+1):
                    output += [[3]]
                output += [[3], [2, j + 5], [2, j + 5], [3], [4], [4], [4], [4]]
            input += '<p>_</p>'
            output += [[4], [4], [4], [1], [4], [4], [4], [4]]
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def multi_choice_question_generate(n=1):
        inputs = []
        outputs = []
        products = list(itertools.product(["~_", "=_"], repeat=n))
        for product in products:
            input = '<p>_</p>'
            output = [[0], [0], [0], [1], [0], [0], [0], [0]]
            pos_groups = [2]
            neg_groups = [2]
            for i in range(len(product)):
                if product[i] == "=_":
                    pos_groups.append(5 + i * 2)
                else:
                    neg_groups.append(5 + i * 2)
            for j in range(len(product)):
                input += '<p>' + str(j + 1) + '.' + product[j] + ';</p>'
                if product[j] == "=_":
                    groups = pos_groups[:] + [4 + j * 2]
                else:
                    groups = neg_groups[:] + [4 + j * 2]
                output += [[0], [0], [0]]
                for _ in str(j + 1):
                    output += [[3]]
                groups = sorted(groups)
                output += [[3], groups, groups, [3], [0], [0], [0], [0]]
            input += '<p>_</p>'
            output += [[0], [0], [0], [1], [0], [0], [0], [0]]
            if len(re.findall(r'=', input)) > 1:
                inputs.append(input)
                outputs.append(output)
        return inputs, outputs

    @staticmethod
    def text_input_question_generate(n=1):
        inputs = []
        outputs = []
        for i in range(n):
            input = '<p>_</p>'
            output = [[0], [0], [0], [1], [0], [0], [0], [0]]
            for j in range(i+1):
                input += '<p>{=_}</p><p>_</p>'
                output += [[0], [0], [0], [j + 2], [j + 2], [j + 2], [j + 2], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def number_input_question_generate(n=1):
        inputs = []
        outputs = []
        for i in range(n):
            input = '<p>_</p>'
            output = [[0], [0], [0], [1], [0], [0], [0], [0]]
            for j in range(i + 1):
                input += '<p>{#_}</p><p>_</p>'
                output += [[0], [0], [0], [j + 2], [j + 2], [j + 2], [j + 2], [0], [0], [0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def matching_question_generate(n=1):
        inputs = []
        outputs = []
        iterable = [chr(i) for i in range(ord('а'), ord('я') + 1, 1)]
        permutations = itertools.permutations(iterable[:n], n)
        for permutation in permutations:
            input = '<p>_</p>'
            output = [[0], [0], [0], [1], [0], [0], [0], [0]]
            for j in range(len(permutation)):
                input += '<p>' + str(j + 1) + ')_</p>'
                output += [[0], [0], [0]]
                for _ in str(j + 1):
                    output += [[3]]
                output += [[3], [3], [0], [0], [0], [0]]
            for j in range(len(permutation)):
                input += '<p>' + iterable[j] + ')_</p>'
                output += [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
            input += '<p>Ответ:'
            output += [[0], [0], [0], [0], [0], [0], [0], [0], [0]]
            for j in range(len(permutation)):
                fragment = str(j + 1) + '-' + permutation[j] + (',' if j < len(permutation) - 1 else '')
                input += fragment
                for _ in fragment:
                    output += [[2, j + 4]]
            input += '</p>'
            output += [[0], [0], [0], [0]]
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
            input = '<p>_</p>'
            output = [[0], [0], [0], [1], [0], [0], [0], [0]]
            for j in range(len(permutation)):
                input += '<p>' + str(j + 1) + ')_</p>'
                output += [[0], [0], [0]]
                for _ in str(j + 1):
                    output += [[3]]
                output += [[3], [3], [0], [0], [0], [0]]
            input += '<p>Ответ:'
            output += [[0], [0], [0], [0], [0], [0], [0], [0], [0]]
            for j in range(len(permutation)):
                fragment = permutation[j] + (',' if j < len(permutation) - 1 else '')
                input += fragment
                for _ in fragment:
                    output += [[2, j + 4]]
            input += '</p>'
            output += [[0], [0], [0], [0]]
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
            input_file.write(inputs[i] + '\n')
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
    def select_sample(lhs_pattern, rhs_pattern, mask, question_type, input_file, output_file):
        lhs = re.findall(lhs_pattern, mask)
        if question_type == 'SCQ':
            inputs, outputs = Datamaker.single_choice_question_generate(len(lhs))
        elif question_type == 'MCQ':
            inputs, outputs = Datamaker.multi_choice_question_generate(len(lhs))
        elif question_type == 'TIQ':
            inputs, outputs = Datamaker.text_input_question_generate(len(lhs))
        elif question_type == 'NIQ':
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
            rhs = re.findall(rhs_pattern, inputs[i])
            if lhs == rhs:
                input_file.write('Вопрос ' + question_numbers[question_type] + '.\n')
                input_file.write(inputs[i] + '\n')
                output_file.write('Вопрос ' + question_numbers[question_type] + '.\n')
                output_file.write(json.dumps(outputs[i], ensure_ascii='utf-8') + '\n')
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
                Datamaker.select_sample(r'[=~]', r'[=~]', masks[i], 'SCQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+2\.\s*', types[i]):
                Datamaker.select_sample(r'[=~]', r'[=~]', masks[i], 'MCQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+3\.\s*', types[i]):
                Datamaker.select_sample(r'{=_}', r'{=_}', masks[i], 'TIQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+4\.\s*', types[i]):
                Datamaker.select_sample(r'{#_}', r'{#_}', masks[i], 'NIQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+5\.\s*', types[i]):
                Datamaker.select_sample(r'\d-\w', r'\d-\w', masks[i], 'MQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+6\.\s*', types[i]):
                Datamaker.select_sample(r'\d(?=[,<])', r'\d(?=[,<])', masks[i], 'OQ', input_file, output_file)
                pass
        input_file.close()
        output_file.close()

    @staticmethod
    def make_train_dataset(n=1):
        Datamaker.clean_files()
        for i in range(n):
            inputs, outputs = Datamaker.single_choice_question_generate(i + 1)
            Datamaker.write_to_files(inputs, outputs, 'Вопрос 1.')
            inputs, outputs = Datamaker.multi_choice_question_generate(i + 1)
            Datamaker.write_to_files(inputs, outputs, 'Вопрос 2.')
            inputs, outputs = Datamaker.text_input_question_generate(i + 1)
            Datamaker.write_to_files(inputs, outputs, 'Вопрос 3.')
            inputs, outputs = Datamaker.number_input_question_generate(i + 1)
            Datamaker.write_to_files(inputs, outputs, 'Вопрос 4.')
            inputs, outputs = Datamaker.matching_question_generate(i + 1)
            Datamaker.write_to_files(inputs, outputs, 'Вопрос 5.')
            inputs, outputs = Datamaker.ordering_question_generate(i + 1)
            Datamaker.write_to_files(inputs, outputs, 'Вопрос 6.')
            # inputs, outputs = Datamaker.composite_question_generate(i+1)
            # Datamaker.write_to_files(inputs, outputs, 'Вопрос 7.')

    @staticmethod
    def encoding(outputs, qt=0):
        length = reduce(lambda x, y: x if x > y else y,
                        list(map(lambda output:
                                 reduce(lambda x, y: x if x > y else y,
                                        list(map(lambda groups:
                                                 reduce(lambda x, y: x if x > y else y, groups), output))), outputs)))
        gr_codes = list(itertools.product('01', repeat=math.ceil(math.log2(length))))
        res = list(map(lambda output:
                       list(map(lambda groups:
                                list(map(lambda group: list(gr_codes[group]), groups)), output)), outputs))
        res = list(map(lambda output:
                       list(map(lambda groups:
                                reduce(lambda x, y: x + y, groups), output)), res))
        qt_codes = list(itertools.product('01', repeat=3))
        res = list(map(lambda output:
                       list(map(lambda x:
                                list(qt_codes[qt]) + x, output)), res))
        return res


if __name__ == "__main__":
    #Datamaker.make_train_dataset(6)
    #Datamaker.make_test_dataset('valid_input.txt')
    Datamaker.make_test_dataset('test_input.txt')
