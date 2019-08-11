import itertools
import json
import re
import math

from functools import reduce


class Datamaker:

    def __init__(self):
        pass

    @staticmethod
    def single_choice_question_generate(n=1, m=1):
        inputs = []
        outputs = []
        for i in range(n):
            input = '<p>T</p>'
            output = ['___T____', '________', '________', '<p>_</p>']
            for _ in range(n):
                output.append('________')
            for j in range(n):
                fragment = '<p>' + ('<s>' if i != j else '') + str(j+1) + '.T;' + ('</s>' if i != j else '') + '</p>'
                for k in range(len(fragment)):
                    output[0] += '_'
                    output[1] += '_' if not (fragment[k] == '<' and (fragment[k+1] == 's' or fragment[k+2] == 's') or fragment[k] == '>' and fragment[k-1] == 's' or fragment[k] == '/' and fragment[k+1] == 's' or fragment[k] in ['s', 'T']) else fragment[k]
                    output[3] += '_' if not (fragment[k] == '<' and (fragment[k+1] == 'p' or fragment[k+2] == 'p') or fragment[k] == '>' and fragment[k-1] == 'p' or fragment[k] == '/' and fragment[k+1] == 'p' or fragment[k] == 'p') else fragment[k]
                    for l in range(n):
                        if l == j:
                            output[l + 4] += '_' if not (fragment[k] == '<' and (fragment[k+1] == 's' or fragment[k+2] == 's') or fragment[k] == '>' and fragment[k-1] == 's' or fragment[k] == '/' and fragment[k+1] == 's' or fragment[k] in ['s', 'T']) else fragment[k]
                        else:
                            output[l + 4] += '_'
                output[2] += '___' + ('___' if i != j else '') + str(j+1) + '.' + '_;' + ('____' if i != j else '') + '____'
                input += fragment
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def multi_choice_question_generate(n=1, m=1):
        inputs = []
        outputs = []
        products = list(itertools.product(["T", "F"], repeat=n))
        for product in products:
            input = '<p>T</p>'
            output = ['___T____', '________', '________']
            for _ in range(2 * len(product)):
                output.append('________')
            for j in range(len(product)):
                fragment = '<p>' + ('<s>' if product[j] == 'F' else '') + str(j + 1) + '.T;' + ('</s>' if product[j] == 'F' else '') + '</p>'
                for k in range(len(fragment)):
                    output[0] += '_'
                    output[1] += '_' if not (fragment[k] == '<' and (fragment[k+1] == 's' or fragment[k+2] == 's') or fragment[k] == '>' and fragment[k-1] == 's' or fragment[k] == '/' and fragment[k+1] == 's' or fragment[k] in ['s', 'T']) else fragment[k]
                    for l in range(len(product)):
                        if l == j:
                            output[2 * l + 3] += '_' if not (fragment[k] == '<' and (fragment[k+1] == 's' or fragment[k+2] == 's') or fragment[k] == '>' and fragment[k-1] == 's' or fragment[k] == '/' and fragment[k+1] == 's' or fragment[k] in ['s', 'T']) else fragment[k]
                            for t in range(len(product)):
                                if product[t] == product[j]:
                                    output[2 * t + 4] += '_' if not (fragment[k] == '<' and (fragment[k+1] == 's' or fragment[k+2] == 's') or fragment[k] == '>' and fragment[k-1] == 's' or fragment[k] == '/' and fragment[k+1] == 's' or fragment[k] in ['s', 'T']) else fragment[k]
                                else:
                                    output[2 * t + 4] += '_'
                        else:
                            output[2 * l + 3] += '_'
                output[2] += '___' + ('___' if product[j] == 'F' else '') + str(j + 1) + '._;' + ('____' if product[j] == 'F' else '') + '____'
                input += fragment
            if len(re.findall(r'(?<=<p>)\d\.T;', input)) > 1:
                inputs.append(input)
                outputs.append(output)
        return inputs, outputs

    @staticmethod
    def text_input_question_generate(n=1, m=1):
        inputs = []
        outputs = []
        input = '<p>T</p>'
        output = ['___T____']
        for _ in range(2 * n):
            output.append('________')
        for i in range(n):
            input += '<p><u>T</u></p>' + '<p>T</p>'
            output[0] += '_______________' + '________'
            for j in range(n):
                if j == i:
                    output[2 * j + 1] += '___<u>T</u>____' + '________'
                    output[2 * j + 2] += '_______________' + '___T____'
                else:
                    output[2 * j + 1] += '_______________' + '________'
                    output[2 * j + 2] += '_______________' + '________'
        inputs.append(input)
        outputs.append(output)
        inputs.append(input[0:len(input)-8])
        outputs.append([el[:len(el)-8] for el in output][:len(output)-1])
        return inputs, outputs

    @staticmethod
    def number_input_question_generate(n=1, m=1):
        inputs = []
        outputs = []
        input = '<p>T</p>'
        output = ['___T____']
        for _ in range(2 * n):
            output.append('________')
        for i in range(n):
            input += '<p><b><u>T</u></b></p>' + '<p>T</p>'
            output[0] += '______________________' + '________'
            for j in range(n):
                if j == i:
                    output[2 * j + 1] += '___<b><u>T</u></b>____' + '________'
                    output[2 * j + 2] += '______________________' + '___T____'
                else:
                    output[2 * j + 1] += '______________________' + '________'
                    output[2 * j + 2] += '______________________' + '________'
        inputs.append(input)
        outputs.append(output)
        inputs.append(input[0:len(input) - 8])
        outputs.append([el[:len(el) - 8] for el in output][:len(output) - 1])
        return inputs, outputs

    @staticmethod
    def matching_question_generate(n=1, m=1):
        inputs = []
        outputs = []
        iterable = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'][:n]
        permutations = itertools.permutations(iterable, n)
        for permutation in permutations:
            input = '<p>T</p><p>T</p><p>'
            output = ['___T_______________', '___________D_______', '___________________', '___________________']
            for _ in range(len(permutation)):
                output.append('___________________')
            for i in range(len(iterable)):
                input += "<span style='color:" + iterable[i] + "'>T</span>" + (',' if i != len(iterable) - 1 else '')
                output[2] += "<span style='color:" + iterable[i] + "'>T</span>" + (',' if i != len(iterable) - 1 else '')
                for j in range(len(output)):
                    if j != 2 and j != i + 4:
                        output[j] += "___________________" + "".join(["_" for _ in iterable[i]]) + "__________" + ('_' if i != len(iterable) - 1 else '')
                    elif j == i + 4:
                        output[j] += "<span style='color:" + iterable[i] + "'>T</span>" + ('_' if i != len(iterable) - 1 else '')
            input += '</p><p>T</p><p>'
            for i in range(len(output)):
                if i != 3:
                    output[i] += '_______________'
                else:
                    output[i] += '_______D_______'
            for i in range(len(permutation)):
                input += "<span style='color:" + permutation[i] + "'>T</span>" + (',' if i != len(permutation) - 1 else '')
                output[2] += "<span style='color:" + permutation[i] + "'>T</span>" + (',' if i != len(permutation) - 1 else '')
                for j in range(len(output)):
                    if j != 2 and j != iterable.index(permutation[i]) + 4:
                        output[j] += "___________________" + "".join(["_" for _ in permutation[i]]) + "__________" + ('_' if i != len(permutation) - 1 else '')
                    elif j == iterable.index(permutation[i]) + 4:
                        output[j] += "<span style='color:" + permutation[i] + "'>T</span>" + ('_' if i != len(permutation) - 1 else '')
            input += '</p>'
            for i in range(len(output)):
                output[i] += '____'
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def ordering_question_generate(n=1, m=1):
        inputs = []
        outputs = []
        iterable = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'][:n]
        permutations = itertools.permutations(iterable, n)
        for permutation in permutations:
            input = '<p>T</p><p>T</p><p>'
            output = ['___T_______________', '___________D_______', '___________________']
            for _ in range(len(permutation)):
                output.append('___________________')
            for i in range(len(iterable)):
                input += "<span style='color:" + permutation[i] + "'>T</span>" + (',' if i != len(permutation) - 1 else '')
                output[2] += "<span style='color:" + permutation[i] + "'>T</span>" + (',' if i != len(permutation) - 1 else '')
                for j in range(len(output)):
                    if j != 2 and j != i + 3:
                        output[j] += "___________________" + "".join(["_" for _ in permutation[i]]) + "__________" + ('_' if i != len(permutation) - 1 else '')
                    elif j == i + 3:
                        output[j] += "<span style='color:" + permutation[i] + "'>T</span>" + ('_' if i != len(permutation) - 1 else '')
            input += '</p>'
            for i in range(len(output)):
                output[i] += '____'
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def composite_question_generate(n1=1, n2=1, n3=1, m=1):
        inputs = []
        outputs = []
        scq_inputs, scq_outputs = Datamaker.single_choice_question_generate(n1)
        tiq_inputs, tiq_outputs = Datamaker.text_input_question_generate(n2)
        niq_inputs, niq_outputs = Datamaker.number_input_question_generate(n3)
        products = Datamaker.generate_products(scq_inputs, tiq_inputs)
        products += Datamaker.generate_products(scq_inputs, niq_inputs)
        products += Datamaker.generate_products(scq_inputs, tiq_inputs, niq_inputs)
        for product in products:
            for tuple in product:
                input = ""
                for element in tuple:
                    input += element
                inputs.append(re.sub(r'_+', '_', input))
        products = Datamaker.generate_products(scq_outputs, tiq_outputs)
        products += Datamaker.generate_products(scq_outputs, niq_outputs)
        products += Datamaker.generate_products(scq_outputs, tiq_outputs, niq_outputs)
        for product in products:
            for tuple in product:
                output = []
                for element in tuple:
                    output += element
                outputs.append(output)
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
    def select_sample(pattern, mask, question_type, input_file, output_file):
        lhs = re.findall(pattern, mask)
        if question_type == 'SCQ':
            inputs, outputs = Datamaker.single_choice_question_generate(len(lhs))
        elif question_type == 'MCQ':
            inputs, outputs = Datamaker.multi_choice_question_generate(len(lhs))
        elif question_type == 'TIQ':
            inputs, outputs = Datamaker.text_input_question_generate(len(lhs))
        elif question_type == 'NIQ':
            inputs, outputs = Datamaker.number_input_question_generate(len(lhs))
        elif question_type == 'MQ':
            inputs, outputs = Datamaker.matching_question_generate(int(len(lhs)/2))
        elif question_type == 'OQ':
            inputs, outputs = Datamaker.ordering_question_generate(len(lhs))
        elif question_type == 'CQ':
            inputs, outputs = Datamaker.composite_question_generate(len(lhs))
        else:
            print("Unknown question type. There are 7 question types: SCQ, MCQ, TIQ, NIQ, MQ, OQ, CQ.")
            return
        question_numbers = {'SCQ': '1', 'MCQ': '2', 'TIQ': '3', 'NIQ': '4', 'MQ': '5', 'OQ': '6', 'CQ': '7'}
        for i in range(len(inputs)):
            if re.sub('\s*', '', inputs[i]) == re.sub('\s*', '', mask):
                input_file.write('Вопрос ' + question_numbers[question_type] + '.\n')
                input_file.write(inputs[i] + '\n')
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
                Datamaker.select_sample(r'\d+\.', masks[i], 'SCQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+2\.\s*', types[i]):
                Datamaker.select_sample(r'\d+\.', masks[i], 'MCQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+3\.\s*', types[i]):
                Datamaker.select_sample(r'<u>', masks[i], 'TIQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+4\.\s*', types[i]):
                Datamaker.select_sample(r'<u>', masks[i], 'NIQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+5\.\s*', types[i]):
                Datamaker.select_sample(r'<span', masks[i], 'MQ', input_file, output_file)
                pass
            elif re.match(r'Вопрос\s+6\.\s*', types[i]):
                Datamaker.select_sample(r'<span', masks[i], 'OQ', input_file, output_file)
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
    """
    inputs, outputs = Datamaker.ordering_question_generate(3)
    f = open('res.txt', 'w', encoding='utf-8')
    for i in range(len(inputs)):
        f.write(inputs[i] + '\n')
        for j in range(len(outputs[i])):
            f.write(outputs[i][j] + '\n')
        f.write('\n\n')
    """
    #Datamaker.make_train_dataset(6)
    Datamaker.make_test_dataset('patterns.txt')
