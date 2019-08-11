import itertools
import json
import re


class Datamaker:

    def __init__(self):
        pass

    @staticmethod
    def single_choice_question_generate(n=1, is_item=True):
        inputs = []
        outputs = []
        for i in range(n):
            input = ['<p>', '_', '</p>']
            output = ['scq_4', 'scq_1', 'scq_4']
            for j in range(n):
                input += ['<p>', str(j+1), '.', ('=' if i == j else '~') + '_', ';', '</p>']
                output += ['scq_4', 'scq_3', 'scq_3', 'scq_2,' + str(j+5), 'scq_3', 'scq_4'] if j < 3 else ['scq_4', 'scq_0', 'scq_0', 'scq_2,' + str(j+5), 'scq_0', 'scq_4']
            input += ['<p>', '_', '</p>']
            output += ['scq_4', 'scq_1', 'scq_4']
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def multi_choice_question_generate(n=1, is_item=True):
        inputs = []
        outputs = []
        products = list(itertools.product(["~_", "=_"], repeat=n))
        for product in products:
            input = ['<p>', '_', '</p>']
            output = ['mcq_0', 'mcq_1', 'mcq_0']
            pos_groups = [2]
            neg_groups = [2]
            for i in range(len(product)):
                if product[i] == "=_":
                    pos_groups.append(5 + i * 2)
                else:
                    neg_groups.append(5 + i * 2)
            pos = round((1.0 / len(pos_groups)) * 100) if len(pos_groups) != 0 else 0
            neg = round((1.0 / len(neg_groups)) * 100) if len(neg_groups) != 0 else 0
            for j in range(len(product)):
                input += ['<p>', str(j + 1), '.', product[j], ';', '</p>']
                if product[j] == "=_":
                    groups = pos_groups[:] + [4 + j * 2]
                else:
                    groups = neg_groups[:] + [4 + j * 2]
                element = ','.join(str(el) for el in sorted(groups))
                output += ['mcq_0', 'mcq_3', 'mcq_3', 'mcq_' + element, 'mcq_3', 'mcq_0'] if j < 3 else ['mcq_4', 'mcq_0', 'mcq_0', 'mcq_' + element, 'mcq_0', 'mcq_4']
            input += ['<p>', '_', '</p>']
            output += ['mcq_0', 'mcq_1', 'mcq_0']
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def text_input_question_generate(n=1, is_item=True):
        inputs = []
        outputs = []
        for i in range(n):
            input = ['<p>', '_', '</p>']
            output = ['tiq_0', 'tiq_1', 'tiq_0']
            for j in range(i+1):
                input += ['<p>', '{=_}', '</p>']
                output += ['tiq_0', 'tiq_' + str(j + 2), 'tiq_0']
            input += ['<p>', '_', '</p>']
            output += ['tiq_0', 'tiq_1', 'tiq_0']
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def number_input_question_generate(n=1, is_item=True):
        inputs = []
        outputs = []
        for i in range(n):
            input = ['<p>', '_', '</p>']
            output = ['niq_0', 'niq_1', 'niq_0']
            for j in range(i + 1):
                input += ['<p>', '{#_}', '</p>']
                output += ['niq_0', 'niq_' + str(j + 2), 'niq_0']
            input += ['<p>', '_', '</p>']
            output += ['niq_0', 'niq_1', 'niq_0']
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def matching_question_generate(n=1, is_item=True):
        inputs = []
        outputs = []
        iterable = [chr(i) for i in range(ord('а'), ord('я') + 1, 1)]
        permutations = itertools.permutations(iterable[:n], n)
        for permutation in permutations:
            input = ['<p>', '_', '</p>']
            output = ['mq_0', 'mq_1', 'mq_0']
            for j in range(len(permutation)):
                input += ['<p>', str(j + 1), ')', '_', '</p>']
                output += ['mq_0', 'mq_3', 'mq_3', 'mq_3', 'mq_0'] if j < 3 else ['mq_0', 'mq_0', 'mq_0', 'mq_0', 'mq_0']
            for j in range(len(permutation)):
                input += ['<p>', iterable[j], ')', '_', '</p>']
                output += ['mq_0', 'mq_0', 'mq_0', 'mq_0', 'mq_0']
            input += ['<p>', 'Ответ:']
            output += ['mq_0', 'mq_0']
            for j in range(len(permutation)):
                input += [str(j + 1) + '-' + permutation[j] + (',' if j < len(permutation) - 1 else '')]
                output += ['mq_' + str(j + 4)]
            input += ['</p>', '<p>', '_', '</p>']
            output += ['mq_0', 'mq_0', 'mq_1', 'mq_0']
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def ordering_question_generate(n=1, is_item=True):
        inputs = []
        outputs = []
        iterable = [str(i + 1) for i in range(n)]
        permutations = itertools.permutations(iterable, n)
        for permutation in permutations:
            input = ['<p>', '_', '</p>']
            output = ['oq_0', 'oq_1', 'oq_0']
            for j in range(len(permutation)):
                input += ['<p>', str(j + 1), ')', '_', '</p>']
                output += ['oq_0', 'oq_3', 'oq_3', 'oq_3', 'oq_0'] if j < 3 else ['oq_0', 'oq_0', 'oq_0', 'oq_0', 'oq_0']
            input += ['<p>', 'Ответ:']
            output += ['oq_0', 'oq_0']
            for j in range(len(permutation)):
                input += [permutation[j] + (',' if j < len(permutation) - 1 else '')]
                output += ['oq_' + str(j + 4)]
            input += ['</p>', '<p>', '_', '</p>']
            output += ['oq_0', 'oq_0', 'oq_1', 'oq_0']
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def composite_question_generate(n=1, is_cloze=True):
        inputs = []
        outputs = []
        scq_inputs, scq_outputs = Datamaker.single_choice_question_generate(n, False)
        tiq_inputs, tiq_outputs = Datamaker.text_input_question_generate(n, False)
        niq_inputs, niq_outputs = Datamaker.number_input_question_generate(n, False)
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
                element.append(json.dumps(iterable))
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
            input_file.write(json.dumps(inputs[i]) + '\n')
            output_file.write(label + '\n')
            output_file.write(json.dumps(outputs[i]) + '\n')
        input_file.close()
        output_file.close()

    @staticmethod
    def clean_files():
        input_file = open('input.txt', 'w', encoding='utf-8')
        input_file.close()
        output_file = open('output.txt', 'w', encoding='utf-8')
        output_file.close()

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
                lhs = re.findall(r'[=~]', masks[i])
                inputs, outputs = Datamaker.single_choice_question_generate(len(lhs))
                for j in range(len(inputs)):
                    rhs = re.findall(r'[=~]', str(inputs[j]))
                    if lhs == rhs:
                        input_file.write('Вопрос 1.\n')
                        input_file.write(json.dumps(inputs[j]) + '\n')
                        output_file.write('Вопрос 1.\n')
                        output_file.write(json.dumps(outputs[j]) + '\n')
            elif re.match(r'Вопрос\s+2\.\s*', types[i]):
                lhs = re.findall(r'[=~]', masks[i])
                inputs, outputs = Datamaker.multi_choice_question_generate(len(lhs))
                for j in range(len(inputs)):
                    rhs = re.findall(r'[=~]', str(inputs[j]))
                    if lhs == rhs:
                        input_file.write('Вопрос 2.\n')
                        input_file.write(json.dumps(inputs[j]) + '\n')
                        output_file.write('Вопрос 2.\n')
                        output_file.write(json.dumps(outputs[j]) + '\n')
            elif re.match(r'Вопрос\s+3\.\s*', types[i]):
                lhs = re.findall(r'{=_}', masks[i])
                inputs, outputs = Datamaker.text_input_question_generate(len(lhs))
                for j in range(len(inputs)):
                    rhs = re.findall(r'{=_}', str(inputs[j]))
                    if lhs == rhs:
                        input_file.write('Вопрос 3.\n')
                        input_file.write(json.dumps(inputs[j]) + '\n')
                        output_file.write('Вопрос 3.\n')
                        output_file.write(json.dumps(outputs[j]) + '\n')
            elif re.match(r'Вопрос\s+4\.\s*', types[i]):
                lhs = re.findall(r'{#_}', masks[i])
                inputs, outputs = Datamaker.number_input_question_generate(len(lhs))
                for j in range(len(inputs)):
                    rhs = re.findall(r'{#_}', str(inputs[j]))
                    if lhs == rhs:
                        input_file.write('Вопрос 4.\n')
                        input_file.write(json.dumps(inputs[j]) + '\n')
                        output_file.write('Вопрос 4.\n')
                        output_file.write(json.dumps(outputs[j]) + '\n')
            elif re.match(r'Вопрос\s+5\.\s*', types[i]):
                lhs = re.findall(r'\d-\w', masks[i])
                inputs, outputs = Datamaker.matching_question_generate(len(lhs))
                for j in range(len(inputs)):
                    rhs = re.findall(r'\d-\w', str(inputs[j]))
                    if lhs == rhs:
                        input_file.write('Вопрос 5.\n')
                        input_file.write(json.dumps(inputs[j]) + '\n')
                        output_file.write('Вопрос 5.\n')
                        output_file.write(json.dumps(outputs[j]) + '\n')
            elif re.match(r'Вопрос\s+6\.\s*', types[i]):
                lhs = re.findall(r'\d', masks[i])
                inputs, outputs = Datamaker.ordering_question_generate(len(lhs))
                for j in range(len(inputs)):
                    rhs = re.findall(r'\d(?!\', \'\))', str(inputs[j]))
                    if lhs == rhs:
                        input_file.write('Вопрос 6.\n')
                        input_file.write(json.dumps(inputs[j]) + '\n')
                        output_file.write('Вопрос 6.\n')
                        output_file.write(json.dumps(outputs[j]) + '\n')
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


if __name__ == "__main__":
    #Datamaker.make_train_dataset(5)
    #Datamaker.make_test_dataset('valid_input.txt')
    Datamaker.make_test_dataset('test_input.txt')
