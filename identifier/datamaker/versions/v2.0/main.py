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
            input = '_'
            for j in range(n):
                input += ('=' if i == j else '~') + '_'
            inputs.append(input)
            outputs.append(json.dumps([1, 0, 0, 0, 0, 0, 0]))
        return inputs, outputs

    @staticmethod
    def multi_choice_question_generate(n=1):
        inputs = []
        outputs = []
        products = list(itertools.product(["~_", "=_"], repeat=n))
        for product in products:
            input = '_'
            for j in range(len(product)):
                input += product[j]
            inputs.append(input)
            outputs.append(json.dumps([0, 1, 0, 0, 0, 0, 0]))
        return inputs, outputs

    @staticmethod
    def text_input_question_generate(n=1):
        inputs = []
        outputs = []
        for i in range(n):
            input = '_'
            for j in range(i+1):
                input += '{=_}_'
            inputs.append(input)
            outputs.append(json.dumps([0, 0, 1, 0, 0, 0, 0]))
        return inputs, outputs

    @staticmethod
    def number_input_question_generate(n=1):
        inputs = []
        outputs = []
        for i in range(n):
            input = '_'
            for j in range(i + 1):
                input += '{#_}_'
            inputs.append(input)
            outputs.append(json.dumps([0, 0, 0, 1, 0, 0, 0]))
        return inputs, outputs

    @staticmethod
    def matching_question_generate(n=1):
        inputs = []
        outputs = []
        iterable = [chr(i) for i in range(ord('а'), ord('я') + 1, 1)]
        permutations = itertools.permutations(iterable[:n], n)
        for permutation in permutations:
            input = '_'
            for j in range(len(permutation)):
                input += str(j + 1) + '-' + permutation[j] + (',' if j < len(permutation) - 1 else '')
            input += '_'
            inputs.append(input)
            outputs.append(json.dumps([0, 0, 0, 0, 1, 0, 0]))
        return inputs, outputs

    @staticmethod
    def ordering_question_generate(n=1):
        inputs = []
        outputs = []
        iterable = [str(i + 1) for i in range(n)]
        permutations = itertools.permutations(iterable, n)
        for permutation in permutations:
            input = '_'
            for j in range(len(permutation)):
                input += permutation[j] + (',' if j < len(permutation) - 1 else '')
            input += '_'
            inputs.append(input)
            outputs.append(json.dumps([0, 0, 0, 0, 0, 1, 0]))
        return inputs, outputs

    @staticmethod
    def composite_question_generate(n=1):
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
                outputs.append(json.dumps([0, 0, 0, 0, 0, 0, 1]))
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
            output_file.write(outputs[i] + '\n')
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
                output_file.write(outputs[i] + '\n')
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
                Datamaker.select_sample(r'\d', r'\d', masks[i], 'OQ', input_file, output_file)
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


if __name__ == "__main__":
    #Datamaker.make_train_dataset(6)
    #Datamaker.make_test_dataset('valid_input.txt')
    #Datamaker.make_test_dataset('test_input.txt')
    Datamaker.make_test_dataset('train_input.txt')
