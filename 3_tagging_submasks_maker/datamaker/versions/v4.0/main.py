import itertools
import json
import re


class Datamaker:

    def __init__(self):
        pass

    @staticmethod
    def create_groups(input, output, n):
        res = [[] for _ in range(n)]
        for i in range(len(output)):
            for j in range(len(output[i])):
                if output[i][j] != 0:
                    res[output[i][j] - 1].append(input[i])
        return res

    @staticmethod
    def single_choice_question_generate(n=1):
        inputs_1 = []
        outputs_1 = []
        inputs_2 = []
        outputs_2 = []
        for i in range(n):
            input_1 = ['<p>', '_', '</p>']
            output_1 = [[4], [1], [4]]
            for j in range(n):
                input_1 += ['<p>', str(j+1), '.', ('=' if i == j else '~') + '_', ';', '</p>']
                output_1 += [[4], [3], [3], [2, j + 5], [3], [4]]
            input_1 += ['<p>', '_', '</p>']
            output_1 += [[4], [1], [4]]
            inputs_1.append(input_1)
            outputs_1.append(output_1)
            groups = Datamaker.create_groups(input_1, output_1, n + 4)
            inputs_2.append([''.join(el) for el in groups])
            output_2 = []
            for k in range(n + 4):
                if k == 0:
                    output_2.append("[item]_[/item]")
                elif k == 1:
                    output_2.append("[singlechoice][/singlechoice]")
                elif k == 2:
                    output_2.append("numbering=123.;")
                elif k == 3:
                    output_2.append("view=vertical")
                else:
                    if groups[k][0] == "~_":
                        output_2.append("[choice]_[/choice]")
                    else:
                        output_2.append("[choice value=1]_[/choice]")
            outputs_2.append(output_2)
        return inputs_1, outputs_1, inputs_2, outputs_2

    @staticmethod
    def multi_choice_question_generate(n=1):
        inputs_1 = []
        outputs_1 = []
        inputs_2 = []
        outputs_2 = []
        products = list(itertools.product(["~_", "=_"], repeat=n))
        for product in products:
            input_1 = ['<p>', '_', '</p>']
            output_1 = [[0], [1], [0]]
            pos_groups = [2]
            neg_groups = [2]
            for i in range(len(product)):
                if product[i] == "=_":
                    pos_groups.append(5 + i * 2)
                else:
                    neg_groups.append(5 + i * 2)
            pos = round((1.0 / (len(pos_groups) - 1)) * 100) if (len(pos_groups) - 1) != 0 else 0
            neg = round((1.0 / (len(neg_groups) - 1)) * 100) if (len(neg_groups) - 1) != 0 else 0
            for j in range(len(product)):
                input_1 += ['<p>', str(j + 1), '.', product[j], ';', '</p>']
                if product[j] == "=_":
                    groups = pos_groups[:] + [4 + j * 2]
                else:
                    groups = neg_groups[:] + [4 + j * 2]
                output_1 += [[0], [3], [3], groups, [3], [0]]
            input_1 += ['<p>', '_', '</p>']
            output_1 += [[0], [1], [0]]
            if len(re.findall(r'=', ''.join(input_1))) > 1:
                inputs_1.append(input_1)
                outputs_1.append(output_1)
                groups = Datamaker.create_groups(input_1, output_1, 2 * len(product) + 3)
                inputs_2.append([''.join(el) for el in groups])
                output_2 = []
                for k in range(2 * len(product) + 3):
                    if k == 0:
                        output_2.append("[item]_[/item]")
                    elif k == 1:
                        output_2.append("[multichoice][/multichoice]")
                    elif k == 2:
                        output_2.append("numbering=123.;")
                    else:
                        if k % 2 != 0:
                            output_2.append("[choice]_[/choice]")
                        else:
                            if groups[k][0] == "~_":
                                output_2.append("value=-" + str(neg) + "%")
                            else:
                                output_2.append("value=" + str(pos) + "%")
                outputs_2.append(output_2)
        return inputs_1, outputs_1, inputs_2, outputs_2

    @staticmethod
    def text_input_question_generate(n=1):
        inputs_1 = []
        outputs_1 = []
        inputs_2 = []
        outputs_2 = []
        for i in range(n):
            input_1 = ['<p>', '_', '</p>']
            output_1 = [[0], [1], [0]]
            for j in range(i + 1):
                input_1 += ['<p>', '{=_}', '</p>']
                output_1 += [[0], [j + 2], [0]]
            input_1 += ['<p>', '_', '</p>']
            output_1 += [[0], [1], [0]]
            inputs_1.append(input_1)
            outputs_1.append(output_1)
            groups = Datamaker.create_groups(input_1, output_1, i + 2)
            inputs_2.append([''.join(el) for el in groups])
            output_2 = []
            for k in range(i + 2):
                if k == 0:
                    output_2.append("[item]_[/item]")
                else:
                    output_2.append("[textinput][answer]_[/answer][/textinput]")
            outputs_2.append(output_2)
        return inputs_1, outputs_1, inputs_2, outputs_2

    @staticmethod
    def number_input_question_generate(n=1):
        inputs_1 = []
        outputs_1 = []
        inputs_2 = []
        outputs_2 = []
        for i in range(n):
            input_1 = ['<p>', '_', '</p>']
            output_1 = [[0], [1], [0]]
            for j in range(i + 1):
                input_1 += ['<p>', '{#_}', '</p>']
                output_1 += [[0], [j + 2], [0]]
            input_1 += ['<p>', '_', '</p>']
            output_1 += [[0], [1], [0]]
            inputs_1.append(input_1)
            outputs_1.append(output_1)
            groups = Datamaker.create_groups(input_1, output_1, i + 2)
            inputs_2.append([''.join(el) for el in groups])
            output_2 = []
            for k in range(i + 2):
                if k == 0:
                    output_2.append("[item]_[/item]")
                else:
                    output_2.append("[numberinput][answer]_[/answer][/numberinput]")
            outputs_2.append(output_2)
        return inputs_1, outputs_1, inputs_2, outputs_2

    @staticmethod
    def matching_question_generate(n=1):
        inputs_1 = []
        outputs_1 = []
        inputs_2 = []
        outputs_2 = []
        iterable = [chr(i) for i in range(ord('а'), ord('я') + 1, 1)]
        permutations = itertools.permutations(iterable[:n], n)
        for permutation in permutations:
            input_1 = ['<p>', '_', '</p>']
            output_1 = [[0], [1], [0]]
            for j in range(len(permutation)):
                input_1 += ['<p>', str(j + 1), ')', '_', '</p>']
                output_1 += [[0], [3], [3], [3], [0]]
            for j in range(len(permutation)):
                input_1 += ['<p>', iterable[j], ')', '_', '</p>']
                output_1 += [[0], [0], [0], [0], [0]]
            input_1 += ['<p>', 'Ответ:']
            output_1 += [[0], [0]]
            for j in range(len(permutation)):
                input_1 += [str(j + 1) + '-' + permutation[j] + (',' if j < len(permutation) - 1 else '')]
                output_1 += [[2, j + 4]]
            input_1 += ['</p>', '<p>', '_', '</p>']
            output_1 += [[0], [0], [1], [0]]
            inputs_1.append(input_1)
            outputs_1.append(output_1)
            groups = Datamaker.create_groups(input_1, output_1, len(permutation) + 3)
            inputs_2.append([''.join(el) for el in groups])
            output_2 = []
            for k in range(len(permutation) + 3):
                if k == 0:
                    output_2.append("[item]_[/item]")
                elif k == 1:
                    output_2.append("[matching]_[/matching]")
                elif k == 2:
                    output_2.append("numbering=123)")
                else:
                    output_2.append("[subitem]" + str(k - 2) + "[answer]" + permutation[k - 3] + "[/answer][/subitem]")
            outputs_2.append(output_2)
        return inputs_1, outputs_1, inputs_2, outputs_2

    @staticmethod
    def ordering_question_generate(n=1):
        inputs_1 = []
        outputs_1 = []
        inputs_2 = []
        outputs_2 = []
        iterable = [str(i + 1) for i in range(n)]
        permutations = itertools.permutations(iterable, n)
        for permutation in permutations:
            input_1 = ['<p>', '_', '</p>']
            output_1 = [[0], [1], [0]]
            for j in range(len(permutation)):
                input_1 += ['<p>', str(j + 1), ')', '_', '</p>']
                output_1 += [[0], [3], [3], [3], [0]]
            input_1 += ['<p>', 'Ответ:']
            output_1 += [[0], [0]]
            for j in range(len(permutation)):
                input_1 += [permutation[j] + (',' if j < len(permutation) - 1 else '')]
                output_1 += [[2, j + 4]]
            input_1 += ['</p>', '<p>', '_', '</p>']
            output_1 += [[0], [0], [1], [0]]
            inputs_1.append(input_1)
            outputs_1.append(output_1)
            groups = Datamaker.create_groups(input_1, output_1, len(permutation) + 3)
            inputs_2.append([''.join(el) for el in groups])
            output_2 = []
            for k in range(len(permutation) + 3):
                if k == 0:
                    output_2.append("[item]_[/item]")
                elif k == 1:
                    output_2.append("[ordering]_[/ordering]")
                elif k == 2:
                    output_2.append("numbering=123)")
                else:
                    output_2.append("[subitem]" + permutation[k - 3] + "[/subitem]")
            outputs_2.append(output_2)
        return inputs_1, outputs_1, inputs_2, outputs_2

    @staticmethod
    def composite_question_generate(n=1, is_cloze=True):
        inputs_1 = []
        outputs_1 = []
        inputs_2 = []
        outputs_2 = []
        scq_inputs_1, scq_outputs_1, scq_inputs_2, scq_outputs_2 = Datamaker.single_choice_question_generate(n, False)
        tiq_inputs_1, tiq_outputs_1, scq_inputs_2, scq_outputs_2 = Datamaker.text_input_question_generate(n, False)
        niq_inputs_1, niq_outputs_1, scq_inputs_2, scq_outputs_2 = Datamaker.number_input_question_generate(n, False)
        products = Datamaker.generate_products(scq_inputs_1, tiq_inputs_1)
        products += Datamaker.generate_products(scq_inputs_1, niq_inputs_1)
        products += Datamaker.generate_products(tiq_inputs_1, niq_inputs_1)
        products += Datamaker.generate_products(scq_inputs_1, tiq_inputs_1, niq_inputs_1)
        for product in products:
            for tuple in product:
                input = ""
                for element in tuple:
                    input += element
                inputs_1.append(re.sub(r'_+', '_', input))
        products = Datamaker.generate_products(scq_outputs_1, tiq_outputs_1)
        products += Datamaker.generate_products(scq_outputs_1, niq_outputs_1)
        products += Datamaker.generate_products(tiq_outputs_1, niq_outputs_1)
        products += Datamaker.generate_products(scq_outputs_1, tiq_outputs_1, niq_outputs_1)
        for product in products:
            for tuple in product:
                output = ("[cloze]_" if is_cloze else "")
                for element in tuple:
                    output += element
                outputs_1.append(re.sub(r'_+', '_', output) + ("[/cloze]" if is_cloze else ""))
        return inputs_1, outputs_1, inputs_2, outputs_2

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
    def select_sample(lhs_pattern, rhs_pattern, mask, question_type, input_file, output_file):
        lhs = re.findall(lhs_pattern, mask)
        if question_type == 'SCQ':
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.single_choice_question_generate(len(lhs))
        elif question_type == 'MCQ':
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.multi_choice_question_generate(len(lhs))
        elif question_type == 'TIQ':
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.text_input_question_generate(len(lhs))
        elif question_type == 'NIQ':
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.number_input_question_generate(len(lhs))
        elif question_type == 'MQ':
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.matching_question_generate(len(lhs))
        elif question_type == 'OQ':
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.ordering_question_generate(len(lhs))
        elif question_type == 'CQ':
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.composite_question_generate(len(lhs))
        else:
            print("Unknown question type. There are 7 question types: SCQ, MCQ, TIQ, NIQ, MQ, OQ, CQ.")
            return
        question_numbers = {'SCQ': '1', 'MCQ': '2', 'TIQ': '3', 'NIQ': '4', 'MQ': '5', 'OQ': '6', 'CQ': '7'}
        for i in range(len(inputs_1)):
            rhs = re.findall(rhs_pattern, str(inputs_1[i]))
            if lhs == rhs:
                k = -1
                for j in range(len(inputs_2)):
                    if inputs_2[j] == ['_', '_']:
                        k += 1
                    if k == i + 1:
                        break
                    if k == i:
                        input_file.write('Вопрос ' + question_numbers[question_type] + '.\n')
                        input_file.write(json.dumps(inputs_2[j], ensure_ascii=False) + '\n')
                        output_file.write('Вопрос ' + question_numbers[question_type] + '.\n')
                        output_file.write(json.dumps(outputs_2[j], ensure_ascii=False) + '\n')

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
                Datamaker.select_sample(r'\d', r'\d(?!\', \'\))', masks[i], 'OQ', input_file, output_file)
                pass
        input_file.close()
        output_file.close()

    @staticmethod
    def make_train_dataset(n=1):
        Datamaker.clean_files()
        for i in range(n):
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.single_choice_question_generate(i + 1)
            Datamaker.write_to_files(inputs_2, outputs_2, 'Вопрос 1.')
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.multi_choice_question_generate(i + 1)
            Datamaker.write_to_files(inputs_2, outputs_2, 'Вопрос 2.')
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.text_input_question_generate(i + 1)
            Datamaker.write_to_files(inputs_2, outputs_2, 'Вопрос 3.')
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.number_input_question_generate(i + 1)
            Datamaker.write_to_files(inputs_2, outputs_2, 'Вопрос 4.')
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.matching_question_generate(i + 1)
            Datamaker.write_to_files(inputs_2, outputs_2, 'Вопрос 5.')
            inputs_1, outputs_1, inputs_2, outputs_2 = Datamaker.ordering_question_generate(i + 1)
            Datamaker.write_to_files(inputs_2, outputs_2, 'Вопрос 6.')
            # inputs, outputs = Datamaker.composite_question_generate(i+1)
            # Datamaker.write_to_files(inputs, outputs, 'Вопрос 7.')


if __name__ == "__main__":
    #Datamaker.make_train_dataset(6)
    Datamaker.make_test_dataset('valid_input.txt')
    #Datamaker.make_test_dataset('test_input.txt')
