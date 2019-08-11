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
            input = "_"
            output = ("[item]" if is_item else "") + "_ "
            for j in range(n):
                if i == j:
                    input += "=_"
                    output += (". [singlechoice][choice value=1]_[/choice]" if j == 0 else
                               " . [choice value=1]_[/choice]")
                else:
                    input += "~_"
                    output += ((" " if j != 0 else "") + ". [singlechoice][choice]_[/choice]" if j == 0 else
                               " . [choice]_[/choice]")
            output += "[/singlechoice]" + ("[/item]" if is_item else "")
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def multi_choice_question_generate(n=1, is_item=True):
        inputs = []
        outputs = []
        tuples = list(itertools.product(['~_', '=_'], repeat=n))
        for tuple in tuples:
            input = "_"
            output = ("[item]" if is_item else "") + "_ "
            pos = 0
            neg = 0
            for element in tuple:
                if element == '=_':
                    pos = pos + 1
                else:
                    neg = neg + 1
            pos = round((1.0 / pos) * 100) if pos != 0 else 0
            neg = round((1.0 / neg) * 100) if neg != 0 else 0
            for i in range(len(tuple)):
                input += tuple[i]
                if tuple[i] == '=_':
                    output += (". [multichoice][choice value=" + str(pos) + "%]_[/choice]" if
                               i == 0 else " . [choice value=" + str(pos) + "%]_[/choice]")
                else:
                    output += ((" " if i != 0 else "") + ". [multichoice][choice value=-" + str(neg) + "%]_[/choice]" if
                               i == 0 else " . [choice value=-" + str(neg) + "%]_[/choice]")
            output += "[/multichoice]" + ("[/item]" if is_item else "")
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def text_input_question_generate(n=1, is_item=True):
        inputs = []
        outputs = []
        for i in range(n):
            input = "_"
            output = ("[item]" if is_item else "") + "_ "
            for j in range(i+1):
                input += '{=_}_'
                output += (". . " + ("[textinput]" if j == 0 else "") +
                           "[answer]_[/answer] . " + ("_ " if j != i else ""))
            output += "_[/textinput]" + ("[/item]" if is_item else "")
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    @staticmethod
    def number_input_question_generate(n=1, is_item=True):
        inputs = []
        outputs = []
        for i in range(n):
            input = "_"
            output = ("[item]" if is_item else "") + "_ "
            for j in range(i + 1):
                input += '{#_}_'
                output += (". . " + ("[numberinput]" if j == 0 else "") +
                           "[answer]_[/answer] . " + ("_ " if j != i else ""))
            output += "_[/numberinput]" + ("[/item]" if is_item else "")
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
            input = "_"
            output = ("[item]" if is_item else "") + "_ "
            for i in range(len(permutation)):
                input += str(i + 1) + '-' + permutation[i] + (',' if i < len(permutation) - 1 else '_')
                output += (("[matching][subitem]" if i == 0 else " [subitem]") + str(i + 1) +
                           " . [answer]" + permutation[i] + "[/answer][/subitem]" +
                           (" ." if i < len(permutation) - 1 else ""))
            output += " _[/matching]" + ("[/item]" if is_item else "")
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
            input = "_"
            output = ("[item]" if is_item else "") + "_ "
            for i in range(len(permutation)):
                input += permutation[i] + (',' if i < len(permutation) - 1 else '_')
                output += (('[ordering][subitem]' if i == 0 else ' [subitem]') + permutation[i] +
                           '[/subitem]' + (' .' if i < len(permutation) - 1 else ''))
            output += " _[/ordering]" + ("[/item]" if is_item else "")
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
        return list(set(inputs)), list(set(outputs))

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
            input_file.write(inputs[i] + '\n')
            output_file.write(label + '\n')
            output_file.write(outputs[i] + '\n')
        input_file.close()
        output_file.close()


if __name__ == "__main__":
    for i in range(5):
        inputs, outputs = Datamaker.single_choice_question_generate(i+1)
        Datamaker.write_to_files(inputs, outputs, 'Вопрос 1.')
        inputs, outputs = Datamaker.multi_choice_question_generate(i+1)
        Datamaker.write_to_files(inputs, outputs, 'Вопрос 2.')
        inputs, outputs = Datamaker.text_input_question_generate(i+1)
        Datamaker.write_to_files(inputs, outputs, 'Вопрос 3.')
        inputs, outputs = Datamaker.number_input_question_generate(i+1)
        Datamaker.write_to_files(inputs, outputs, 'Вопрос 4.')
        inputs, outputs = Datamaker.matching_question_generate(i+1)
        Datamaker.write_to_files(inputs, outputs, 'Вопрос 5.')
        inputs, outputs = Datamaker.ordering_question_generate(i+1)
        Datamaker.write_to_files(inputs, outputs, 'Вопрос 6.')
        #inputs, outputs = Datamaker.composite_question_generate(i+1)
        #Datamaker.write_to_files(inputs, outputs, 'Вопрос 7.')