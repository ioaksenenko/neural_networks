import re
import os
import json

from lxml import etree


class HTMLParser:

    def __init__(self):
        self.parser = None
        self.tree = None
        self.root = None
        self.parse_data = []
        self.text_set = []

    def load(self, file_path):
        self.parser = etree.HTMLParser()
        self.tree = etree.parse(file_path, self.parser)
        self.root = self.tree.getroot()

    def parse(self):
        if self.root is not None:
            annotations = self.root.cssselect("math annotation")
            for annotation in annotations:
                annotation.getparent().remove(annotation)
            body_childs = self.root.cssselect("html>body>div")[0].getchildren()
            for body_child in body_childs:
                content = str(etree.tounicode(body_child, method="html"))
                content = re.sub(r'[\r\n]', ' ', content)
                content = re.sub(r'\s*class=\"[a-zA-Z0-9_-]+\"|\s+(?=[<>\"\':;])', '', content)
                content = re.sub(
                    r'\s*[a-zA-Z_-]+=(\'[a-zA-Z\s\d.,_\-;=:#\"/&?%]+\'|\"[a-zA-Z\s\d.,_\-;=:#\'/&?%]+\")\s*', '',
                    content
                )
                content = re.sub(r'<p><span></span></p>', '', content)
                content = re.sub(r'<span>', '', content)
                content = re.sub(r'</span>', '', content)
                content = re.sub(r'<p></p>', '', content)
                content = re.sub(r'<p><b></b></p>', '', content)
                content = re.sub(r'<a></a>', '', content)
                content = re.sub(r'<b><i></i></b>', '', content)
                content = re.sub(r'<p><b><i></i></b></p>', '', content)
                content = re.sub(r'<b></b>', '', content)
                content = re.sub(r'<p><i></i></p>', '', content)
                content = re.sub(r'<math><semantics><mo>–</mo></semantics></math>', '–', content)
                content = re.sub(r'<math><semantics><mo>—</mo></semantics></math', '—', content)
                content = re.sub(r'<i></i>', '', content)
                content = re.sub(r'<p><a><b><u></u></b></a></p>', '', content)
                content = re.sub(r'<p><b><u></u></b></p>', '', content)
                content = re.sub(r'<h3></h3>', '', content)
                content = re.sub(r'<p></p>', '', content)
                content = re.sub(r'\s+', ' ', content)
                self.parse_data.append(content)

    def parse_all_files(self, path_to_input_files="input\\"):
        files = os.listdir(path_to_input_files)
        for file in files:
            self.load(os.path.join(path_to_input_files, file))
            self.parse()

    def create_text_set(self):
        self.text_set = []
        for data in self.parse_data:
            self.text_set.append(data)

    @staticmethod
    def write_to_json_file(path_to_json_file, text):
        file = open(path_to_json_file, 'w', encoding="utf-8")
        file.write(json.dumps(text, ensure_ascii=False))
        file.close()

    @staticmethod
    def write_to_text_file(path_to_text_file, text):
        file = open(path_to_text_file, 'w', encoding="utf-8")
        file.write(text)
        file.close()


if __name__ == "__main__":
    parser = HTMLParser()
    parser.parse_all_files()
    parser.create_text_set()
    file = open("output\\output.txt", 'w', encoding="utf-8")
    for text in parser.text_set:
        file.write(text)
    file.close()
