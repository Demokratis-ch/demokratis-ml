import re

def parse_markdown(markdown):
    lines = markdown.split('\n')
    document = {"label": "", "type": "document", "content": [], "children": []}
    stack = [document]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('#'):
            level = len(re.match(r'#+', line).group(0))
            content = line[level:].strip()
            node = {"label": str(level), "type": "heading", "content": [content], "children": []}
            stack[-1]["children"].append(node)
            stack.append(node)
        elif line.startswith('-'):
            match = re.match(r'- \[(.*?)\] (.*)', line)
            if match:
                label, content = match.groups()
                node = {"label": label, "type": "list_item", "content": [content], "children": []}
                if stack[-1]["type"] != "list":
                    list_node = {"label": "", "type": "list", "content": [], "children": []}
                    stack[-1]["children"].append(list_node)
                    stack.append(list_node)
                stack[-1]["children"].append(node)
                stack.append(node)
        else:
            node = {"label": "", "type": "content", "content": [line], "children": []}
            stack[-1]["children"].append(node)

    return document

# Example usage
markdown = """
# Art. 1 Sachüberschrift, Abs. 1 und 1bis
Gegenstand und Geltungsbereich

- [1.] Dieses Gesetz regelt:
  - [a.] die Veranstaltung, die Aufbereitung, die Übertragung und den Empfang von Radio- und Fernsehprogrammen;
  - [b.] die Fördermassnahmen zugunsten der elektronischen Medien.
"""

parsed = parse_markdown(markdown)
print(parsed)
