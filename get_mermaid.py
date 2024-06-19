import random
import base64
import requests, io, os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random_word import RandomWords
from tqdm import tqdm, trange
from time import sleep
import string

# from IPython.display import SVG

STYLE = {
    "orientation": ["TB", "BT", "RL", "LR", "TD"],
    "node": {
        "shape":
        # [ ['(', ')'], ['([', '])'], ['[[',']]'], ['[(', ')]'], ['((', '))'], ['>', ']'], ['{','}'], ['{{','}}'], ['[/','/]'],
        # ['[\\','/]'],  ['[\\','\\]'],  ['[/','\\]'],['(((',')))'], ]
        [
            ["(", ")"],
        ]
    },
    "arrow": {
        "head": ["<", ">"],  # ['<', 'x', 'o', '>'],
        "line": [
            "----",
        ],
    },  #  '====', '-..-',
    "message": {
        "head_right": [">", ">>", "x", ")"]
    },  # https://mermaid.js.org/syntax/sequenceDiagram.html
}
OUTPUT_FOLDER = "mermaid_easy_v2"
HAVE_MESSAGE = False
HAVE_TITLE = False
USE_ALPHA = False
FORCE_ARROW = True
RANDOM_DIRECTION = True

ALL_ALPHA = [c for c in string.ascii_uppercase]


def random_colour():
    defaults = ["FFFFFF", "C0C0C0", ""]  # white, gray, random
    randomColour = random.choice(defaults)
    if randomColour == "":
        hex_chars = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
        ]
        for i in range(0, 6):
            randomColour += random.choice(hex_chars)
    return randomColour


def random_str(length=None, max_length=10, max_char_length=20):
    if USE_ALPHA:
        return random_alphabet()
    if length is None:
        length = random.randint(1, max_length)
    out = ""
    for _ in range(length):
        # func = random.choice([RandomWords().get_random_word, RandomWords().get_random_word, names.get_first_name, names.get_last_name])
        func = RandomWords().get_random_word
        new_word = func()
        if len(out + new_word) > max_char_length:
            break
        out += new_word + " "
    return out[:-1]


def random_alphabet():
    alpha = random.choice(ALL_ALPHA)
    ALL_ALPHA.remove(alpha)
    # if len(ALL_ALPHA) == 0:
    #     n = len(alpha)
    #     ALL_ALPHA=[ A+a for A, a in zip(string.ascii_uppercase, string.ascii_uppercase[n:] + string.ascii_uppercase[:n])]
    return alpha


def myRand():
    return random.choice([True, False])


def get_random_connection():
    out = ""
    # left arrow head
    # if myRand():
    #     out += random.choice(STYLE['arrow']['head'][:-1])
    out += "-"  # '<---' is not valid. only '-->' or '<--->' is valid

    out += random.choice(STYLE["arrow"]["line"])

    # right arrow head
    if FORCE_ARROW:
        out += ">"
    else:
        if myRand():
            out += random.choice(STYLE["arrow"]["head"][1:])

    if HAVE_MESSAGE and random.random() < 0.3:
        out += " |" + random_str(max_length=10) + "|"
    else:
        out += " "
    return out


def get_random_link_heads(n_nodes):
    links = []

    for i in range(n_nodes):
        link = [0, 0]
        while link[0] == link[1]:
            link = [i, random.randint(0, n_nodes - 1)]
        if link[0] > link[1]:
            link = link[::-1]
        if link not in links:
            links.append(link)
    for _ in range(random.choice([0, 0, 1, 2, 3])):
        link = [0, 0]
        while link[0] == link[1]:
            link = [random.randint(0, n_nodes - 1), random.randint(0, n_nodes - 1)]
        if link[0] < link[1]:
            link = link[::-1]
        if link not in links:
            links.append(link)
    return sorted(links, key=lambda x: x[0])


def create_random_flowchart(min_nodes=5, max_nodes=10, prev="\n", have_subgraphs=False):
    output = ""
    if HAVE_TITLE and myRand():
        title = random_str(max_length=10)
        output += f"---\ntitle: {title}\n---\n"
    if RANDOM_DIRECTION:
        direct = random.choice(STYLE["orientation"])
    else:
        direct = "TD"
    output += f"flowchart {direct}"

    n_nodes = random.randint(min_nodes, max_nodes)

    for i in range(n_nodes):
        node_text = random_str(max_length=5)
        node_shape = random.choice(STYLE["node"]["shape"])
        output += prev + f"entity{i}" + node_shape[0] + node_text + node_shape[1]

    groups = []
    if have_subgraphs:
        with_subgraph = myRand()
    else:
        with_subgraph = False

    # links = []
    # n_links = random.randint(n_nodes-2, n_nodes+4)
    # for i in range(n_links):
    # start = random.randint(0, n_nodes-2)
    # link = [start, random.randint(start+1, n_nodes-1)]
    # link = [random.randint(0,n_nodes-1), random.randint(0, n_nodes-1)]
    # if link not in links and link[::-1] not in links:
    #   links.append(link)
    links = get_random_link_heads(n_nodes)
    for link in links:
        if with_subgraph:
            group = random_str(max_length=1)
            if group in groups:
                continue
            output += prev + "subgraph " + group
            output += prev + "direction " + random.choice(STYLE["orientation"][:-1])
            output += (
                prev
                + f"entity{link[0]} "
                + get_random_connection()
                + f"entity{link[1]}"
            )
            output += prev + "end"
            groups.append(group)
        else:
            output += (
                prev
                + f"entity{link[0]} "
                + get_random_connection()
                + f"entity{link[1]}"
            )

    if with_subgraph:
        for i in range(random.randint(0, len(groups) + 1)):
            link = [
                random.choice(groups),
                random.choice(
                    [f"entity{random.randint(0,n_nodes-1)}", random.choice(groups)]
                ),
            ]
            if link not in links:
                output += (
                    prev
                    + f"entity{link[0]} "
                    + get_random_connection()
                    + f"entity{link[1]}"
                )
            links.append(link)
    return output


def main():
    if not os.path.exists(f"./{OUTPUT_FOLDER}"):
        for subdir in ["txt", "bad_txt", "jpeg", "svg"]:
            os.makedirs(f"./{OUTPUT_FOLDER}/{subdir}")

    for file_idx in trange(3000):
        global ALL_ALPHA
        ALL_ALPHA = [c for c in string.ascii_uppercase]
        chart = create_random_flowchart()
        chart = create_random_flowchart(min_nodes=3, max_nodes=10)
        graphbytes = chart.__str__().encode("ascii")
        base64_bytes = base64.b64encode(graphbytes)
        base64_string = base64_bytes.decode("ascii")
        base64_string += "?bgColor=" + random_colour()
        # print('https://mermaid.ink/img/' + base64_string)

        # Save
        r = requests.get("https://mermaid.ink/img/" + base64_string)
        if r.status_code == 404:
            # bad chart or requests
            with open(f"./{OUTPUT_FOLDER}/bad_txt/{file_idx}.txt", "w") as f:
                f.write(chart.__str__())
            print(file_idx, "failed")
            continue

        else:
            with open(f"./{OUTPUT_FOLDER}/txt/{file_idx}.txt", "w") as f:
                f.write(chart.__str__())

        with open(f"./{OUTPUT_FOLDER}/jpeg/{file_idx}.jpeg", "wb") as f:
            f.write(r.content)

        r = requests.get("https://mermaid.ink/svg/" + base64_string)
        with open(f"./{OUTPUT_FOLDER}/svg/{file_idx}.svg", "wb") as f:
            f.write(r.content)

        # if file_idx %50 == 0 and file_idx >0:
        #     delay = random.randint(10, 100)
        #     print(f'{file_idx+1}/3000, sleep {delay}s')
        #     sleep(delay)
        # else:
        #     print(f'{file_idx+1}/3000')
        # break


if __name__ == "__main__":
    main()
