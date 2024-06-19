"""
	1. Parse text from SVG and get a bounding box for each text
	   For each text, match the entity index in the mermaid code.
	2. Parse links from SVG and obtaining start and end point. 
        In addition, obtain bounding boxes center at each start/endpoint with a size of (bbox_l*2, bbox_l*2) .
        End point is always with arrow head.
	Example Output:
		{''text': {
'0': {'x0': 44.67578125, 'x1': 141.87890625, 'y0': 476.0, 'y1': 510.0, 'text': 'steatopygic', 'mermaid_entity_i': 0}
'1': {'x0': 92.78125, 'x1': 177.375, 'y0': 242.0, 'y1': 276.0, 'text': 'tartuffian', 'mermaid_entity_i': 1}
'2': {'x0': 38.48828125, 'x1': 148.06640625, 'y0': 710.0, 'y1': 744.0, 'text': 'apotheosized', 'mermaid_entity_i': 2}
'3': {'x0': 8.0, 'x1': 114.953125, 'y0': 8.0, 'y1': 42.0, 'text': 'assyria shirrs', 'mermaid_entity_i': 3}
},
'links': {
'1': {'start_point': (93.27734375, 476.0), 'end_point': (118.15876116071428, 276.0), 'start_text_i': '0', 'end_text_i': '1', 'link_bbox': [93.27734375, 118.15876116071428, 276.0, 476.0], 'start_point_bbox': [83.27734375, 466.0, 103.27734375, 486.0], 'end_point_bbox': [108.15876116071428, 266.0, 128.15876116071428, 286.0]}
'2': {'start_point': (105.28701636904762, 242.0), 'end_point': (61.4765625, 42.0), 'start_text_i': '1', 'end_text_i': '3', 'link_bbox': [61.47656249999995, 105.28701636904762, 42.0, 242.0], 'start_point_bbox': [95.28701636904762, 232.0, 115.28701636904762, 252.0], 'end_point_bbox': [51.4765625, 32.0, 71.4765625, 52.0]}
'3': {'start_point': (59.43861607142857, 710.0), 'end_point': (40.50957961309524, 42.0), 'start_text_i': '2', 'end_text_i': '3', 'link_bbox': [9.675781249999998, 59.43861607142857, 42.0, 710.0], 'start_point_bbox': [49.43861607142857, 700.0, 69.43861607142857, 720.0], 'end_point_bbox': [30.50957961309524, 32.0, 50.50957961309524, 52.0]}
'4': {'start_point': (93.27734375, 710.0), 'end_point': (93.27734375, 510.0), 'start_text_i': '2', 'end_text_i': '0', 'link_bbox': [93.27734375, 93.27734375, 510.0, 710.0], 'start_point_bbox': [83.27734375, 700.0, 103.27734375, 720.0], 'end_point_bbox': [83.27734375, 500.0, 103.27734375, 520.0]}
'5': {'start_point': (127.11607142857143, 710.0), 'end_point': (151.99748883928572, 276.0), 'start_text_i': '2', 'end_text_i': '1', 'link_bbox': [127.11607142857143, 176.87890625, 276.0, 710.0], 'start_point_bbox': [117.11607142857143, 700.0, 137.11607142857144, 720.0], 'end_point_bbox': [141.99748883928572, 266.0, 161.99748883928572, 286.0]}
'6': {'start_point': (95.31529017857143, 42.0), 'end_point': (139.12574404761904, 242.0), 'start_text_i': '3', 'end_text_i': '1', 'link_bbox': [95.31529017857143, 145.078125, 42.0, 242.0], 'start_point_bbox': [85.31529017857143, 32.0, 105.31529017857143, 52.0], 'end_point_bbox': [129.12574404761904, 232.0, 149.12574404761904, 252.0]}
},}
"""

from svgpathtools import svg2paths
from matplotlib import pyplot as plt
from svgelements import *
import difflib
import matplotlib.patches as patches

import os
import numpy as np
import math
import copy
import json
from tqdm import tqdm
from xml.dom import minidom

root = "./mermaid_easy_v2"
bbox_l = 10


def get_mermaid_entity_map(mermaid_file):
    entity_map = {}
    # Parse mermaid and get entity map
    mermaid = open(mermaid_file, "r").readlines()
    mermaid = [l for l in mermaid if "entity" in l and "--" not in l]
    for l in mermaid:
        i = l.split("(")[0].replace("entity", "")
        name = l.split("(")[1].split(")")[0]
        entity_map[name] = int(i)
    return entity_map


DEBUG = False
pbar = tqdm(os.listdir(f"{root}/jpeg/")[5389:])
for file in pbar:
    if file.endswith(".jpeg") == False:
        continue
    file_idx = file.split(".")[0]
    pbar.set_description(f"{file_idx}")
    svg_file = f"{root}/svg/{file_idx}.svg"
    img_file = f"{root}/jpeg/{file_idx}.jpeg"
    mermaid_file = f"{root}/txt/{file_idx}.txt"
    parsed = {"text": {}, "links": {}}

    image = plt.imread(img_file)
    fig, ax = plt.subplots()
    if DEBUG:
        plt.imshow(image)

    mermaid_entity_map = get_mermaid_entity_map(mermaid_file)

    # Get SVG offsets
    paths, attributes = svg2paths(svg_file)
    svg = SVG.parse(svg_file)
    offset = {"x": svg.viewbox.x, "y": svg.viewbox.y}

    # Parse text
    doc = minidom.parse(svg_file)
    nodes = [
        elem
        for elem in doc.getElementsByTagName("g")
        if elem.getAttribute("class") == "node default default flowchart-label"
    ]
    for node in nodes:
        svg_i = node.getAttribute("id").split("-")[-1]
        # transform = node.getElementsByTagName('g')[0].getAttribute('transform')
        # x0, y0 = [float(x) for x in transform.split('(')[1][:-1].split(',')]
        transform = node.getAttribute("transform")
        x0, y0 = [float(x) for x in transform.split("(")[1][:-1].split(",")]
        for rect in node.getElementsByTagName("rect"):
            if rect.getAttribute("class") == "basic label-container":
                x = float(rect.getAttribute("x"))
                y = float(rect.getAttribute("y"))
                width = float(rect.getAttribute("width"))
                height = float(rect.getAttribute("height"))
        x0 += x - offset["x"]
        y0 += y - offset["y"]
        x1, y1 = x0 + width, y0 + height
        out = {
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "text": node.getElementsByTagName("span")[0].firstChild.nodeValue,
        }
        out["mermaid_entity_i"] = mermaid_entity_map[out["text"]]
        parsed["text"][svg_i] = copy.deepcopy(out)
        if DEBUG:
            plt.plot((x0, x1), (y0, y1), "x", color="b")

    # Parse links
    link_i = 0
    colors = [
        "b",
        "g",
        "r",
        "c",
        "m",
        "y",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    for path, attribute in zip(paths, attributes):
        if "marker-end" in attribute:  # and attribute['class'] == 'arrowMarkerPath':
            tmp = attribute["class"].split(" ")[-2:]
            tmp = [i.split("-entity")[-1] for i in tmp]

            lbbox = list(path.bbox())  # xmin, xmax, ymin, ymax
            lbbox[0] -= offset["x"]
            lbbox[1] -= offset["x"]
            lbbox[2] -= offset["y"]
            lbbox[3] -= offset["y"]
            rect = patches.Rectangle(
                [lbbox[0], lbbox[2]],
                lbbox[1] - lbbox[0],
                lbbox[3] - lbbox[2],
                linewidth=1,
                edgecolor=colors[link_i],
                facecolor="none",
            )
            if DEBUG:
                ax.add_patch(rect)

            link = {
                "start_point": (
                    path.start.real - offset["x"],
                    path.start.imag - offset["y"],
                ),
                "end_point": (path.end.real - offset["x"], path.end.imag - offset["y"]),
                "start_text_i": tmp[0],
                "end_text_i": tmp[1],
                "link_bbox": [lbbox[0], lbbox[2], lbbox[1], lbbox[3]],
                # 'start_text_mermaid_i':parsed['text'][tmp[0]]['mermaid_i'],
                # 'end_text_mermaid_i': parsed['text'][tmp[1]]['mermaid_i'],
            }
            for pref in ["start", "end"]:
                # bbox is fixed size center at the intersection point between text node and link
                link[f"{pref}_point_bbox"] = [
                    link[f"{pref}_point"][0] - bbox_l,
                    link[f"{pref}_point"][1] - bbox_l,
                    link[f"{pref}_point"][0] + bbox_l,
                    link[f"{pref}_point"][1] + bbox_l,
                ]

                if DEBUG:
                    rect = patches.Rectangle(
                        link[f"{pref}_point_bbox"][:2],
                        bbox_l * 2,
                        bbox_l * 2,
                        linewidth=1,
                        edgecolor=colors[link_i],
                        facecolor="none",
                    )
                    ax.add_patch(rect)
            link_i += 1
            parsed["links"][link_i] = link

    if DEBUG:
        for k in parsed:
            print("'" + k + "': {")
            for j in parsed[k]:
                print(f"'{j}':", parsed[k][j])
            print("},")

    # Convert the dictionary to a JSON string with indentation
    json_string = json.dumps(parsed, indent=4)

    # Write the JSON string to a file
    with open(f"{root}/parsed_v3/{file_idx}.json", "w") as f:
        f.write(json_string)
