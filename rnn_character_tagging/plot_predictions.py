import fnmatch
from joblib import load
import matplotlib
import matplotlib.pyplot as plt
import os
from glob import glob
import argparse

def prediction_to_html(text, predictions, labels, cmap='Reds'):
    cmap = matplotlib.cm.get_cmap(cmap)
    html_chars = []
    for c, p, l in zip(text, predictions, labels):
        if c == '\n':
            html_chars.append('<br>')
        else:
            r, g, b, a = cmap(p)
            r, g, b = int(256*r), int(256*g), int(256*b)
            if l:
                c = '<font face="Times New Roman" \nsize="5">%s</font>' % c
            else:
                c = '<font face="monospace" \nsize="3">%s</font>' % c
            html_chars.append('<span style="background-color:rgb(%s, %s, %s); color:black;">%s</span>' % (r, g, b, c))
    tot_html = "".join(html_chars)
    return tot_html


def main(predictions_dir, output_dir, colormap='Reds'):
    try:
        os.makedirs(output_dir)
    except os.error:
        pass
    files = glob(os.path.join(predictions_dir, "*"))
    for i, f in enumerate(files[100:110]):
        text, prediction, labels = load(f)
        html = prediction_to_html(text, prediction, labels, cmap=colormap)
        out_path = os.path.join(output_dir, 'part-' + str(i).zfill(5) + ".html")
        with open(out_path, "w") as out:
            out.write(html)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("takes output of apply_tagger.py and plots in as html")
    parser.add_argument("input_dir", help="where to look for inputs (same as output directory of "
                                          "apply_tagger.py)")
    parser.add_argument("output_dir", help="where to put resulting html files")
    parser.add_argument("--colormap", default="Reds", help="name of the matplotlib colormap to use")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.colormap)
