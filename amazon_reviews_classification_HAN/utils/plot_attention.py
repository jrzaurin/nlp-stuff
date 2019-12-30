"""
Code here based on: https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768
"""

import matplotlib


def plot_word_attention(doc, doc_w, cmap="Greens"):

    cmap = matplotlib.cm.get_cmap(cmap)
    template = '<font face="monospace" \nsize="3"; span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_doc = ""

    for sent, sent_w in zip(doc, doc_w):
        for t, w in zip(sent.split(), sent_w):
            color = matplotlib.colors.rgb2hex(cmap(w)[:3])
            colored_doc += template.format(color, "&nbsp" + t + "&nbsp")
        colored_doc += "</br>"

    return colored_doc


def plot_sent_attention(doc, doc_w, cmap="Greens"):

    cmap = matplotlib.cm.get_cmap(cmap)
    template = '<font face="monospace" \nsize="3"; span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_doc = ""

    for sent, sent_w in zip(doc, doc_w):
        color = matplotlib.colors.rgb2hex(cmap(sent_w)[:3])
        colored_doc += template.format(color, "&nbsp" + sent + "&nbsp") + "</br>"

    return colored_doc
