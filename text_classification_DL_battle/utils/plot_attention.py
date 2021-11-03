"""
Code here based on: https://gist.github.com/ihsgnef/f13c35cd46624c8f458a4d23589ac768
"""

import matplotlib


def plot_word_attention(doc, doc_w, cmap="Greens"):
    r"""
    Helper function to color text with the aim of visualizing attention

    Parameters:
    ----------
    doc: List
        List of str containing the sentences per review
    doc_w: np.ndarray
        np array with floats (between 0, 1) that are the attention weights per word.
        They have the same length as s.split() where s is each element in doc

    Returns:
    --------
    colored_doc: str
        str containing the html colored review
    """
    cmap = matplotlib.cm.get_cmap(cmap)
    template = '<font face="monospace" \nsize="3"; span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_doc = ""
    for sent, sent_w in zip(doc, doc_w):
        sent_len, pad_count = len(sent.split()), 0
        for t, w in zip(sent.split(), sent_w):
            if t == "xxpad":
                pad_count += 1
                continue
            color = matplotlib.colors.rgb2hex(cmap(w)[:3])
            colored_doc += template.format(color, "&nbsp" + t + "&nbsp")
        if pad_count != sent_len:
            colored_doc += "</br>"
    return colored_doc


def plot_sent_attention(doc, doc_w, cmap="Greens"):
    r"""
    Helper function to color text with the aim of visualizing attention

    Parameters:
    ----------
    doc: List
        List of str containing the sentences per review
    doc_w: np.ndarray
        np array with floats (between 0, 1) that are the attention weights per sentence.
        They have the same length as doc.

    Returns:
    --------
    colored_doc: str
        str containing the html colored review
    """
    cmap = matplotlib.cm.get_cmap(cmap)
    template = '<font face="monospace" \nsize="3"; span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_doc = ""

    for sent, sent_w in zip(doc, doc_w):
        sent = " ".join([t for t in sent.split() if t != "xxpad"])
        if len(sent) > 0:
            color = matplotlib.colors.rgb2hex(cmap(sent_w)[:3])
            colored_doc += template.format(color, "&nbsp" + sent + "&nbsp") + "</br>"
    return colored_doc
