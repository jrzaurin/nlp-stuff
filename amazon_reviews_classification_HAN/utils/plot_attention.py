import matplotlib
import matplotlib.pyplot as plt


def attention_to_html(text, attnw, out_path, cmap="Greens"):
    cmap = matplotlib.cm.get_cmap(cmap)
    html_chars = []
    for t, w in zip(text, attnw):
        r, g, b, a = cmap(w)
        r, g, b = int(256 * r), int(256 * g), int(256 * b)
        t = '<font face="monospace" \nsize="3">%s</font>' % t
        html_chars.append(
            '<span style="background-color:rgb(%s, %s, %s); color:black;">%s</span>' % (r, g, b, t)
        )
    tot_html = " ".join(html_chars)
    with open(out_path, "w") as out:
        out.write(tot_html)
