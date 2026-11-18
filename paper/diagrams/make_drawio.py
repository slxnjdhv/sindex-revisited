"""Write the paper's method diagrams as editable draw.io files.

    .venv/bin/python paper/diagrams/make_drawio.py
    drawio -x -f pdf --crop -o paper/figures/pipeline.pdf paper/diagrams/pipeline.drawio

Colours: blue = this paper's additions, grey = unchanged SINdex steps,
orange = the original evaluation protocol we audit.
"""
from __future__ import annotations

import html
from pathlib import Path

HERE = Path(__file__).resolve().parent

BLUE = "fillColor=#cde2fb;strokeColor=#2a78d6;"
GREY = "fillColor=#f5f5f3;strokeColor=#52514e;"
ORANGE = "fillColor=#fbe1d6;strokeColor=#eb6834;"
AQUA = "fillColor=#d3f1e6;strokeColor=#1baf7a;"
BOX = "rounded=1;whiteSpace=wrap;html=1;arcSize=12;fontFamily=Helvetica;fontSize=12;fontColor=#0b0b0b;strokeWidth=1.5;"
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;html=1;endArrow=block;endFill=1;"
        "strokeColor=#52514e;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11;fontColor=#52514e;")


class Diagram:
    def __init__(self, name: str):
        self.name, self.cells, self.n = name, [], 2

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def box(self, x, y, w, h, text, style=GREY, extra=""):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{html.escape(text.replace(chr(10), "<br>"))}" style="{BOX}{style}{extra}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def diamond(self, x, y, w, h, text, style=GREY):
        return self.box(x, y, w, h, text, style, "rhombus;rounded=0;")

    def text(self, x, y, w, h, text, extra="fontSize=11;fontColor=#52514e;align=left;"):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{html.escape(text.replace(chr(10), "<br>"))}" style="text;html=1;whiteSpace=wrap;fontFamily=Helvetica;{extra}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def edge(self, a, b, label="", extra=""):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{html.escape(label)}" style="{EDGE}{extra}" edge="1" parent="1" source="{a}" target="{b}">'
                          f'<mxGeometry relative="1" as="geometry"/></mxCell>')
        return i

    def frame(self, x, y, w, h, title):
        i = self._id()
        self.cells.append(f'<mxCell id="{i}" value="{html.escape(title)}" style="rounded=1;whiteSpace=wrap;html=1;arcSize=4;dashed=1;'
                          f'fillColor=none;strokeColor=#9a9993;fontFamily=Helvetica;fontSize=11;fontColor=#52514e;verticalAlign=top;align=left;spacingLeft=8;" '
                          f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def save(self):
        body = "".join(self.cells)
        xml = (f'<mxfile host="drawio"><diagram id="{self.name}" name="{self.name}"><mxGraphModel dx="1200" dy="600" grid="1" gridSize="10" '
               f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" background="#ffffff" math="0" shadow="0">'
               f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>')
        (HERE / f"{self.name}.drawio").write_text(xml)


def pipeline():
    d = Diagram("pipeline")
    q = d.box(0, 0, 100, 56, "Question <i>q</i>")
    s = d.box(130, 0, 130, 56, "LLM sampled<br><i>P</i> = 10 times (<i>T</i> = 1.0)")
    e = d.box(290, 0, 170, 56, "Sentence encoder (MiniLM)<br>on <i>q</i> [SEP] <i>g<sub>i</sub></i>")
    w = d.box(490, 0, 140, 56, "Cosine similarity<br>matrix <i>W</i>")
    dia = d.diamond(490, 100, 140, 90, "0.85 &le; <i>W<sub>ij</sub></i> &lt; 1 ?")
    nli = d.box(670, 115, 150, 60, "NLI cross-encoder<br>(both directions)", BLUE)
    veto = d.box(670, 225, 150, 60, "P(contradiction) &gt; 0.5<br>&rArr; set <i>W<sub>ij</sub></i> = &minus;1", BLUE)
    hac = d.box(270, 117, 180, 56, "Agglomerative clustering<br>average link, cut 0.05")
    cl = d.box(0, 117, 230, 56, "Clusters <i>C<sub>k</sub></i>: shares <i>p<sub>k</sub></i>,<br>cohesion <i>c&#772;<sub>k</sub></i> (from original <i>W</i>)")
    sc = d.box(0, 225, 230, 60, "SINdex++ = <i>H</i>(<i>p</i>) + &lambda; &Sigma;<sub><i>k</i></sub> <i>p<sub>k</sub></i>(1 &minus; <i>c&#772;<sub>k</sub></i>)", BLUE)
    out = d.box(0, 325, 230, 50, "Hallucination score<br>(higher = less reliable)")
    for a, b in [(q, s), (s, e), (e, w), (w, dia), (nli, veto), (hac, cl), (cl, sc), (sc, out)]:
        d.edge(a, b)
    d.edge(dia, hac, "no")
    d.edge(dia, nli, "yes")
    d.edge(veto, hac)
    d.box(470, 320, 24, 16, "", BLUE)
    d.text(500, 314, 260, 28, "this paper (C1, C2)")
    d.box(470, 348, 24, 16, "", GREY)
    d.text(500, 342, 260, 28, "as in SINdex")
    d.save()


def evaluation():
    d = Diagram("evaluation")
    ml = d.box(0, 160, 120, 60, "Most-likely<br>answer (<i>T</i> = 0.1)")
    fr = d.frame(150, 0, 300, 380, "Labeling rules")
    d.box(165, 30, 270, 44, "Embedding cosine &gt; 0.95<br>(SINdex protocol)", ORANGE)
    d.box(165, 84, 270, 34, "Exact match")
    d.box(165, 126, 270, 34, "Token F1 &ge; 0.5")
    d.box(165, 168, 270, 34, "Exact match &or; contains")
    jd = d.box(165, 214, 270, 70, "3 LLM judges, revised prompt:<br>Qwen2.5-7B &middot; Qwen2.5-14B<br>&middot; Granite-3.1-8B", AQUA)
    mv = d.box(165, 318, 270, 44, "Majority vote &rarr; validated labels", BLUE)
    d.edge(ml, fr)
    d.edge(jd, mv)
    hum = d.box(510, 0, 250, 60, "Human re-annotation (author)<br>40-item sample: 40/40, &kappa; = 1.00", AQUA)
    ref = d.box(510, 95, 250, 60, "Reference annotation<br>201 items, blind, LLM-assisted", AQUA)
    d.edge(hum, ref, "validates")
    acc = d.box(510, 190, 250, 76, "Label accuracy (Table II)<br>embedding rule: 56.8 %, &kappa; = 0.26<br>majority vote: 97.3 %, &kappa; = 0.94")
    d.edge(ref, acc)
    d.edge(fr, acc, "each rule", "dashed=1;")
    au = d.box(510, 312, 250, 60, "AUROC with paired and pooled<br>bootstrap (Tables III&ndash;V)", BLUE)
    det = d.box(800, 312, 170, 60, "Detector scores<br>(SINdex, variants, baselines)")
    d.edge(mv, au)
    d.edge(det, au)
    d.save()


def example():
    import json
    ex = json.loads((HERE.parent / "tables" / "example_kfc.json").read_text())
    d = Diagram("example")
    d.box(0, 0, 860, 46, "Q: How many herbs and spices make up the secret blend of KFC&#39;s Original Recipe? "
          "&nbsp; Reference answer: <b>11</b> &nbsp; Most-likely answer: <b>30</b> (wrong)", "fillColor=#ffffff;strokeColor=#9a9993;")
    fr1 = d.frame(0, 70, 380, 230, "Embedding only (SINdex): 1 cluster")
    groups = {"30": [], "33": [], "11": []}
    for t, a in zip(ex["texts"], ex["after"]):
        key = "30" if "30" in t else ("33" if "33" in t else "11")
        groups[key].append(t)
    y = 100
    lines = []
    for k in ("30", "33", "11"):
        for t in groups[k]:
            lines.append(t)
    d.box(15, 100, 350, 150, "<br>".join(html.escape(t) for t in lines).replace("&lt;", "<").replace("&gt;", ">"), GREY,
          "align=left;spacingLeft=8;fontSize=10;")
    d.text(15, 260, 360, 30, "SINdex = 0.016 (looks confident; 37th percentile)", "fontSize=11;fontColor=#0b0b0b;align=left;fontStyle=1;")
    fr2 = d.frame(480, 70, 380, 230, "With NLI veto (ours): 3 clusters")
    d.box(495, 100, 110, 150, "<b>&ldquo;30&rdquo;</b> &times;5<br><br>" + "<br>".join(html.escape(t) for t in groups["30"][:2]) + "<br>&hellip;", BLUE, "fontSize=10;")
    d.box(620, 100, 110, 150, "<b>&ldquo;33&rdquo;</b> &times;3<br><br>" + html.escape(groups["33"][0]) + "<br>&hellip;", ORANGE, "fontSize=10;")
    d.box(745, 100, 100, 150, "<b>&ldquo;11&rdquo;</b> &times;2<br><br>" + html.escape(groups["11"][0]), AQUA, "fontSize=10;")
    d.text(495, 260, 360, 30, "SINdex++ = 1.039 (uncertain; 64th percentile)", "fontSize=11;fontColor=#0b0b0b;align=left;fontStyle=1;")
    d.edge(fr1, fr2, "NLI<br>veto", "labelBackgroundColor=#ffffff;")
    d.save()


if __name__ == "__main__":
    pipeline()
    evaluation()
    example()
    print("wrote", sorted(p.name for p in HERE.glob("*.drawio")))
