from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

W, H = 2400, 1080
BG = "#FFFFFF"
INK = "#252525"
MUTED = "#6B7280"
BLUE = "#DCEEFF"
BLUE2 = "#B9DCF5"
ORANGE = "#FFE1C7"
GREEN = "#E7F1D1"
PURPLE = "#E9E1F5"
YELLOW = "#FFF2B8"
RED = "#F7B7A3"

def font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()

F16, F19, F22, F25, F30 = font(16), font(19), font(22), font(25, True), font(30, True)

im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)

def rr(box, fill="#FFFFFF", outline=INK, width=3, radius=14):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def center_text(box, text, f=F19, fill=INK):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    heights = [d.textbbox((0, 0), s, font=f)[3] for s in lines]
    total = sum(heights) + (len(lines)-1)*4
    y = (y1+y2-total)/2
    for s, h in zip(lines, heights):
        b = d.textbbox((0, 0), s, font=f)
        d.text(((x1+x2-(b[2]-b[0]))/2, y), s, font=f, fill=fill)
        y += h+4

def arrow(x1, y1, x2, y2, color=INK, width=4, dashed=False):
    if dashed:
        n = max(1, int(((x2-x1)**2+(y2-y1)**2)**0.5/18))
        for i in range(0, n, 2):
            a, b = i/n, min((i+1)/n, 1)
            d.line((x1+(x2-x1)*a, y1+(y2-y1)*a, x1+(x2-x1)*b, y1+(y2-y1)*b), fill=color, width=width)
    else:
        d.line((x1, y1, x2, y2), fill=color, width=width)
    import math
    ang = math.atan2(y2-y1, x2-x1)
    l = 14
    pts = [(x2,y2),(x2-l*math.cos(ang-.55),y2-l*math.sin(ang-.55)),(x2-l*math.cos(ang+.55),y2-l*math.sin(ang+.55))]
    d.polygon(pts, fill=color)

def header(box, letter, title):
    rr(box, fill="#FAFAFA", width=4)
    x1,y1,x2,y2=box
    d.ellipse((x1+14,y1+10,x1+72,y1+68), fill="#FFC21C", outline=INK, width=3)
    center_text((x1+14,y1+10,x1+72,y1+68), letter, F25)
    center_text((x1+82,y1,x2-8,y2), title, F25)

header((35,30,455,105), "A", "Input & Masking")
header((475,30,1980,105), "B", "Hybrid I-JEPA–LeWM")
header((2000,30,2365,105), "C", "Objectives")

# Main section frames
rr((35,120,455,1000), fill="#FCFCFC", width=4)
rr((475,120,1980,1000), fill="#FCFCFC", width=4)
rr((2000,120,2365,1000), fill="#FCFCFC", width=4)

# Abstract industrial image tile
rr((70,170,420,405), fill="#D8E6EA", width=2)
d.rectangle((70,170,420,250), fill="#93B6C4")
d.polygon([(70,405),(70,280),(180,225),(300,290),(420,235),(420,405)], fill="#B9CAD0")
d.rectangle((115,285,375,365), fill="#5C7881", outline=INK, width=2)
for i in range(6):
    x=135+i*38
    d.ellipse((x,305,x+17,322), fill="#FFD35A", outline=INK, width=1)
    d.line((x+8,322,x+8,350), fill="#D6E7EC", width=4)
center_text((70,410,420,455), "industrial image", F19)

# Patch grid and mask grid
def grid(x,y,cell=28,n=6,masked=False):
    for r in range(n):
        for c in range(n):
            fill = BLUE
            if masked and ((r in (1,2) and c in (3,4)) or (r in (4,5) and c in (0,1,2))): fill="#555555"
            d.rectangle((x+c*cell,y+r*cell,x+(c+1)*cell-2,y+(r+1)*cell-2), fill=fill, outline="#6D8FB5", width=1)
grid(96,520,38,6,False)
arrow(210,460,210,510)
center_text((80,750,430,790), "Patchify  •  14 × 14 tokens", F19)
grid(140,805,25,6,True)
center_text((80,955,430,995), "Multi-block mask", F19)

# B: token streams
center_text((505,130,1945,172), "Shared ViT encoder, two aligned learning paths", F22)

def token_stack(x,y,color=BLUE,accent=None, count=7):
    for i in range(count):
        fill = accent if accent and i in (2,5) else color
        rr((x,y+i*34,x+32,y+28+i*34), fill=fill, outline="#6D8FB5", width=2, radius=5)

token_stack(535,245)
center_text((495,490,620,530), "visible\ntokens", F16)
arrow(570,360,690,360)

# Encoder stacked slabs
for i in range(5):
    rr((690+i*18,245-i*8,820+i*18,475-i*8), fill="#F5F5F5", width=3, radius=10)
center_text((730,275,900,430), "Eθ\nViT-Tiny", F25)
center_text((690,485,900,525), "context encoder", F16)
arrow(920,360,1015,360)

# Context features / predictor
token_stack(1020,245,color=BLUE2,count=7)
center_text((995,490,1100,530), "context z", F16)
arrow(1060,360,1180,360)
for i in range(4):
    rr((1180+i*15,260-i*7,1350+i*15,460-i*7), fill=ORANGE, width=3, radius=18)
center_text((1210,300,1400,415), "Pφ\nPredictor ×6", F25)
arrow(1420,360,1530,360)
token_stack(1535,245,color=BLUE,accent=RED,count=7)
center_text((1505,490,1615,530), "predicted\ntargets", F16)

# Target path
grid(535,665,25,6,False)
center_text((500,835,720,875), "full token grid", F16)
arrow(700,740,790,740)
for i in range(4):
    rr((790+i*16,635-i*8,925+i*16,850-i*8), fill="#F5F5F5", width=3, radius=10)
center_text((825,680,990,805), "Eθ\nshared", F25)
arrow(990,740,1090,740)
token_stack(1095,625,color=GREEN,count=7)
center_text((1070,870,1180,910), "targets h", F16)
arrow(1180,740,1535,445,color="#6D8FB5",width=3,dashed=True)
center_text((1240,545,1450,580), "stop-gradient", F16, MUTED)

# CLS / SIGReg lower branch
rr((1450,575,1935,900), fill="#F8FBF1", outline="#9DB36A", width=2)
center_text((1465,590,1920,625), "LeWM branch", F22)
rr((1540,680,1630,760), fill=PURPLE, width=3)
center_text((1540,680,1630,760), "CLS", F19)
arrow(1635,720,1760,720)
d.ellipse((1760,675,1830,745), fill=PURPLE, outline=INK, width=3)
center_text((1760,675,1830,745), "R", F22)
center_text((1510,790,1890,855), "random projections  •  SIGReg", F19)
arrow(990,820,1450,720,color="#7A5AA6",width=3)

# Objective panel
rr((2040,205,2325,410), fill=ORANGE, width=3)
center_text((2050,225,2315,320), "Lpred", F30)
center_text((2050,320,2315,390), "patch prediction", F19)
arrow(1610,360,2040,300,color="#D96832",width=3,dashed=True)

rr((2040,500,2325,705), fill=PURPLE, width=3)
center_text((2050,520,2315,615), "Lsigreg", F30)
center_text((2050,615,2315,685), "isotropic features", F19)
arrow(1830,720,2040,585,color="#7A5AA6",width=3,dashed=True)

rr((2040,785,2325,945), fill=YELLOW, width=4)
center_text((2050,800,2315,865), "Ltotal", F30)
center_text((2050,865,2315,930), "Lpred + 0.09 Lsigreg", F19)
arrow(2180,410,2180,500)
arrow(2180,705,2180,785)

# Small legend
d.rectangle((510,940,535,965),fill=BLUE2,outline="#6D8FB5")
d.text((545,940),"I-JEPA",font=F16,fill=INK)
d.rectangle((650,940,675,965),fill=PURPLE,outline="#7A5AA6")
d.text((685,940),"LeWM",font=F16,fill=INK)
d.rectangle((790,940,815,965),fill=RED,outline="#D96832")
d.text((825,940),"target token",font=F16,fill=INK)

png_path = OUT / "hybrid-jepa-lewm-overview.png"
im.save(png_path, dpi=(220,220))

# Draw.io XML: same architecture using fully editable primitives.
cells = []
def cell(id_, value, style, x, y, w, h, parent="1", vertex=True, edge=False, source=None, target=None):
    attrs = f'id="{id_}" value="{escape(value)}" style="{style}" parent="{parent}"'
    if vertex: attrs += ' vertex="1"'
    if edge:
        attrs += ' edge="1"'
        if source: attrs += f' source="{source}"'
        if target: attrs += f' target="{target}"'
    geo = f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>' if vertex else '<mxGeometry relative="1" as="geometry"/>'
    cells.append(f'<mxCell {attrs}>{geo}</mxCell>')

rounded = "rounded=1;whiteSpace=wrap;html=1;strokeWidth=2;fontSize=16;"
label = "text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;fontSize=18;fontStyle=1;"
edge = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeWidth=2;"
dashed = edge + "dashed=1;"

cell("hA","A   Input &amp; Masking",rounded+"fillColor=#fafafa;fontSize=22;fontStyle=1;",20,20,390,62)
cell("hB","B   Hybrid I-JEPA–LeWM",rounded+"fillColor=#fafafa;fontSize=22;fontStyle=1;",430,20,1370,62)
cell("hC","C   Objectives",rounded+"fillColor=#fafafa;fontSize=22;fontStyle=1;",1820,20,330,62)
cell("pA","",rounded+"fillColor=#fcfcfc;",20,95,390,810)
cell("pB","",rounded+"fillColor=#fcfcfc;",430,95,1370,810)
cell("pC","",rounded+"fillColor=#fcfcfc;",1820,95,330,810)
cell("img","INDUSTRIAL\nIMAGE",rounded+"fillColor=#d8e6ea;fontSize=22;fontStyle=1;",65,150,300,180)
cell("patch","PATCH GRID\n14 × 14",rounded+"fillColor=#dceeff;fontSize=18;",95,410,240,150)
cell("mask","MULTI-BLOCK\nMASK",rounded+"fillColor=#555555;fontColor=#ffffff;fontSize=18;",95,650,240,135)
cell("e1","",edge,0,0,0,0,vertex=False,edge=True,source="img",target="patch")
cell("e2","",edge,0,0,0,0,vertex=False,edge=True,source="patch",target="mask")
cell("vis","VISIBLE\nTOKENS",rounded+"fillColor=#dceeff;",475,250,100,180)
cell("enc","Eθ\nViT-Tiny",rounded+"fillColor=#f5f5f5;fontSize=24;fontStyle=1;",650,230,180,220)
cell("ctx","CONTEXT z",rounded+"fillColor=#b9dcf5;",900,250,110,180)
cell("pred","Pφ\nPredictor ×6",rounded+"fillColor=#ffe1c7;fontSize=22;fontStyle=1;",1085,230,215,220)
cell("pt","PREDICTED\nTARGETS",rounded+"fillColor=#f7b7a3;",1375,250,125,180)
cell("cls","CLS",rounded+"fillColor=#e9e1f5;fontSize=22;fontStyle=1;",1545,260,90,75)
cell("sig","R\nSIGReg",rounded+"fillColor=#e9e1f5;fontSize=20;fontStyle=1;",1680,245,90,110)
cell("full","FULL TOKEN\nGRID",rounded+"fillColor=#dceeff;",500,590,145,145)
cell("enc2","Eθ\nshared",rounded+"fillColor=#f5f5f5;fontSize=22;fontStyle=1;",730,570,170,190)
cell("target","TARGETS h",rounded+"fillColor=#e7f1d1;",990,590,125,145)
for i,(s,t) in enumerate([("vis","enc"),("enc","ctx"),("ctx","pred"),("pred","pt"),("ctx","cls"),("cls","sig"),("full","enc2"),("enc2","target")]):
    cell(f"f{i}","",edge,0,0,0,0,vertex=False,edge=True,source=s,target=t)
cell("stop","stop-gradient",dashed+"strokeColor=#6d8fb5;fontSize=15;",0,0,0,0,vertex=False,edge=True,source="target",target="pt")
cell("lp","Lpred\npatch prediction",rounded+"fillColor=#ffe1c7;fontSize=22;fontStyle=1;",1860,180,250,150)
cell("ls","Lsigreg\nisotropic features",rounded+"fillColor=#e9e1f5;fontSize=22;fontStyle=1;",1860,420,250,150)
cell("lt","Ltotal\nLpred + 0.09 Lsigreg",rounded+"fillColor=#fff2b8;fontSize=22;fontStyle=1;",1860,685,250,145)
cell("l1","",dashed+"strokeColor=#d96832;",0,0,0,0,vertex=False,edge=True,source="pt",target="lp")
cell("l2","",dashed+"strokeColor=#7a5aa6;",0,0,0,0,vertex=False,edge=True,source="sig",target="ls")
cell("l3","",edge,0,0,0,0,vertex=False,edge=True,source="lp",target="lt")
cell("l4","",edge,0,0,0,0,vertex=False,edge=True,source="ls",target="lt")

xml = f'''<mxfile host="app.diagrams.net" modified="2026-07-11T00:00:00.000Z" agent="Codex" version="24.7.17" type="device">
  <diagram id="hybrid-jepa" name="Hybrid I-JEPA-LeWM">
    <mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2200" pageHeight="950" math="1" shadow="0">
      <root><mxCell id="0"/><mxCell id="1" parent="0"/>{''.join(cells)}</root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
(OUT / "hybrid-jepa-lewm-overview.drawio").write_text(xml, encoding="utf-8")
print(png_path)
print(OUT / "hybrid-jepa-lewm-overview.drawio")
