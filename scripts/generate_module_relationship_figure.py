from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

W, H = 2200, 1180
BG, INK, MUTED = "#FFFFFF", "#20242B", "#6B7280"
BLUE, BLUE_D = "#DDEEFF", "#7DA9D6"
PEACH, PEACH_D = "#FFE1C8", "#D98C54"
LILAC, LILAC_D = "#E9E2F6", "#9074B5"
GREEN, GREEN_D = "#E8F2D7", "#88A75A"
YELLOW, YELLOW_D = "#FFF1B8", "#C6A33D"
GRAY = "#F4F5F7"

def fnt(size, bold=False):
    p = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    return ImageFont.truetype(str(p), size) if p.exists() else ImageFont.load_default()

F16, F18, F21, F24, F28, F34 = fnt(16), fnt(18), fnt(21), fnt(24, True), fnt(28, True), fnt(34, True)
im = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(im)

def rr(box, fill=BG, outline=INK, width=3, radius=16):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def txt(box, text, font=F18, fill=INK, align="center"):
    x1,y1,x2,y2=box; lines=text.split("\n")
    hs=[d.textbbox((0,0),s,font=font)[3] for s in lines]
    y=(y1+y2-(sum(hs)+(len(lines)-1)*5))/2
    for s,h in zip(lines,hs):
        b=d.textbbox((0,0),s,font=font); tw=b[2]-b[0]
        x=(x1+x2-tw)/2 if align=="center" else x1
        d.text((x,y),s,font=font,fill=fill); y+=h+5

def arrow(points, color=INK, width=4, dashed=False):
    import math
    for a,b in zip(points[:-1],points[1:]):
        if dashed:
            dist=((b[0]-a[0])**2+(b[1]-a[1])**2)**.5; n=max(1,int(dist/18))
            for i in range(0,n,2):
                p=i/n; q=min((i+1)/n,1)
                d.line((a[0]+(b[0]-a[0])*p,a[1]+(b[1]-a[1])*p,a[0]+(b[0]-a[0])*q,a[1]+(b[1]-a[1])*q),fill=color,width=width)
        else: d.line((*a,*b),fill=color,width=width)
    a,b=points[-2],points[-1]; ang=math.atan2(b[1]-a[1],b[0]-a[0]); L=15
    d.polygon([b,(b[0]-L*math.cos(ang-.55),b[1]-L*math.sin(ang-.55)),(b[0]-L*math.cos(ang+.55),b[1]-L*math.sin(ang+.55))],fill=color)

def stack(box, fill, layers=5):
    x1,y1,x2,y2=box
    for i in reversed(range(layers)):
        off=i*12
        rr((x1-off,y1+off,x2-off,y2+off),fill=fill,width=2,radius=13)

def tokens(x,y,colors):
    for r,row in enumerate(colors):
        for c,col in enumerate(row):
            rr((x+c*30,y+r*30,x+24+c*30,y+24+r*30),fill=col,outline=BLUE_D,width=2,radius=4)

# Title and implementation boundary
txt((60,28,2140,82),"Hybrid I-JEPA–LeWM: Module Relationships",F34)
d.rounded_rectangle((55,115,2145,930),radius=24,outline="#AEB4BD",width=3)
txt((85,125,500,168),"CURRENT TRAINING CORE",F21,MUTED,align="left")

# Shared encoder hub
stack((160,360,470,630),GRAY,6)
txt((125,375,455,545),"Eθ",F34)
txt((125,500,455,610),"Shared Vision\nTransformer",F24)
txt((120,700,480,735),"representation backbone",F18,MUTED)

# Branch junction
arrow([(470,495),(570,495)],INK)
d.ellipse((570,472,616,518),fill=YELLOW,outline=INK,width=3)

# Top: I-JEPA masked prediction path
arrow([(616,495),(665,495),(665,280),(760,280)],BLUE_D)
tokens(770,220,[[BLUE,BLUE,BLUE],[BLUE,PEACH,PEACH],[BLUE,BLUE,BLUE],[BLUE,PEACH,BLUE]])
txt((735,345,930,390),"Context selection",F18)
arrow([(930,280),(1040,280)],BLUE_D)
stack((1060,185,1340,370),PEACH,4)
txt((1025,205,1325,285),"Pφ",F34)
txt((1025,275,1325,350),"Masked Predictor",F24)
arrow([(1340,280),(1480,280)],PEACH_D)
tokens(1500,220,[[BLUE,PEACH,BLUE],[PEACH,BLUE,PEACH],[BLUE,PEACH,BLUE],[PEACH,BLUE,BLUE]])
txt((1455,345,1675,390),"Predicted embeddings",F18)
arrow([(1675,280),(1800,280)],PEACH_D)
rr((1810,205,2070,355),fill=PEACH,outline=PEACH_D,width=3)
txt((1810,220,2070,290),"Lpred",F28)
txt((1810,290,2070,340),"masked prediction",F18)

# Target relation from shared encoder to prediction loss
arrow([(300,630),(300,795),(1130,795),(1130,685)],BLUE_D,3)
tokens(1080,555,[[GREEN],[GREEN],[GREEN],[GREEN],[GREEN]])
txt((1015,710,1195,750),"Target branch",F18)
arrow([(1180,620),(1810,320)],BLUE_D,3,True)
txt((1375,525,1580,565),"stop-gradient",F16,MUTED)

# Middle: LeWM regularization path
arrow([(616,495),(760,495)],LILAC_D)
rr((780,430,930,560),fill=LILAC,outline=LILAC_D,width=3)
txt((780,445,930,525),"CLS",F28)
txt((780,520,930,550),"global token",F16)
arrow([(930,495),(1040,495)],LILAC_D)
# projection fan
for i in range(5):
    d.line((1065,495,1190,440+i*28),fill=LILAC_D,width=2)
    d.ellipse((1180,430+i*28,1200,450+i*28),fill=LILAC,outline=LILAC_D,width=2)
txt((1035,590,1230,630),"Random projections",F18)
arrow([(1210,495),(1370,495)],LILAC_D)
rr((1390,420,1625,570),fill=LILAC,outline=LILAC_D,width=3)
txt((1390,435,1625,500),"SIGReg",F28)
txt((1390,500,1625,555),"isotropic prior",F18)
arrow([(1625,495),(1810,495)],LILAC_D)
rr((1810,420,2070,570),fill=LILAC,outline=LILAC_D,width=3)
txt((1810,435,2070,500),"Lsigreg",F28)
txt((1810,500,2070,555),"collapse prevention",F18)

# Joint objective
arrow([(2070,280),(2110,280),(2110,700),(1785,700)],PEACH_D,3)
arrow([(1810,495),(1785,495),(1785,700)],LILAC_D,3)
rr((1490,635,1785,765),fill=YELLOW,outline=YELLOW_D,width=4)
txt((1490,645,1785,705),"Joint Objective",F28)
txt((1490,705,1785,750),"Lpred + 0.09 Lsigreg",F18)

# Optional extension outside training core
d.rounded_rectangle((55,965,2145,1135),radius=22,fill="#FBFCF9",outline=GREEN_D,width=3)
txt((85,980,430,1020),"OPTIONAL DYNAMICS EXTENSION",F21,GREEN_D,align="left")
rr((520,1000,790,1100),fill=GREEN,outline=GREEN_D,width=3)
txt((520,1010,790,1090),"Action Encoder",F24)
arrow([(790,1050),(940,1050)],GREEN_D)
stack((960,990,1320,1090),GREEN,3)
txt((925,1000,1300,1080),"AdaLN Conditional Blocks",F24)
arrow([(1320,1050),(1470,1050)],GREEN_D)
rr((1490,1000,1800,1100),fill=GREEN,outline=GREEN_D,width=3)
txt((1490,1010,1800,1090),"Autoregressive Rollout",F24)
arrow([(300,930),(300,1050),(520,1050)],GREEN_D,3,True)

png=OUT/"hybrid-jepa-lewm-modules.png"
im.save(png,dpi=(220,220))

# Editable draw.io version
cells=[]
def cell(i,v,style,x=0,y=0,w=0,h=0,vertex=True,edge=False,source=None,target=None):
    attrs=f'id="{i}" value="{escape(v)}" style="{style}" parent="1"'
    if vertex: attrs+=' vertex="1"'
    if edge:
        attrs+=' edge="1"'
        if source: attrs+=f' source="{source}"'
        if target: attrs+=f' target="{target}"'
    geo=f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>' if vertex else '<mxGeometry relative="1" as="geometry"/>'
    cells.append(f'<mxCell {attrs}>{geo}</mxCell>')

box="rounded=1;whiteSpace=wrap;html=1;strokeWidth=2;fontSize=18;fontStyle=1;"
edge="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeWidth=2;"
cell("title","Hybrid I-JEPA–LeWM: Module Relationships","text;html=1;align=center;fontSize=28;fontStyle=1;",70,20,1900,45)
cell("core","CURRENT TRAINING CORE","rounded=1;whiteSpace=wrap;html=1;verticalAlign=top;align=left;spacing=16;strokeWidth=2;dashed=1;fontSize=16;fontColor=#6b7280;",45,85,2000,720)
cell("enc","Eθ\nShared Vision Transformer",box+"fillColor=#f4f5f7;fontSize=24;",110,310,300,220)
cell("junction","","ellipse;html=1;strokeWidth=2;fillColor=#fff1b8;",500,395,40,40)
cell("context","Context Selection",box+"fillColor=#ddeeff;strokeColor=#7da9d6;",665,160,220,115)
cell("pred","Pφ\nMasked Predictor",box+"fillColor=#ffe1c8;strokeColor=#d98c54;fontSize=22;",985,145,270,145)
cell("pembed","Predicted Embeddings",box+"fillColor=#ddeeff;strokeColor=#7da9d6;",1355,160,235,115)
cell("lp","Lpred\nmasked prediction",box+"fillColor=#ffe1c8;strokeColor=#d98c54;fontSize=22;",1710,145,260,145)
cell("target","Target Branch",box+"fillColor=#e8f2d7;strokeColor=#88a75a;",900,595,220,110)
cell("cls","CLS\nglobal token",box+"fillColor=#e9e2f6;strokeColor=#9074b5;fontSize=22;",665,390,180,120)
cell("rp","Random Projections",box+"fillColor=#e9e2f6;strokeColor=#9074b5;",945,390,230,120)
cell("sig","SIGReg\nisotropic prior",box+"fillColor=#e9e2f6;strokeColor=#9074b5;fontSize=22;",1275,390,230,120)
cell("ls","Lsigreg\ncollapse prevention",box+"fillColor=#e9e2f6;strokeColor=#9074b5;fontSize=22;",1710,390,260,120)
cell("joint","Joint Objective\nLpred + 0.09 Lsigreg",box+"fillColor=#fff1b8;strokeColor=#c6a33d;fontSize=22;",1425,620,310,115)
for n,s,t,c in [("a","enc","junction","#20242b"),("b","junction","context","#7da9d6"),("c","context","pred","#7da9d6"),("d","pred","pembed","#d98c54"),("e","pembed","lp","#d98c54"),("f","junction","cls","#9074b5"),("g","cls","rp","#9074b5"),("h","rp","sig","#9074b5"),("i","sig","ls","#9074b5"),("j","enc","target","#7da9d6"),("k","lp","joint","#d98c54"),("l","ls","joint","#9074b5")]:
    st=edge+f"strokeColor={c};"
    if n=="j": st+="dashed=1;"
    cell(n,"stop-gradient" if n=="j" else "",st,vertex=False,edge=True,source=s,target=t)
cell("optional","OPTIONAL DYNAMICS EXTENSION","rounded=1;whiteSpace=wrap;html=1;verticalAlign=top;align=left;spacing=14;strokeWidth=2;dashed=1;strokeColor=#88a75a;fontSize=16;fontColor=#88a75a;",45,845,2000,180)
cell("action","Action Encoder",box+"fillColor=#e8f2d7;strokeColor=#88a75a;",430,895,260,90)
cell("adaln","AdaLN Conditional Blocks",box+"fillColor=#e8f2d7;strokeColor=#88a75a;",840,895,330,90)
cell("rollout","Autoregressive Rollout",box+"fillColor=#e8f2d7;strokeColor=#88a75a;",1320,895,320,90)
for n,s,t in [("m","action","adaln"),("n","adaln","rollout")]: cell(n,"",edge+"strokeColor=#88a75a;",vertex=False,edge=True,source=s,target=t)
xml=f'''<mxfile host="app.diagrams.net" modified="2026-07-11T00:00:00.000Z" agent="Codex" version="24.7.17" type="device"><diagram id="modules" name="Module Relationships"><mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" page="1" pageScale="1" pageWidth="2100" pageHeight="1100" math="1"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{''.join(cells)}</root></mxGraphModel></diagram></mxfile>'''
drawio=OUT/"hybrid-jepa-lewm-modules.drawio"
drawio.write_text(xml,encoding="utf-8")
print(png); print(drawio)
