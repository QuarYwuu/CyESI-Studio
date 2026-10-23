"""Small native UI primitives. Workbench styles use their own names."""
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from PIL import Image,ImageTk,ImageSequence

BG='#eef2f7';WHITE='#ffffff';INK='#18324d';MUTED='#708195';LINE='#dce4ed'
NAV='#11283f';NAV_ACTIVE='#23445f';TEAL='#0b9185';BLUE='#367de0';PURPLE='#8c68d3'
AMBER='#b8771c';PALE='#f6f9fc'
FONT='Microsoft YaHei UI'


def styles(root):
    style=ttk.Style(root)
    # Use workbench-specific style names to avoid changing other windows.
    style.configure('WB.TEntry',padding=6,fieldbackground=WHITE,font=(FONT,10))
    style.configure('WB.TCombobox',padding=5,font=(FONT,10))
    style.configure('WB.Treeview',font=(FONT,10),rowheight=32,background=WHITE,fieldbackground=WHITE,foreground=INK)
    style.configure('WB.Treeview.Heading',font=(FONT,10,'bold'),padding=8)
    style.map('WB.Treeview',background=[('selected','#e0f3f0')],foreground=[('selected',INK)])
    style.configure('WB.TNotebook',background=BG,borderwidth=0)
    style.configure('WB.TNotebook.Tab',padding=(16,9),font=(FONT,10))
    style.map('WB.TNotebook.Tab',background=[('selected',WHITE)],foreground=[('selected',TEAL)])


def label(parent,text='',size=10,color=INK,bold=False,**kwargs):
    return tk.Label(parent,text=text,font=(FONT,size,'bold' if bold else 'normal'),fg=color,
        bg=kwargs.pop('bg',parent.cget('bg')),anchor=kwargs.pop('anchor','w'),**kwargs)


def button(parent,text,command,primary=False,**kwargs):
    color=TEAL if primary else WHITE
    return tk.Button(parent,text=text,command=command,font=(FONT,10,'bold' if primary else 'normal'),
        bg=color,fg=WHITE if primary else INK,activebackground='#087c73' if primary else '#e8eef6',
        activeforeground=WHITE if primary else INK,relief='flat',bd=0,padx=15,pady=9,cursor='hand2',**kwargs)


class Card(tk.Frame):
    def __init__(self,parent,title=None,subtitle=None,**kwargs):
        super().__init__(parent,bg=WHITE,highlightbackground=LINE,highlightthickness=1,**kwargs)
        if title:
            head=tk.Frame(self,bg=WHITE);head.pack(fill='x',padx=16,pady=(14,8))
            label(head,title,size=11,bold=True).pack(anchor='w')
            if subtitle:label(head,subtitle,size=9,color=MUTED,wraplength=850,justify='left').pack(anchor='w',pady=(3,0))


class Metric(Card):
    def __init__(self,parent,title,unit,color=TEAL):
        super().__init__(parent)
        self.title=label(self,title,size=9,color=MUTED);self.title.pack(anchor='w',padx=14,pady=(11,2))
        self.value=label(self,'—',size=22,color=color,bold=True);self.value.pack(anchor='w',padx=14)
        self.note=label(self,unit,size=9,color=MUTED);self.note.pack(anchor='w',padx=14,pady=(0,10))

    def set(self,value,note=None):
        self.value.configure(text='—' if value is None else f'{value:.4g}' if isinstance(value,(int,float)) else str(value))
        if note is not None:self.note.configure(text=note)


class ScrollFrame(tk.Frame):
    def __init__(self,parent,bg=WHITE):
        super().__init__(parent,bg=bg)
        self.canvas=tk.Canvas(self,bg=bg,highlightthickness=0)
        bar=ttk.Scrollbar(self,orient='vertical',command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=bar.set);bar.pack(side='right',fill='y');self.canvas.pack(side='left',fill='both',expand=True)
        self.body=tk.Frame(self.canvas,bg=bg);self.item=self.canvas.create_window(0,0,window=self.body,anchor='nw')
        self.body.bind('<Configure>',lambda e:self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>',lambda e:self.canvas.itemconfigure(self.item,width=e.width))
        # Bind locally on the widgets; do not capture mouse wheels in other
        # windows by installing bind_all handlers.
        self.canvas.bind('<MouseWheel>',self.wheel)
        self.body.bind('<MouseWheel>',self.wheel)
    def wheel(self,event):self.canvas.yview_scroll(-int(event.delta/120),'units')


class Plot(tk.Frame):
    def __init__(self,parent):
        super().__init__(parent,bg=WHITE)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        class VisibleCanvas(FigureCanvasTkAgg):
            def draw(canvas):
                # Mapping and rapid page changes can briefly allocate a 1px
                # canvas. Wait for the next resize/map draw instead of running
                # constrained layout on a hidden or collapsed figure.
                widget=canvas.get_tk_widget()
                if not widget.winfo_exists() or not widget.winfo_ismapped() or widget.winfo_width()<160 or widget.winfo_height()<100:return
                super().draw()
        self.figure=Figure(figsize=(8,4),dpi=100,facecolor=WHITE,layout='constrained')
        self.canvas=VisibleCanvas(self.figure,master=self)
        self.canvas.get_tk_widget().pack(fill='both',expand=True)
        self.canvas.get_tk_widget().bind('<Map>',lambda e:self.canvas.draw_idle(),add='+')
        self.key=None
    def axes(self,n=1):
        self.figure.clear()
        axes=self.figure.subplots(1,n,squeeze=False)[0]
        for ax in axes:
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(labelsize=9,colors=MUTED);ax.set_facecolor(WHITE)
        return axes
    def show(self):
        if self.canvas.get_tk_widget().winfo_ismapped():self.canvas.draw_idle()
    def save(self,path):self.figure.savefig(path,dpi=160)


class Picture(tk.Canvas):
    def __init__(self,parent):
        super().__init__(parent,bg=WHITE,highlightthickness=0)
        self.path=None;self.frames=[];self.photo=None;self.index=0;self.token=None
        self.bind('<Configure>',lambda e:self.render())
    def load(self,path):
        path=Path(path)
        token=(str(path),path.stat().st_mtime_ns,path.stat().st_size) if path.exists() else (str(path),None,None)
        if self.token==token:return
        self.token=token
        self.path=path;self.frames=[]
        if path.exists():
            with Image.open(path) as im:self.frames=[frame.convert('RGB').copy() for frame in ImageSequence.Iterator(im)]
        self.index=0;self.render()
    def render(self,index=None):
        self.delete('all')
        if index is not None:self.index=index
        if not self.frames:
            self.create_text(max(150,self.winfo_width()/2),100,text='尚无此图; 可查看现有数值场或生成原图.',fill=MUTED,font=(FONT,10));return
        im=self.frames[self.index%len(self.frames)].copy()
        im.thumbnail((max(200,self.winfo_width()-20),max(150,self.winfo_height()-20)),Image.Resampling.LANCZOS)
        self.photo=ImageTk.PhotoImage(im);self.create_image(self.winfo_width()/2,self.winfo_height()/2,image=self.photo)


def table(parent,columns,headings,widths=None,height=8):
    box=tk.Frame(parent,bg=WHITE)
    tree=ttk.Treeview(box,columns=columns,show='headings',height=height,style='WB.Treeview',selectmode='browse')
    for i,(key,title) in enumerate(zip(columns,headings)):
        tree.heading(key,text=title);tree.column(key,width=(widths or [130]*len(columns))[i],minwidth=75,anchor='center')
    y=ttk.Scrollbar(box,orient='vertical',command=tree.yview);y.pack(side='right',fill='y')
    x=ttk.Scrollbar(box,orient='horizontal',command=tree.xview);x.pack(side='bottom',fill='x')
    tree.configure(yscrollcommand=y.set,xscrollcommand=x.set);tree.pack(fill='both',expand=True)
    return box,tree
