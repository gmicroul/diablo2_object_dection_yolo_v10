#!/usr/bin/env python3
"""暗黑2 触屏控制面板 — 最终版"""
import tkinter as tk
import pyautogui
import subprocess
import os, time

def wid():
    r = subprocess.run(["xdotool","search","--name","Wine Desktop"],
                       capture_output=True,text=True)
    return r.stdout.strip().split("\n")[0] if r.stdout.strip() else None

def refocus():
    w = wid()
    if w:
        subprocess.run(["xdotool","windowactivate",w])

def gk(k):
    pyautogui.press(k)

def gc(b):
    pyautogui.click(button=b)

BG = "#1a1a1a"; FG = "white"; BTN = "#333"

class Panel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("D2")
        self.root.overrideredirect(True)
        self.root.attributes('-topmost',True,'-alpha',0.75)
        self.root.configure(bg=BG)
        self.root.geometry("180x600+1340+100")
        self.root.bind('<Button-1>',self.ds)
        self.root.bind('<B1-Motion>',self.dm)
        self.build()

    def ds(self,e): self._dx=e.x; self._dy=e.y
    def dm(self,e): self.root.geometry(f"+{self.root.winfo_x()+e.x-self._dx}+{self.root.winfo_y()+e.y-self._dy}")

    def bt(self,t,c):
        def do():
            c()
            self.root.after(30, refocus)
        b = tk.Button(self.root,text=t,command=do,
                     font=("Segoe UI",9,"bold"),bg=BTN,fg=FG,
                     relief="flat",bd=1,width=7,height=1,
                     activebackground="#555",activeforeground=FG)
        return b

    def gr(self,w,r,c): w.grid(row=r,column=c,padx=3,pady=3,sticky="ew")
    def sp(self,r): tk.Frame(self.root,bg="#444",height=1).grid(row=r,column=0,columnspan=2,sticky="ew",padx=6,pady=2)
    def lb(self,t,r): tk.Label(self.root,text=t,font=("Segoe UI",7),bg=BG,fg="#999").grid(row=r,column=0,columnspan=2,pady=(4,0))

    def build(self):
        r=0
        self.lb("【战斗】",r); r+=1
        self.gr(self.bt("ESC",lambda: gk('esc')),r,0)
        self.gr(self.bt("🗺地图",lambda: gk('tab')),r,1)
        r+=1
        self.gr(self.bt("左键",lambda: gc('left')),r,0)
        self.gr(self.bt("右键",lambda: gc('right')),r,1)
        r+=1
        self.gr(self.bt("🔦高亮",lambda: (pyautogui.keyDown('alt'), time.sleep(0.1), pyautogui.keyUp('alt'))),r,0)
        self.gr(self.bt("🏃跑步",lambda: gk('r')),r,1)
        r+=1; self.sp(r); r+=1

        self.lb("【药水】",r); r+=1
        for i,k in enumerate(["1","2","3","4"]):
            self.gr(self.bt(k,lambda kk=k: gk(kk)),r+(i//2),i%2)
        r+=2; self.sp(r); r+=1

        self.lb("【快捷施法】",r); r+=1
        for i in range(4):
            self.gr(self.bt(f"F{i+1}",lambda k=f"F{i+1}": gk(k)),r+(i//2),i%2)
        r+=2
        for i in range(4):
            self.gr(self.bt(f"F{i+5}",lambda k=f"F{i+5}": gk(k)),r+(i//2),i%2)
        r+=2; self.sp(r); r+=1

        self.lb("【功能】",r); r+=1
        for i,(t,k) in enumerate([("回城","esc"),("装备","b"),("技能","s"),("任务","q"),("切换","w")]):
            self.gr(self.bt(t,lambda kk=k: gk(kk)),r+(i//2),i%2)
        r+=2
        for i,(t,k) in enumerate([("属性","a"),("背包","i"),("角色","c"),("佣兵","o")]):
            self.gr(self.bt(t,lambda kk=k: gk(kk)),r+(i//2),i%2)
        r+=2; self.sp(r); r+=1

        self.gr(self.bt("⏹退出",lambda: os._exit(0)),r,0)
        r+=1
        self.root.grid_columnconfigure(0,weight=1)
        self.root.grid_columnconfigure(1,weight=1)

    def run(self): self.root.mainloop()

if __name__=="__main__": Panel().run()