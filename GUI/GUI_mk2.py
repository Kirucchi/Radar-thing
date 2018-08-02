#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
#matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk
import os
from tkinter import filedialog
from pulson440_unpack import main
import scriptsave7

class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("UAS-SAR GUI")
        
        tab_font = ("arial 40")
        
        width = 1300
        height = 700
        x = int((self.winfo_screenwidth()-width)/2)
        y = int((self.winfo_screenheight()-height)/3)
        self.geometry("%dx%d+%d+%d" % (width, height, x, y))
        
        
        menu = tk.Frame(self)
        
        scan_tab = tk.Label(menu, text="Scan", font=tab_font, background="gray48", fg="gray90", relief="raised")
        scan_tab.bind("<Button-1>", lambda e: self.show_frame(Scan, self.current, scan_tab))
        scan_tab.pack(side="top", fill="both", expand=True)
        settings_tab = tk.Label(menu, text="Settings", font=tab_font, background="gray48", fg="gray90", relief="raised")
        settings_tab.bind("<Button-1>", lambda e: self.show_frame(Settings, self.current, settings_tab))
        settings_tab.pack(side="top", fill="both", expand=True)
        unpack_tab = tk.Label(menu, text="Unpack", font=tab_font, background="gray48", fg="gray90", relief="raised")
        unpack_tab.bind("<Button-1>", lambda e: self.show_frame(Unpack, self.current, unpack_tab))
        unpack_tab.pack(side="top", fill="both", expand=True)
        image_tab = tk.Label(menu, text="Image", font=tab_font, background="gray48", fg="gray90", relief="raised")
        image_tab.bind("<Button-1>", lambda e: self.show_frame(Image, self.current, image_tab))
        image_tab.pack(side="top", fill="both", expand=True)
        
        self.current = scan_tab
        
        container = tk.Frame(self)
        
        menu.place(anchor="w", relheight=1.0, rely=0.5, relwidth=0.2)
        container.place(anchor="w", relheight=1.0, rely=0.5, relwidth=0.8, relx=0.2)
        
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        
        for F in (Scan, Settings, Unpack, Image):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            frame.configure(background="gray32")
        
        self.show_frame(Scan, self.current, scan_tab)
        
        
        
    def show_frame(self, cont, current_tab, new_tab):
        frame = self.frames[cont]
        frame.tkraise()
        current_tab.configure(background="gray48", fg="gray99", relief="raised")
        new_tab.configure(background="gray32", fg="gray90", relief="flat")
        self.current = new_tab


class Scan(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        button_font = ("arial 15")
        
        self.pick_a_wifi = tk.Label(self, text="Pick a wifi network:", font=("arial 20"), background="azure2")
        self.pick_a_wifi.place(anchor="w", relheight=0.2, relwidth=0.5, rely=0.1, relx=0.25)
        
        self.UASSAR3_button = tk.Button(self, text="UASSAR3", font=button_font, command=lambda: self.connect("UASSAR3", self.label_1))
        self.UASSAR3_button.place(anchor="w", relheight=0.1, relwidth=0.15, rely=0.25, relx=0.08)
        self.MIT_button = tk.Button(self, text="MIT", font=button_font, command=lambda: self.connect("MIT", self.label_2))
        self.MIT_button.place(anchor="w", relheight=0.1, relwidth=0.15, rely=0.25, relx=0.31)
        self.MIT_GUEST_button = tk.Button(self, text="MIT GUEST", font=button_font, command=lambda: self.connect("MIT GUEST", self.label_3))
        self.MIT_GUEST_button.place(anchor="w", relheight=0.1, relwidth=0.15, rely=0.25, relx=0.54)
        self.Hyatt_wifi_button = tk.Button(self, text="@Hyatt_WiFi", font=button_font, command=lambda: self.connect("@Hyatt_WiFi", self.label_4))
        self.Hyatt_wifi_button.place(anchor="w", relheight=0.1, relwidth=0.15, rely=0.25, relx=0.77)
        self.label_1 = tk.Label(self, text="", background="azure2", font="10")
        self.label_1.place(anchor="w", relheight=0.03, relwidth=0.15, rely=0.32, relx=0.08)
        self.label_2 = tk.Label(self, text="", background="azure2", font="10")
        self.label_2.place(anchor="w", relheight=0.03, relwidth=0.15, rely=0.32, relx=0.31)
        self.label_3 = tk.Label(self, text="", background="azure2", font="10")
        self.label_3.place(anchor="w", relheight=0.03, relwidth=0.15, rely=0.32, relx=0.54)
        self.label_4 = tk.Label(self, text="", background="azure2", font="10")
        self.label_4.place(anchor="w", relheight=0.03, relwidth=0.15, rely=0.32, relx=0.77)
        
        self.current = self.label_2
        
        self.scan_start_button = tk.Button(self, text="Start scan", command=self.start_scan, font=button_font, background="chartreuse")
        self.scan_start_button.place(anchor="w", relheight=0.15, relwidth=0.2, rely=0.75, relx=0.15)
        self.scan_stop_button = tk.Button(self, text="Stop scan", command=self.stop_scan, font=button_font, background="orange red2")
        self.scan_stop_button.place(anchor="w", relheight=0.15, relwidth=0.2, rely=0.75, relx=0.65)
        self.currently_scanning_label = tk.Label(self, text="", font=("Arial 25 italic"), background="azure2")
        self.currently_scanning_label.place(anchor="w", relheight=0.15, relwidth=0.3, rely=0.75, relx=0.35)
        self.cannot_scan_label = tk.Label(self, text="", font="10", background="azure2")
        self.cannot_scan_label.place(anchor="w", relheight=0.05, relwidth=0.2, rely=0.85, relx=0.15)
        
        self.currently_scanning = False
        
        self.pick_scan_name_label = tk.Label(self, text="Scan data file name:", font="10", background="azure2")
        self.pick_scan_name_label.place(anchor="w", relheight=0.06, relwidth=0.2, relx=0.143, rely=0.5)
        self.pick_scan_name_entry = tk.Entry(self, font="Arial 20")
        self.pick_scan_name_entry.place(anchor="w", relheight=0.06, relwidth=0.5, relx=0.35, rely=0.5)
        
        self.scan_save_name = ""
        
        
    def connect(self, name, new_label):
        if self.currently_scanning == True:
            return
        os.system('netsh wlan connect name="%s"' % (name))
        self.current.configure(text="")
        new_label.configure(text="Connected!")
        self.current = new_label
        self.cannot_scan_label.configure(text="")
    
    def start_scan(self):
        if self.current != self.label_1:
            self.cannot_scan_label.configure(text="Connect to UASSAR3!")
            return
        if self.currently_scanning == True:
            self.cannot_scan_label.configure(text="Already scanning!")
            return
        if self.pick_scan_name_entry.get() == "":
            self.cannot_scan_label.configure(text="Enter a save name!")
            return
        self.save_name = self.pick_scan_name_entry.get()
        self.currently_scanning = True
        self.cannot_scan_label.configure(text="")
        self.currently_scanning_label.configure(text="Scanning...")
        os.system("cd d:/Desktop/GUI && bash < start_scan.sh")
    
    def stop_scan(self):
        if self.currently_scanning == False:
            return
        self.currently_scanning = False
        self.currently_scanning_label.configure(text="Scan complete!")
        self.cannot_scan_label.configure(text="")
        os.system("cd d:/Desktop/GUI && bash < stop_scan.sh")
        os.system("bash < get_file.sh")
        os.rename("scan_data/untitled_data0", "scan_data/"+self.save_name)
        
    

class Settings(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        rowheight = 0.07
        relativex = 0.25
        
        label_font = ("Arial 25 bold")
        
        self.dT_0 = tk.Label(self, text="dt_0", font=label_font, background="azure2")
        self.dT_0.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.11)
        self.range_start = tk.Label(self, text="range_start", font=label_font, background="azure2")
        self.range_start.place(anchor="w", relheight=rowheight, relx = relativex, rely=0.22)
        self.range_stop = tk.Label(self, text="range_stop", font=label_font, background="azure2")
        self.range_stop.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.33)
        self.tx_gain_ind = tk.Label(self, text="tx_gain_ind", font=label_font, background="azure2")
        self.tx_gain_ind.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.44)
        self.pii = tk.Label(self, text="pii", font=label_font, background="azure2")
        self.pii.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.55)
        self.code_channel = tk.Label(self, text="code_channel", font=label_font, background="azure2")
        self.code_channel.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.66)
        
        relativex_entry = 0.65
        buttonwidth = 0.1
        entryfont=("Arial 15")
        
        settings = open("radar_settings.txt", "r")
        
        self.dT_0_entry = tk.Entry(self, font=entryfont)
        self.dT_0_entry.insert(0, settings.readline()[5:].rstrip())
        self.dT_0_entry.place(anchor="w", relheight=rowheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.11)
        self.range_start_entry = tk.Entry(self, font=entryfont)
        self.range_start_entry.insert(0, settings.readline()[12:].rstrip())
        self.range_start_entry.place(anchor="w", relheight=rowheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.22)
        self.range_stop_entry = tk.Entry(self, font=entryfont)
        self.range_stop_entry.insert(0, settings.readline()[11:].rstrip())
        self.range_stop_entry.place(anchor="w", relheight=rowheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.33)
        self.tx_gain_ind_entry = tk.Entry(self, font=entryfont)
        self.tx_gain_ind_entry.insert(0, settings.readline()[12:].rstrip())
        self.tx_gain_ind_entry.place(anchor="w", relheight=rowheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.44)
        self.pii_entry = tk.Entry(self, font=entryfont)
        self.pii_entry.insert(0, settings.readline()[4:].rstrip())
        self.pii_entry.place(anchor="w", relheight=rowheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.55)
        self.code_channel_entry = tk.Entry(self, font=entryfont)
        self.code_channel_entry.insert(0, settings.readline()[13:].rstrip())
        self.code_channel_entry.place(anchor="w", relheight=rowheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.66)
        
        settings.close()
        
        self.Update_button = tk.Button(self, text="Update settings", command=self.update_settings, font=("Arial 15"))
        self.Update_button.place(anchor="w", relheight=0.1, relwidth=0.2, relx=0.4, rely=0.83)
        
    def update_settings(self):
        file = open("radar_settings.txt", "w")
        file.write("dT_0=" + self.dT_0_entry.get() + "\n")
        file.write("range_start=" + self.range_start_entry.get() + "\n")
        file.write("range_stop=" + self.range_stop_entry.get() + "\n")
        file.write("tx_gain_ind=" + self.tx_gain_ind_entry.get() + "\n")
        file.write("pii=" + self.pii_entry.get() + "\n")
        file.write("code_channel=" + self.code_channel_entry.get() + "\n")
        file.close()
        os.system("cd d:/Desktop/GUI && bash < update_settings.sh")

class Unpack(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.directory = tk.Entry(self, text="", font=("10"))
        self.directory.place(anchor="w", relwidth=0.5, relheight=0.05, relx=0.17, rely=0.04)
        self.choose_directory_button = tk.Button(self, text="Choose directory", command=self.choose_directory, font=("Arial 12"))
        self.choose_directory_button.place(anchor="w", relwidth=0.16, relheight=0.05, relx=0.67, rely=0.04)
        
        self.unpack_button = tk.Button(self, text="Unpack!", command=self.unpack)
        self.unpack_button.place(anchor="w", relwidth=0.1, relheight=0.05, relx=0.45, rely=0.11)
        
        self.image_label = tk.Label(self, text="Image\ngoes here", font=("Arial 60"), relief="solid", borderwidth=3)
        self.image_label.place(anchor="w", relwidth=0.7, relheight=0.7, relx=0.15, rely=0.55)
        
    
    def choose_directory(self):
        name = filedialog.askopenfilename()
        self.directory.delete(0, "end")
        self.directory.insert(0, name)
        
    def unpack(self):
        name = self.directory.get()
        figure = main(['-f', name, '-v'])
        canvas = FigureCanvasTkAgg(figure, self)
        canvas.draw()
        canvas.get_tk_widget().place(anchor="w", relwidth=0.7, relheight=0.7, relx=0.15, rely=0.55)
        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        #image.show()
        #os.system("python D:\Desktop\GUI\pulson440_unpack.py -f " + name + " -v")
        
        
    
class Image(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.directory = tk.Entry(self, text="", font=("10"))
        self.directory.place(anchor="w", relwidth=0.5, relheight=0.05, relx=0.17, rely=0.04)
        self.choose_directory_button = tk.Button(self, text="Choose directory", command=self.choose_directory, font=("Arial 12"))
        self.choose_directory_button.place(anchor="w", relwidth=0.16, relheight=0.05, relx=0.67, rely=0.04)
        
        rowheight = 0.07
        relativex = 0.25
        
        label_font = ("Arial 25 bold")
        
        self.DT_0_label = tk.Label(self, text="DT_0", font=label_font, background="azure2")
        self.DT_0_label.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.11)
        self.pulse_data_label = tk.Label(self, text="Pulse data", font=label_font, background="azure2")
        self.pulse_data_label.place(anchor="w", relheight=rowheight, relx = relativex, rely=0.22)
        self.platform_position_data_label = tk.Label(self, text="Platform position data", font=label_font, background="azure2")
        self.platform_position_data_label.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.33)
        self.given_object_label = tk.Label(self, text="Given object", font=label_font, background="azure2")
        self.given_object_label.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.44)
        self.size_label = tk.Label(self, text="Size", font=label_font, background="azure2")
        self.size_label.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.55)
        self.eyeballing_time_start_label = tk.Label(self, text="Eyeballing time start", font=label_font, background="azure2")
        self.eyeballing_time_start_label.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.66)
        
        """
        self.image_button = tk.Button(self, text="Create image!", command=make_image)
        self.image_button.place(anchor="w", relwidth=0.1, relheight=0.05, relx=0.45, rely=0.11)
        """
    def choose_directory(self):
        name = filedialog.askopenfilename()
        self.directory.delete(0, "end")
        self.directory.insert(0, name)
    
    #def make_image():
        

app = GUI()
app.mainloop()