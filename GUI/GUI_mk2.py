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
from final_script_gui import Script

class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("UAS-SAR GUI")
        
        self.background_color = "gray20"
        self.foreground_color = "gray30"
        self.text_color = "white"
        self.button_color = "DodgerBlue2"
        
        tab_font = ("arial 40 bold")
        
        width = 1700
        height = 900
        x = int((self.winfo_screenwidth()-width)/2)
        y = int((self.winfo_screenheight()-height)/3)
        self.geometry("%dx%d+%d+%d" % (width, height, x, y))
        
        
        menu = tk.Frame(self)
        
        scan_tab = tk.Label(menu, text="Scan", font=tab_font, background=self.foreground_color, fg="gray90", relief="raised")
        scan_tab.bind("<Button-1>", lambda e: self.show_frame(Scan, self.current, scan_tab))
        scan_tab.pack(side="top", fill="both", expand=True)
        settings_tab = tk.Label(menu, text="Settings", font=tab_font, background=self.foreground_color, fg="gray90", relief="raised")
        settings_tab.bind("<Button-1>", lambda e: self.show_frame(Settings, self.current, settings_tab))
        settings_tab.pack(side="top", fill="both", expand=True)
        unpack_tab = tk.Label(menu, text="Unpack", font=tab_font, background=self.foreground_color, fg="gray90", relief="raised")
        unpack_tab.bind("<Button-1>", lambda e: self.show_frame(Unpack, self.current, unpack_tab))
        unpack_tab.pack(side="top", fill="both", expand=True)
        image_tab = tk.Label(menu, text="Image", font=tab_font, background=self.foreground_color, fg="gray90", relief="raised")
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
            frame.configure(background=self.background_color)
        
        self.show_frame(Scan, self.current, scan_tab)
        
    def show_frame(self, cont, current_tab, new_tab):
        frame = self.frames[cont]
        frame.tkraise()
        current_tab.configure(background=self.foreground_color, fg="gray90", relief="raised")
        new_tab.configure(background=self.background_color, fg="gray90", relief="flat")
        self.current = new_tab
    
    def get_Unpack(self):
        return self.frames[Unpack]
    
    
    def get_color(self):
        return self.background_color
    
    def get_text_color(self):
        return self.text_color
    
    def get_button_color(self):
        return self.button_color


class Scan(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        button_font = ("arial 15")
        self.background_color = self.master.master.get_color()
        self.text_color = self.master.master.get_text_color()
        self.button_color = self.master.master.get_button_color()
        
        self.pick_a_wifi = tk.Label(self, text="Pick a wifi network:", font=("arial 20"), background=self.background_color, fg=self.text_color)
        self.pick_a_wifi.place(anchor="w", relheight=0.2, relwidth=0.5, rely=0.1, relx=0.25)
        
        self.UASSAR3_button = tk.Button(self, text="UASSAR3", font=button_font, bg=self.button_color, command=lambda: self.connect("UASSAR3", self.label_1))
        self.UASSAR3_button.place(anchor="w", relheight=0.1, relwidth=0.15, rely=0.25, relx=0.08)
        self.MIT_button = tk.Button(self, text="MIT", font=button_font, bg=self.button_color, command=lambda: self.connect("MIT", self.label_2))
        self.MIT_button.place(anchor="w", relheight=0.1, relwidth=0.15, rely=0.25, relx=0.31)
        self.MIT_GUEST_button = tk.Button(self, text="MIT GUEST", font=button_font, bg=self.button_color, command=lambda: self.connect("MIT GUEST", self.label_3))
        self.MIT_GUEST_button.place(anchor="w", relheight=0.1, relwidth=0.15, rely=0.25, relx=0.54)
        self.Hyatt_wifi_button = tk.Button(self, text="@Hyatt_WiFi", font=button_font, bg=self.button_color, command=lambda: self.connect("@Hyatt_WiFi", self.label_4))
        self.Hyatt_wifi_button.place(anchor="w", relheight=0.1, relwidth=0.15, rely=0.25, relx=0.77)
        self.label_1 = tk.Label(self, text="", background=self.background_color, fg=self.text_color, font="10")
        self.label_1.place(anchor="w", relheight=0.03, relwidth=0.15, rely=0.32, relx=0.08)
        self.label_2 = tk.Label(self, text="", background=self.background_color, fg=self.text_color, font="10")
        self.label_2.place(anchor="w", relheight=0.03, relwidth=0.15, rely=0.32, relx=0.31)
        self.label_3 = tk.Label(self, text="", background=self.background_color, fg=self.text_color, font="10")
        self.label_3.place(anchor="w", relheight=0.03, relwidth=0.15, rely=0.32, relx=0.54)
        self.label_4 = tk.Label(self, text="", background=self.background_color, fg=self.text_color, font="10")
        self.label_4.place(anchor="w", relheight=0.03, relwidth=0.15, rely=0.32, relx=0.77)
        
        self.current = self.label_2
        
        self.scan_start_button = tk.Button(self, text="Start scan", command=self.start_scan, font=button_font, background="chartreuse")
        self.scan_start_button.place(anchor="w", relheight=0.15, relwidth=0.2, rely=0.75, relx=0.15)
        self.scan_stop_button = tk.Button(self, text="Stop scan", command=self.stop_scan, font=button_font, background="orange red2")
        self.scan_stop_button.place(anchor="w", relheight=0.15, relwidth=0.2, rely=0.75, relx=0.65)
        self.currently_scanning_label = tk.Label(self, text="", font=("Arial 25 italic"), background=self.background_color, fg=self.text_color)
        self.currently_scanning_label.place(anchor="w", relheight=0.15, relwidth=0.3, rely=0.75, relx=0.35)
        self.cannot_scan_label = tk.Label(self, text="", font="10", background=self.background_color, fg=self.text_color)
        self.cannot_scan_label.place(anchor="w", relheight=0.05, relwidth=0.2, rely=0.85, relx=0.15)
        
        self.currently_scanning = False
        
        self.pick_scan_name_label = tk.Label(self, text="Scan data file name:", font="Arial 17", background=self.background_color, fg=self.text_color)
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
        
        self.background_color = self.master.master.get_color()
        self.text_color = "#3F95E8" #self.master.master.get_text_color()
        self.button_color = self.master.master.get_button_color()
        
        rowheight = 0.08
        relativex = 0.27
        
        label_font = ("Arial 35 bold")
        
        self.dT_0 = tk.Label(self, text="dt_0", font=label_font, background=self.background_color, fg=self.text_color)
        self.dT_0.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.11)
        self.range_start = tk.Label(self, text="range_start", font=label_font, background=self.background_color, fg=self.text_color)
        self.range_start.place(anchor="w", relheight=rowheight, relx = relativex, rely=0.22)
        self.range_stop = tk.Label(self, text="range_stop", font=label_font, background=self.background_color, fg=self.text_color)
        self.range_stop.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.33)
        self.tx_gain_ind = tk.Label(self, text="tx_gain_ind", font=label_font, background=self.background_color, fg=self.text_color)
        self.tx_gain_ind.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.44)
        self.pii = tk.Label(self, text="pii", font=label_font, background=self.background_color, fg=self.text_color)
        self.pii.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.55)
        self.code_channel = tk.Label(self, text="code_channel", font=label_font, background=self.background_color, fg=self.text_color)
        self.code_channel.place(anchor="w", relheight=rowheight, relx=relativex, rely=0.66)
        
        relativex_entry = 0.66
        buttonwidth = 0.07
        entryfont=("Arial 20")
        entryheight = 0.07
        
        settings = open("radar_settings.txt", "r")
        
        self.dT_0_entry = tk.Entry(self, font=entryfont)
        self.dT_0_entry.insert(0, settings.readline()[5:].rstrip())
        self.dT_0_entry.place(anchor="w", relheight=entryheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.11)
        self.range_start_entry = tk.Entry(self, font=entryfont)
        self.range_start_entry.insert(0, settings.readline()[12:].rstrip())
        self.range_start_entry.place(anchor="w", relheight=entryheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.22)
        self.range_stop_entry = tk.Entry(self, font=entryfont)
        self.range_stop_entry.insert(0, settings.readline()[11:].rstrip())
        self.range_stop_entry.place(anchor="w", relheight=entryheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.33)
        self.tx_gain_ind_entry = tk.Entry(self, font=entryfont)
        self.tx_gain_ind_entry.insert(0, settings.readline()[12:].rstrip())
        self.tx_gain_ind_entry.place(anchor="w", relheight=entryheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.44)
        self.pii_entry = tk.Entry(self, font=entryfont)
        self.pii_entry.insert(0, settings.readline()[4:].rstrip())
        self.pii_entry.place(anchor="w", relheight=entryheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.55)
        self.code_channel_entry = tk.Entry(self, font=entryfont)
        self.code_channel_entry.insert(0, settings.readline()[13:].rstrip())
        self.code_channel_entry.place(anchor="w", relheight=entryheight, relwidth=buttonwidth, relx=relativex_entry, rely=0.66)
        
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
        
        self.background_color = self.master.master.get_color()
        self.text_color = "#3F95E8" #self.master.master.get_text_color()
        self.button_color = self.master.master.get_button_color()
        
        self.relativeheight = 0.1
        self.relativewidth = 0.1
        self.relativey = 0.3
        self.yconstant = 0.1
        self.font = "Arial 15"
        
        self.directory = tk.Entry(self, text="", font=("10"))
        self.directory.place(anchor="w", relwidth=0.5, relheight=0.05, relx=0.17, rely=0.04)
        self.choose_directory_button = tk.Button(self, text="Choose scan data", command=lambda: self.choose_directory(self.directory), font=("Arial 12"))
        self.choose_directory_button.place(anchor="w", relwidth=0.16, relheight=0.05, relx=0.67, rely=0.04)
        self.directory2 = tk.Entry(self, text="", font=("10"))
        self.directory2.place(anchor="w", relwidth=0.5, relheight=0.05, relx=0.17, rely=0.1)
        self.choose_directory_button2 = tk.Button(self, text="Choose MoCap data", command=lambda: self.choose_directory(self.directory2), font=("Arial 12"))
        self.choose_directory_button2.place(anchor="w", relwidth=0.16, relheight=0.05, relx=0.67, rely=0.1)
        self.directory3 = tk.Entry(self, text="", font=("10"))
        self.directory3.place(anchor="w", relwidth=0.5, relheight=0.05, relx=0.17, rely=0.16)
        self.choose_directory_button3 = tk.Button(self, text="Choose reference data", command=lambda: self.choose_directory(self.directory3), font=("Arial 12"))
        self.choose_directory_button3.place(anchor="w", relwidth=0.16, relheight=0.05, relx=0.67, rely=0.16)
        
        self.unpack_button = tk.Button(self, text="Unpack!", command=self.unpack)
        self.unpack_button.place(anchor="w", relwidth=0.1, relheight=0.05, relx=0.865, rely=0.1)
        
        self.image_label = tk.Label(self, text="Image\ngoes here", font=("Arial 60"), relief="solid", borderwidth=3)
        self.image_label.place(anchor="w", relwidth=0.7, relheight=0.8, relx=0.001, rely=0.6)
        
        self.upper_index = 0
        self.lower_index = 0
        
        self.upper_index_label = tk.Label(self, text="Upper index:", font=self.font, bg=self.background_color, fg=self.text_color)
        self.upper_index_label.place(anchor="w", relwidth=self.relativewidth, relheight=self.relativeheight, relx=0.73, rely=self.relativey)
        self.lower_index_label = tk.Label(self, text="Lower index:", font=self.font, bg=self.background_color, fg=self.text_color)
        self.lower_index_label.place(anchor="w", relwidth=self.relativewidth, relheight=self.relativeheight, relx=0.73, rely=self.relativey+self.yconstant)
        self.time_offset_label = tk.Label(self, text="Time offset:", font=self.font, bg=self.background_color, fg=self.text_color)
        self.time_offset_label.place(anchor="w", relwidth=self.relativewidth, relheight=self.relativeheight, relx=0.73, rely=self.relativey+2*self.yconstant)
        self.range_offset_label = tk.Label(self, text="Range offset:", font=self.font, bg=self.background_color, fg=self.text_color)
        self.range_offset_label.place(anchor="w", relwidth=self.relativewidth, relheight=self.relativeheight, relx=0.73, rely=self.relativey+3*self.yconstant)
        self.upper_index_entry = tk.Entry(self, font="Arial 13")
        self.upper_index_entry.bind("<Return>", lambda e: self.update_upper_line())
        self.upper_index_entry.place(anchor="w", relwidth=self.relativewidth, relheight=self.relativeheight/2, relx=0.85, rely=self.relativey)
        self.lower_index_entry = tk.Entry(self, font="Arial 13")
        self.lower_index_entry.bind("<Return>", lambda e: self.update_lower_line())
        self.lower_index_entry.place(anchor="w", relwidth=self.relativewidth, relheight=self.relativeheight/2, relx=0.85, rely=self.relativey+self.yconstant)
        self.time_offset_entry = tk.Entry(self, font="Arial 13")
        self.time_offset_entry.place(anchor="w", relwidth=self.relativewidth, relheight=self.relativeheight/2, relx=0.85, rely=self.relativey+2*self.yconstant)
        self.range_offset_entry = tk.Entry(self, font="Arial 13")
        self.range_offset_entry.place(anchor="w", relwidth=self.relativewidth, relheight=self.relativeheight/2, relx=0.85, rely=self.relativey+3*self.yconstant)
        
    
    def choose_directory(self, directory_entry):
        name = filedialog.askopenfilename()
        directory_entry.delete(0, "end")
        directory_entry.insert(0, name)
        
    def get_directory(self):
        return self.directory.get()
        
    def unpack(self):
        name = self.directory.get()
        self.f, self.ax, self.img = main(['-f', name, '-v'])
        
        self.data = Script(self.directory.get(), self.directory2.get(), self.directory3.get())
        
        self.edge_x, self.edge_y = self.data.get_graph()
        
        self.numrows, self.numcol = self.img.get_size()
        self.upper_index = 100
        self.lower_index = self.numrows-100
        self.upper_index_entry.insert(0, str(self.upper_index))
        self.lower_index_entry.insert(0, str(self.lower_index))
        self.upper_line = self.ax.plot(range(self.numcol), [self.upper_index]*self.numcol, 'w')
        self.lower_line = self.ax.plot(range(self.numcol), [self.lower_index]*self.numcol, 'k')
        self.edge_line = self.ax.plot(self.edge_x, self.edge_y)
        
        
        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(anchor="w", relwidth=0.7, relheight=0.8, relx=0.001, rely=0.6)
        self.canvas.get_tk_widget().bind("<Up>", lambda e: self.upper_line_up())
        self.canvas.get_tk_widget().bind("<Down>", lambda e: self.upper_line_down())
        self.canvas.get_tk_widget().bind("<Shift-Up>", lambda e: self.lower_line_up())
        self.canvas.get_tk_widget().bind("<Shift-Down>", lambda e: self.lower_line_down())
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.update()
        
    def update_upper_line(self):
        self.upper_line.pop().remove()
        self.upper_index = self.upper_index_entry.get()
        self.upper_line = self.ax.plot(range(self.numcol), [self.upper_index]*self.numcol, 'w')
        self.canvas.draw()
        
    def update_lower_line(self):
        self.lower_line.pop().remove()
        self.lower_index = self.lower_index_entry.get()
        self.lower_line = self.ax.plot(range(self.numcol), [self.lower_index]*self.numcol, 'w')
        self.canvas.draw()
        
    def upper_line_up(self):
        if self.upper_index <= 5:
            return
        self.upper_line.pop().remove()
        self.upper_index = self.upper_index - 15
        self.upper_line = self.ax.plot(range(self.numcol), [self.upper_index]*self.numcol, 'w')
        self.canvas.draw()
        self.upper_index_entry.delete(0, "end")
        self.upper_index_entry.insert(0, str(self.get_upper_index()))
        
    def upper_line_down(self):
        if self.upper_index >= self.numrows-5:
            return
        self.upper_line.pop().remove()
        self.upper_index = self.upper_index + 15
        self.upper_line = self.ax.plot(range(self.numcol), [self.upper_index]*self.numcol, 'w')
        self.canvas.draw()
        self.upper_index_entry.delete(0, "end")
        self.upper_index_entry.insert(0, str(self.get_upper_index()))
        
    def lower_line_up(self):
        if self.lower_index <= 5:
            return
        self.lower_line.pop().remove()
        self.lower_index = self.lower_index - 15
        self.lower_line = self.ax.plot(range(self.numcol), [self.lower_index]*self.numcol, 'k')
        self.canvas.draw()
        self.lower_index_entry.delete(0, "end")
        self.lower_index_entry.insert(0, str(self.get_lower_index()))
        
    def lower_line_down(self):
        if self.lower_index >= self.numrows-5:
            return
        self.lower_line.pop().remove()
        self.lower_index = self.lower_index + 15
        self.lower_line = self.ax.plot(range(self.numcol), [self.lower_index]*self.numcol, 'k')
        self.canvas.draw()
        self.lower_index_entry.delete(0, "end")
        self.lower_index_entry.insert(0, str(self.get_lower_index()))
        
    def get_upper_index(self):
        return self.upper_index
    
    def get_lower_index(self):
        return self.lower_index
    
    def get_time_offset(self):
        return self.time_offset_entry.get()
    
    def get_range_offset(self):
        return self.range_offset_entry.get()
    
    def get_data(self):
        return self.data
        
class Image(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        
        self.background_color = self.master.master.get_color()
        self.text_color = "#3F95E8" #self.master.master.get_text_color()
        self.button_color = self.master.master.get_button_color()
        
        rowheight = 0.05
        relativex = 0.05
        relativey = 0.04
        yconstant = 0.06
        
        label_font = ("Arial 20 bold")
        
        self.size_label = tk.Label(self, text="Size", font=label_font, background=self.background_color, fg=self.text_color)
        self.size_label.place(anchor="w", relheight=rowheight, relx=relativex, rely=relativey)
        self.meters_label = tk.Label(self, text="Meters", font=label_font, background=self.background_color, fg=self.text_color)
        self.meters_label.place(anchor="w", relheight=rowheight, relx=relativex, rely=relativey+yconstant)
        
        
        self.size_entry = tk.Entry(self, font="Arial 20")
        self.size_entry.place(anchor="w", relheight=rowheight, relwidth=0.1, relx=0.4, rely=relativey)
        self.meters_entry = tk.Entry(self, font="Arial 20")
        self.meters_entry.place(anchor="w", relheight=rowheight, relwidth=0.1, relx=0.4, rely=relativey+yconstant)
        
        self.back_projection_button = tk.Button(self, text="Back-Projection", command=self.back_projection, font=("Arial 18"))
        self.back_projection_button.place(anchor="w", relwidth=0.17, relheight=0.1, relx=0.815, rely=0.3)
        self.entropy_button = tk.Button(self, text="Entropy", font=("Arial 18"), command=self.entropy_method)
        self.entropy_button.place(anchor="w", relwidth=0.17, relheight=0.1, relx=0.815, rely=0.45)
        
        
        self.image_label = tk.Label(self, text="Image\ngoes here", font=("Arial 60"), relief="solid", borderwidth=3)
        self.image_label.place(anchor="w", relwidth=0.8, relheight=0.85, relx=0.001, rely=0.57)
        
    def entropy_method(self):
        self.meters = self.get_meters()
        self.size = self.get_size()
        self.eyeballing_start_time = self.get_upper()
        self.eyeballing_end_time = self.get_lower()
        self.time_offset = self.get_time_offset()
        self.range_offset = self.get_range_offset()
        
        data = self.master.master.get_Unpack().get_data()
        f, ax, img = data.main_entropy(self.meters, self.size, self.eyeballing_start_time, self.eyeballing_end_time, self.time_offset, self.range_offset)
        self.canvas=FigureCanvasTkAgg(f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(anchor="w", relwidth=0.8, relheight=0.85, relx=0.001, rely=0.57)

        
    def back_projection(self):
        self.meters = self.get_meters()
        self.size = self.get_size()
        self.eyeballing_start_time = self.get_upper()
        self.eyeballing_end_time = self.get_lower()
        self.time_offset = self.get_time_offset()
        self.range_offset = self.get_range_offset()
        data = self.master.master.get_Unpack().get_data()
        f, ax, img = data.main_func(self.meters, self.size, self.eyeballing_start_time, self.eyeballing_end_time, self.time_offset, self.range_offset)
        self.canvas = FigureCanvasTkAgg(f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(anchor="w", relwidth=0.8, relheight=0.85, relx=0.001, rely=0.57)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.update()

        
    def choose_directory(self, widget):
        name = filedialog.askopenfilename()
        widget.delete(0, "end")
        widget.insert(0, name)
    
    def get_upper(self):
        return int(self.master.master.get_Unpack().get_upper_index())
    
    def get_lower(self):
        return int(self.master.master.get_Unpack().get_lower_index())
    
    def get_meters(self):
        return int(self.meters_entry.get())
    
    def get_size(self):
        return int(self.size_entry.get())
    
    def get_time_offset(self):
        return float(self.master.master.get_Unpack().get_time_offset())
    
    def get_range_offset(self):
        return float(self.master.master.get_Unpack().get_range_offset())
    
    #def make_image():
        

app = GUI()
app.mainloop()