#!/bin/bash

sshpass -p "LDonovan" scp -r pi@192.168.2.1:/home/pi/PulsOn_Code_New/untitled_data0 /mnt/d/Desktop/GUI/scan_data

sshpass -p "LDonovan" ssh pi@192.168.2.1 << 'EOT'

cd PulsOn_Code_New
rm untitled_data0

EOT
