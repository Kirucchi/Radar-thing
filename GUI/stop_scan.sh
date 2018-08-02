#!/bin/bash

sshpass -p "LDonovan" ssh pi@192.168.2.1 /bin/bash << 'EOT'

cd PulsOn_Code_New

echo 1 > control

exit
EOT
