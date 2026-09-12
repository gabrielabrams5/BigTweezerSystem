# Magnetoacoustic Microrobotic Manipulation System
This system is a magnetic and acoustic microrobotic manipulation device. Its intention is to be a user friendly, portable, micro-robotic
experimentation platform. It contains an embedded single board computer, a microscope, power supplies, power
amplifiers, and control circuitry necessary for generating the complex magnetic fields neccesary for actuating a variety of micro robots. Custom
control software is written in Python and C++ for handling live image feed from a microscope camera, custom tracking
and detection algorthims, and outputting the control signals to the electromagnetic coils and acoustic transducers.
Furthermore, the system is the first of its kinda to combine both magnetic and acoustic actuation
methods into a single, stand alone, portable device. Many of the mechanical components can be 3D
printed allowing others to build the device at a low cost. The device can be divided into three
sections (Mechanical, Electrical, and Software) 

# Importing Excel Actions
A predefined excel file with desired action commands can be imported into the system and applied. The software will begin executing each entry (or row) from the excel file once the "Apply Excel Actions" button is pressed. Each row of actions are applied at each subsequent frame from the camera. 

# Instructions for initial installation of system components on Nvidia Jetson AGX Orin:
Note*** Software and system also works on Windows and MAC-OS. Jetson Orin System not neccesary.

1) need to configure nvme ssd using nvidia sdkmanager:  
    - https://developer.nvidia.com/embedded/learn-get-started-jetson-agx-orin-devkit

2) need to build opencv with cuda support: 
    - https://github.com/mdegans/nano_build_opencv.git

3) need to add permissions in order to read and write to the arduino port: 
    - add this to /etc/rc.local to execute on boot: $ chmod 666 /dev/ttyACM0

4)  need to install qt5
    - sudo apt install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools  
    - sudo apt install qt5-default

5) need to install Spinnaker FLIR camera SDK and python API: 
    - https://flir.app.boxcn.net/v/SpinnakerSDK/file/1093743440079
    - may need: sudo apt-mark manual qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools for spinview 

6) need to install all python dependencies
    - python3.8 -m pip install -r JetsonOrinReqs.txt

7) need to add "self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)" above self.cam.BeginAcquistion() line in $ .local/lib/python3.8/site-packages/EasyPySpin.videocapture.py

8) need to change in lib/python3.8/site-packages/adafruit_blinka/microcontroller/tegra/t234/pin.py from "GPIO.setmode(GPIO.TEGRA_SOC)" to GPIO.setmode(GPIO.BOARD)
    - otherwise the acoustic class and hall effect class will clash

9) need to install xboxdrv and jstest-gtk for joystick implimentation 
        $ sudo apt-get install -y xboxdrv         
        "https://github.com/FRC4564/Xbox"
        
10) VSCode: https://github.com/JetsonHacksNano/installVSCode.git

11) optional: install arduino using jetsonhacks github and upload main.ino from src/arduino

pyuic5 uis/GUI.ui -o gui_widgets.py

/opt/homebrew/bin/python3.10 -m PyInstaller --onedir --windowed --icon MagScopeBox.icns --name MagScope main.py

