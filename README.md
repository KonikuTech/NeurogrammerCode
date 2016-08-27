# NeurogrammerCode

Contains all the files you will need for the Neurogrammer developer conference.  

All .hoc and .mod files are to be used with the NEURON modeling environment.  HOC files (High-order compiler) files are to be compiled and run after executing the nrngui command from your terminal, after a successful installation of the most recent version of NEURON (http://www.neuron.yale.edu/neuron/).  MOD files must be compiled first before they can be run.  The simple way to do this is to open two folder (path) windows, one with the MOD file, and one with the installation of the NEURON program.  Inside the neuron folder, locate the program entitled mknrndll.  Simply drag the MOD file to this program, and drop it in.  The NEURON environment will compile it for you.  A new folder should now appear in the same directory as where the original MOD file was located.  You are now ready to go!

The .py file is to be used with the STICK model presentation.  It contains all the functions delinated in the STICK paper.  You will need two additional python libraries for this program to work, numpy and matplotlib.  A quick google search should reveal download links for your operating system.  In order to run this program, simply go to a terminal (command line) window and type 

python STICK.py

The program will now run.
