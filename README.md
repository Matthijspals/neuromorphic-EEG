# Neuromorphic EEG classification for energy efficient brain-computer interfaces
Code for creating a deep learning based spiking EEG classifier, accompanying: link to paper.

In order to to run the whole pipeline:
1. Use train_CNNs.ipynb to train CNNs on Graz IV 2b (http://www.bbci.de/competition/iv/, load with moabb)
2. Use convert_all.py to convert the CNNs to their spiking equivalent
3. Use runSCNN.py to run the spiking neural networks

Note: if you don't have access to a SpiNNaker you can run the model in Nest

The code is tested with python 3.8.3 and the following package versions:

tensorflow.keras == 2.4.0 \
tensorflow == 2.3.0 \
scikit-learn == 0.23.1 \
scipy ==  1.6.1 \
numpy  ==  1.16.1 \
opencv-python == 4.5.1.48 \
mne == 0.22.0 (https://mne.tools/stable/index.html) \
moabb == 0.2.1 (https://github.com/NeuroTechX/moabb) \
braindecode == 0.5 (https://github.com/braindecode/braindecode) \
pynn == 0.9.2 (https://neuralensemble.org/PyNN/) \
pynest == 2.20.0 (https://www.nest-simulator.org/) \
snntoolbox == 0.5.0 (https://snntoolbox.readthedocs.io/en/latest/guide/installation.html) \

For snntoolbox, if you want to use SpatialDropout2D, you have to add this to the layers to be 
ignored in the conversion process (on snntoolbox/parsing/utils.py, line 450).

When using the SpiNNaker chip, if you get a decrease in accuracy due to lost packages,
you can try to increase the spikes_per_second parameter in the SpiNNaker config file.
