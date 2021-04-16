import numpy as np
import warnings;
warnings.filterwarnings('ignore');
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import os
import pickle
from six.moves import cPickle
from IPython.display import clear_output

# if you don't have access to a SpiNNaker chip,
# you can set 'spin' to False and run the model
# in Nest

spin=True
if spin:
    import pyNN.spiNNaker as pynn
else:
    import pyNN.nest as pynn

#set sim parameters
participants = np.arange(1,10)
n_classes = 2        # number of motor imagery classes
plot = False         # run 1 trial and generate spike plot
sim_time = 50        # simulation time
dt = 0.1             # simulation timestep
refrac = dt          # refractory period of neuron
weight_scale = 100   # scale weights, to account for increased threshold on SpiNNaker
rescale_fac = 0.1/dt # rescale input

# LIF parameters
cell_params = {
    'v_thresh' : 1,
    'tau_refrac' : refrac,
    'v_reset' : 0,
    'v_rest' : 0,
    'cm' : 1,
    'tau_m' : 1000,
    'tau_syn_E' : 0.01,
    'tau_syn_I' : 0.01
} 

#slightly modified load function from snn toolbox
def load_assembly(path, filename, sim):
    """Load the populations in an assembly """

    filepath = os.path.join(path, filename)
    assert os.path.isfile(filepath), \
        "Spiking neuron layers were not found at specified location."
    if sys.version_info < (3,):
        s = cPickle.load(open(filepath, 'rb'))
    else:
        s = cPickle.load(open(filepath, 'rb'), encoding='bytes')

    # Iterate over populations in assembly
    layers = []
    n_neurons = []
    for label in s['labels']:
        celltype = getattr(sim, s[label]['celltype'])
        n_neurons.append(s[label]['size'])
        population = sim.Population(s[label]['size'], celltype,
                                            celltype.default_parameters,
                                            structure=s[label]['structure'],
                                            label=label)
        # Set the rest of the specified variables, if any.
        for variable in s['variables']:
            if getattr(population, variable, None) is None:
                setattr(population, variable, s[label][variable])
        if label != 'InputLayer':
            population.set(i_offset=s[label]['i_offset'])
        layers.append(population)
  
    return layers, s['labels'], n_neurons

# run for every participant and every fold
accs_all = []
stds_all = []
for part in participants:
    print("running particpant: " +str(part))
    allfoldacc=[]
    for fold in np.arange(0,5):
        pynn.setup(dt, min_delay=dt)
        if spin:
            pynn.set_number_of_neurons_per_core(pynn.IF_curr_exp, 64)

        path = "models/subject_"+str(part)+"/fold"+str(fold)
    
        # load data
        test_data = np.load(path+"/x_test.npz")['arr_0']
        test_labels = np.load(path+"/y_test.npz")['arr_0']
        pred_labels = []
        num_test=len(test_data)
        
        # create network
        network, labels, n_neurons = load_assembly(path, "tsfold"+str(fold)+"_nest", pynn)
        
        # load connections
        for l in range(len(network)-1):
            conn = np.genfromtxt(path + "/" + labels[l+1])
            # when we switch to spinnaker we need to split the connection in ex and inh:
            ex = conn[conn[:,2]>0]
            inh = conn[conn[:,2]<0]
            if ex.any():
                ex[:,2] *= weight_scale
                pynn.Projection(network[l], network[l+1], \
                                pynn.FromListConnector(ex, ['weight', 'delay']), receptor_type='excitatory')
            if inh.any():
                inh[:,2] *= weight_scale
                pynn.Projection(network[l], network[l+1], \
                                pynn.FromListConnector(inh, ['weight', 'delay']), receptor_type='inhibitory')

            network[l + 1].set(**cell_params)
            network[l + 1].initialize(v=network[l + 1].get('v_rest'))

        # record spikes
        if plot:
            for layer in network:
                layer.record("spikes")
        else:
            network[-1].record("spikes")

        # run experiment
        for trial, j in enumerate(test_data):

            spiketrains_all=[]
            
            # set input
            x_flat = np.ravel(j)
            rates = 1000 * x_flat / rescale_fac
            network[0].set(rate=rates)
          
            # run simulation
            pynn.run(sim_time)

            # get spikes
            for ind, n in enumerate(n_neurons):
                shape = (n, int(sim_time/dt))
                spiketrains = network[ind].get_data(clear=True).segments[-1].spiketrains
                spiketrains_flat = np.zeros(shape)
                for k, spiketrain in enumerate(spiketrains):
                    for t in spiketrain:
                        spiketrains_flat[k, int(t / dt)] = 1

                # get spikes for plotting
                spiketrains_all.append(spiketrains_flat)
                
            # calculate model prediction
            spikesum = np.sum(spiketrains_flat, axis = 1)
            estimate = np.argmax(spikesum+np.random.randn(n_classes)*0.001)
            pred_labels.append(np.eye(n_classes)[estimate])
            clear_output(wait=True)
            print('part ' +str(part) + ' fold ' + str(fold) + ' trial ' +str(trial) + ' of ' +str(len(test_data)))
            print('estimate = ' + str(estimate))
            print('true = ' + str(test_labels[trial]))
            print(spikesum)
            
            #reset simulation
            pynn.reset()
            if plot:
                break

        #end simulation
        pynn.end()
  
        print('simulation end')
    
    
        #create spike plot
        if plot:
            s = []
            a = []
            i = 0
            colors = 'black'
            lineoffsets = 1
            linelengths = 2
            plt.gcf().subplots_adjust(bottom=0.25)

            fig, axs = plt.subplots(1,6, figsize=(40,6))
            color = 'lightblue'
            nfilters = [784,144,100,25,25,1]
            for spiketrain, n in zip(spiketrains_all, n_neurons):
                eventdata = []
                colors = []
                if i == 0:
                    for ne in range(392):
                        colors.append('#262424')
                    for ne in range(392):
                        colors.append('#898989')
                else:
                    for ne in range(n):
                        if ne%nfilters[i]==0:
                            if color=='#5e3c99':
                                color = '#e66101'
                            else:
                                color = '#5e3c99'
                        colors.append(color)

                for dat in spiketrain:
                    eventdata.append(np.nonzero(dat)[0])
                
                axs[i].eventplot(eventdata, colors=colors, lineoffsets=lineoffsets,
                                linelengths=linelengths, linewidths = 2.5)
                axs[i].set_xlim(0, int(sim_time/dt))
                axs[i].set_xticks(np.arange(0, int(sim_time/dt)+1, int(10/dt)))
                
                if i<5:
                    axs[i].set_yticks(np.arange(0, n+1, int(n/4)))
                    axs[i].set_yticklabels(np.arange(0, n+1, int(n/4)), fontsize = 20)
                else:
                    axs[i].set_yticks([0, 2])
                    axs[i].set_yticklabels([0,1], fontsize=20)
                
                axs[i].set_xticklabels([0,10,20, 30, 40, 50], fontsize=20)
                
                axs[i].set_ylim(-0.6, n+.6)   
                i+=1
            axs[0].set_ylabel("Neuron Index", fontsize=36)
            axs[0].set_xlabel("Time (ms)", fontsize=36)
            axs[5].set_ylim(-1.2,3.2)   
            plt.savefig("spikes_plot.svg")

            print('plot end')

        # calculate accuracy
        good_preds=0.0
        for p in range(len(pred_labels)):
            good_preds +=np.dot(pred_labels[p], test_labels[p])
        print("accuracy: "+str(good_preds/(p+1)))
        allfoldacc.append(good_preds/(p+1))
        if plot:
            break
    
    # calculate mean + std accuracy
    for i in range(len(allfoldacc)):
        allfoldacc[i]*=100
    print(allfoldacc)
    print("all fold acc: " + str(np.mean(allfoldacc)) + " +- " + str(np.std(allfoldacc)))
    accs_all.append(np.mean(allfoldacc))
    stds_all.append(np.std(allfoldacc))
    
print(accs_all)
print(stds_all)