import numpy as np
import nengo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nni
import pickle
from scipy.stats import ks_2samp

class Inputs():
    def __init__(self, deltaP, maxSamples, empirical=None, seed=0):
        self.deltaP = deltaP
        self.maxSamples = maxSamples
        self.empirical = empirical
        self.winning = None
        self.pA = None
        self.pB = None
        self.dP_actual = None
        self.As = []
        self.Bs = []
        self.rng = np.random.RandomState(seed=seed)
    def set_AB(self):
        self.winning = "A" if self.rng.rand()<0.5 else "B"
        highs = np.arange(0.1+self.deltaP, 0.9, 0.1)
        high = highs[self.rng.randint(len(highs))]
        low = high - self.deltaP
        nUpWin = int(high*self.maxSamples)
        nUpLose = int(low*self.maxSamples)
        self.As = np.zeros((self.maxSamples))
        self.Bs = np.zeros((self.maxSamples))
        if self.winning=="A":
            self.As[:nUpWin] = 1
            self.As[nUpWin:] = -1
            self.Bs[:nUpLose] = 1
            self.Bs[nUpLose:] = -1
            self.pA = nUpWin / self.maxSamples
            self.pB = nUpLose / self.maxSamples
        else:
            self.Bs[:nUpWin] = 1
            self.Bs[nUpWin:] = -1
            self.As[:nUpLose] = 1
            self.As[nUpLose:] = -1
            self.pB = nUpWin / self.maxSamples
            self.pA = nUpLose / self.maxSamples
        self.rng.shuffle(self.As)
        self.rng.shuffle(self.Bs)
        self.dP_actual = np.abs(self.pA - self.pB)
    def set_AB_empirical(self, trial, initialize=True):
        self.pA = self.empirical['pA'].to_numpy()[trial]
        self.pB = self.empirical['pB'].to_numpy()[trial]
        self.winning = "A" if self.pA>self.pB else "B"
        self.As = np.zeros((self.maxSamples))
        self.Bs = np.zeros((self.maxSamples))
        if initialize:  # populate the A and B arrays with the samples actually drawn in the empirical trial
            empAs = list(str(self.empirical['A'].to_numpy()[trial]))
            empAs = np.array([2*int(x)-1 for x in empAs])
            empBs = list(str(self.empirical['B'].to_numpy()[trial]))
            empBs = np.array([2*int(x)-1 for x in empBs])
            self.As[:len(empAs)] = empAs
            self.Bs[:len(empBs)] = empBs
            self.As[len(empAs):] = 2*self.rng.randint(2, size=self.maxSamples-len(empAs))-1
            self.Bs[len(empBs):] = 2*self.rng.randint(2, size=self.maxSamples-len(empBs))-1
        else:
            if self.winning=="A":
                self.As[:nUpWin] = 1
                self.As[nUpWin:] = -1
                self.Bs[:nUpLose] = 1
                self.Bs[nUpLose:] = -1
            else:
                self.Bs[:nUpWin] = 1
                self.Bs[nUpWin:] = -1
                self.As[:nUpLose] = 1
                self.As[nUpLose:] = -1
            self.rng.shuffle(self.As)
            self.rng.shuffle(self.Bs)
        self.dP_actual = np.abs(self.pA - self.pB)
    def get_AB(self, t):
        AB = [self.As[int(t)], self.Bs[int(t)]] if t<self.maxSamples else [0,0]
        AB = [AB[0], 0] if t%1.0<0.5 else [0, AB[1]]  # A presented for 500ms, then B presented for 500ms
        return AB


def build_network(inputs, nNeurons=1000, synapse=0.1, seed=0,
        tau=0, M=0.2, delta=0, T=0.3, S=[0,0]):
    net = nengo.Network(seed=seed)
    net.config[nengo.Connection].synapse = 0.03
    net.config[nengo.Probe].synapse = 0.03
    
    func_evidence = lambda t: inputs.get_AB(t)
    func_thr = lambda t: T
    func_start = lambda t: S if t<1 else [0,0]
    func_urgency = lambda t: -tau * t
    func_ramp = lambda x: synapse * M * x
    func_uncertainty = lambda x: -delta * np.abs(x[0]-x[1])
    func_greater = lambda x: [x[0]-x[1], x[1]-x[0]] 
    
    ePos = nengo.dists.Choice([[1]])
    iPos = nengo.dists.Uniform(0.01, 1)
    dInh = -1e1*np.ones((2*nNeurons, 1))

    with net:
        # Inputs
        evidence = nengo.Node(func_evidence)
        time = nengo.Node(func_urgency)
        thr = nengo.Node(func_thr)
        start = nengo.Node(func_start)
        
        # Ensembles
        value = nengo.networks.EnsembleArray(nNeurons, 2)
        accumulate = nengo.networks.EnsembleArray(nNeurons, 2)
        combined = nengo.Ensemble(2*nNeurons, 2)
        gate = nengo.Ensemble(nNeurons, 1, encoders=ePos, intercepts=iPos)
        action = nengo.networks.EnsembleArray(nNeurons, 2, encoders=ePos, intercepts=iPos)
        
        # Connections
        nengo.Connection(evidence, value.input)
        nengo.Connection(start[0], accumulate.ea_ensembles[0], synapse=synapse, function=func_ramp)
        nengo.Connection(start[1], accumulate.ea_ensembles[1], synapse=synapse, function=func_ramp)
        nengo.Connection(value.ea_ensembles[0], accumulate.ea_ensembles[0], synapse=synapse, function=func_ramp)
        nengo.Connection(value.ea_ensembles[1], accumulate.ea_ensembles[1], synapse=synapse, function=func_ramp)
        nengo.Connection(accumulate.output, accumulate.input, synapse=synapse)
        nengo.Connection(accumulate.output, combined)
        nengo.Connection(combined, gate, function=func_uncertainty)
        nengo.Connection(combined, action.input, function=func_greater)
        nengo.Connection(time, gate)
        nengo.Connection(thr, gate)
        nengo.Connection(gate, action.input, transform=[[-1], [-1]])

        # Probes
        net.pEvidence = nengo.Probe(evidence)
        net.pValue = nengo.Probe(value.output)
        net.pAccumulate = nengo.Probe(accumulate.output)
        net.pGate = nengo.Probe(gate)
        net.pAction = nengo.Probe(action.output)
        
        # references
        net.M = M
        net.delta = delta
        
    return net


def is_correct(inputs, net, sim, dt=0.001, action_threshold=0.0):
    chooseA = np.argwhere(sim.data[net.pAction][:,0] > action_threshold)
    chooseB = np.argwhere(sim.data[net.pAction][:,1] > action_threshold)
    timeA = chooseA[0][0]*dt if len(chooseA)>0 else sim.trange()[-1]
    timeB = chooseB[0][0]*dt if len(chooseB)>0 else sim.trange()[-1]
    if inputs.winning=="A":
        if timeA == timeB:  # forced response
            correct = True if sim.data[net.pAccumulate][-1][0] > sim.data[net.pAccumulate][-1][1] else False
            time = timeA
        elif timeA < timeB:
            correct = True
            time = timeA
        elif timeB < timeA:
            correct = False
            time = timeB
    elif inputs.winning=="B":
        if timeB == timeA:  # forced response
            correct = True if sim.data[net.pAccumulate][-1][1] > sim.data[net.pAccumulate][-1][0] else False
            time = timeB
        elif timeB < timeA:
            correct = True
            time = timeB
        elif timeA < timeB:
            correct = False
            time = timeA
    return correct, int(np.ceil(2*time)), time

def evaluate_fit(simulated_cues, empirical_cues, trial):
    delta_cues = np.abs(simulated_cues - empirical_cues[trial])
    # print('sim', simulated_cues, 'emp', empirical_cues[trial])
    return delta_cues

def main(args):

    with open('../pid_dict.pkl', 'rb') as f:
        pid_dict = pickle.load(f)

    pid = args['pid']
    participant_ID = pid_dict[pid]
    empirical_data = pd.read_pickle("../empirical_data.pkl").query("maxSamples==12 & delta==0.1 & participant_id==@participant_ID")
    empirical_cues = empirical_data['cues'].to_numpy()
    empirical_accuracy = empirical_data['correct'].to_numpy()
    maxSamples = 12
    nTrials = empirical_data.shape[0]
    # nTrials = 3

    seed = args['pid']
    T = args["T"]
    M = args["M"]
    tau = args["tau"]
    delta = args["delta"]
    print(seed, T, M, tau, delta)

    inputs = Inputs(deltaP=0.1, maxSamples=maxSamples, seed=seed, empirical=empirical_data)
    inputs.set_AB_empirical(0)
    net = build_network(inputs, T=T, M=M, tau=tau, delta=delta, seed=seed)
    delta_cues = np.zeros((nTrials))

    total_loss = 0
    for trial in range(nTrials):
        print(f"trial {trial}")
        inputs.set_AB_empirical(trial)
        sim = nengo.Simulator(net, progress_bar=False)
        with sim:
            sim.run(maxSamples, progress_bar=True)
        correct, cue, time = is_correct(inputs, net, sim)
        loss = evaluate_fit(cue, empirical_cues, trial)
        delta_cues[trial] = loss
        total_loss += loss
        nni.report_intermediate_result(total_loss)
    nni.report_final_result(total_loss)

if __name__ == '__main__':
    params = nni.get_next_parameter()
    # params = {"pid": 0, "T": 0, "M": 0, "tau": 0, "delta": 0}
    main(params)