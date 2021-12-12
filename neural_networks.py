import params
from math import tanh


class Brain:

    def __init__(self, genome, cell):
        self.genome = genome
        self.cell = cell
        self.LPD = 0
        self.Kill = 0
        self.OSC = 0
        self.SG = 0
        self.Res = 0
        self.Mfd = 0
        self.Mrn = 0
        self.Mrv = 0
        self.MRL = 0
        self.MX = 0
        self.MY = 0
        # action neurons sum up each of their inputs, and apply tanh to it self.action_neurons = [self.LPD,
        # self.Kill, self.OSC, self.SG, self.Res, self.Mfd, self.Mrn, self.Mrv, self.MRL, self.MX, self.MY]  # list
        # of everything a cell could possibly do
        self.action_neurons = ['LPD', 'Kill', 'OSC', 'SG', 'Res', 'Mfd', 'Mrn', 'Mrv', 'MRL', 'MX', 'MY']
        self.prev_action = [0 for _ in self.action_neurons]

        # sensory neurons produce values from 0 to 1 self.sensory_neurons = [self.cell.Slr, self.cell.Sfd,
        # self.cell.Sg, self.cell.Age, self.cell.Rnd, self.cell.Blr,                                self.cell.Osc,
        # self.cell.Bfd, self.cell.Plr,                                self.cell.Pop, self.cell.Pfd, self.cell.LPf,
        # self.cell.LMy, self.cell.LBf,self.cell.LMx, self.cell.BDy, self.cell.Gen,self.cell.BDx, self.cell.BD,
        # self.cell.Lx,self.cell.Ly]  # list of everything a cell could possibly detect
        self.sensory_neurons = ['Slr', 'Sfd', 'Sg', 'Age', 'Rnd',
                                'Blr', 'Osc', 'Bfd', 'Plr', 'Pop',
                                'Pfd', 'LPf', 'LMy', 'LBf', 'LMx',
                                'BDy', 'Gen', 'BDx', 'BD', 'Lx',
                                'Ly']  # list of everything a cell could possibly detect
        self.inner_neurons = []
        self.create_internal_neurons()
        self.depth = 0
        self.neurons = self.sensory_neurons + self.action_neurons + self.inner_neurons
        self.sources = self.sensory_neurons + self.inner_neurons
        self.sinks = self.action_neurons + self.inner_neurons
        # self.recalculate_values()

    def create_internal_neurons(self):
        for i in range(params.INNER_NEURON_COUNT):
            setattr(self, 'InnerNeuron' + str(i), 0)
            self.inner_neurons.append('InnerNeuron' + str(i))

    def recalculate_values(self):
        self.depth += 1  # make sure we don't dig too deep or too greedily
        if self.depth == 10:
            self.depth = 0
            return
        self.prev_action = [int(getattr(self, val)) for val in self.action_neurons]
        temp_inner = [0 for i in range(params.INNER_NEURON_COUNT)]
        for gene in [self.genome[i:i + 8] for i in range(0, len(self.genome), 8)]:
            gene_binary = bin(int(gene, 16))[2:].zfill(32)
            source = int(gene_binary[0])  # 1 = internal neuron, 0 = input
            source_id = int(gene_binary[1:8], 2)# id of source neuron
            sink = int(gene_binary[9])  # 1 = internal neuron, 0 = output
            sink_id = int(gene_binary[10:16], 2)  # id of sink neuron
            mult = -1 if gene_binary[17] else 1  # sign magnitude value
            weight = mult * int(gene_binary[18:31], 2) / 8192  # give a floating point range from -4 to 4
            if sink == 0:
                if source == 0:
                    setattr(self, self.sinks[sink_id % len(self.sinks)], getattr(self, self.sinks[sink_id % len(self.sinks)]) + weight * getattr(self.cell, self.sensory_neurons[source_id % len(self.sensory_neurons)]))
                else:
                    setattr(self, self.sinks[sink_id % len(self.sinks)], getattr(self, self.sinks[sink_id % len(self.sinks)]) + weight * getattr(self, self.inner_neurons[source_id % params.INNER_NEURON_COUNT]))
            else:
                if source == 0:
                    temp_inner[sink_id % params.INNER_NEURON_COUNT] += weight * getattr(self.cell, self.sensory_neurons[source_id % len(self.sensory_neurons)])
                else:
                    temp_inner[sink_id % params.INNER_NEURON_COUNT] += weight * getattr(self, self.inner_neurons[source_id % params.INNER_NEURON_COUNT])
            for neurons in self.action_neurons:
                setattr(self, neurons, tanh(getattr(self, neurons)))
            for i, neurons in enumerate(self.inner_neurons):
                setattr(self, neurons, tanh(temp_inner[i]))

        total_change = 0  # see if the system is stable
        for i in range(len(self.action_neurons)):
            total_change += (getattr(self, self.action_neurons[i]) - self.prev_action[i]) ** 2
        if total_change > params.MAX_CHANGE:
            self.recalculate_values()
        else:
            self.depth = 0

    @property
    def LPD(self):
        """set long-probe distance"""
        return self._LPD

    @property
    def Kill(self):
        """kill forward neighbor"""
        return self._Kill

    @property
    def OSC(self):
        """set oscillator period"""
        return self._OSC

    @property
    def SG(self):
        """emit pheromone"""
        return self._SG

    @property
    def Res(self):
        """set responsiveness"""
        return self._Res

    @property
    def Mfd(self):
        """move forward"""
        return self._Mfd

    @property
    def Mrn(self):
        """move random"""
        return self._Mrn

    @property
    def Mrv(self):
        """move reverse"""
        return self._Mrv

    @property
    def MRL(self):
        """move left/right"""
        return self._MRL

    @property
    def MX(self):
        """move east/west"""
        return self._MX

    @property
    def MY(self):
        """move north/south"""
        return self._MY

    @LPD.setter
    def LPD(self, value):
        """set long-probe distance"""
        self._LPD = value

    @Kill.setter
    def Kill(self, value):
        """kill forward neighbor"""
        self._Kill = value

    @OSC.setter
    def OSC(self, value):
        """set oscillator period"""
        self._OSC = value

    @SG.setter
    def SG(self, value):
        """emit pheromone"""
        self._SG = value

    @Res.setter
    def Res(self, value):
        """set responsiveness"""
        self._Res = value

    @Mfd.setter
    def Mfd(self, value):
        """move forward"""
        self._Mfd = value

    @Mrn.setter
    def Mrn(self, value):
        """move random"""
        self._Mrn = value

    @Mrv.setter
    def Mrv(self, value):
        """move reverse"""
        self._Mrv = value

    @MRL.setter
    def MRL(self, value):
        """move left/right"""
        self._MRL = value

    @MX.setter
    def MX(self, value):
        """move east/west"""
        self._MX = value

    @MY.setter
    def MY(self, value):
        """move north/south"""
        self._MY = value
