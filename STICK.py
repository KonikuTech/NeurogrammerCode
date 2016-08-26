#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import math

#Ryad's parameters
#Threshold potential
V_t = 0.01 #V
#Reset potential
V_reset = 0 #V
#Synaptic delay
T_syn = 0.001 #second
#Spike delay (time-step resolution)
T_neu = 1e-5 #seconds
#Excitatory weight for V-synapses
w_e = V_t
#Inhibitory weight for V-synapses
w_i = -w_e
#Minimum encoding time
T_min = 0.01 #seconds
#Elementary time-step
T_cod = 0.1 #seconds
#Maximum encoding time
T_max = T_min + 1*T_cod
#Time constants
tau_m = 100 #seconds
tau_f = 0.02 #seconds
#Standard weights for ge-synapses
#Weight value for ge-synapses to cause a neuron to spike from its reset state after time Tmax
w_acc = V_t*tau_m/T_max
w_hat_acc = V_t * tau_m/T_cod

time = 0


def encode_value(x):
	delta_t = T_min + x*T_cod
	return delta_t

def decode_value(delta_t):
	if type(delta_t) == dict:
		x= {}
		for k in delta_t:
			if delta_t[k] != None:
				x[k] = (delta_t[k] - T_min) / T_cod
			else:
				x[k] = None
		return x
	else:
		if delta_t != None:
			x = (delta_t - T_min) / T_cod
			return x
		else:
			return None

def encode_sequence(delta_t):
	sequence = np.zeros(int(delta_t/T_neu) + 1, dtype=bool)
	sequence[0] = True
	sequence[int(delta_t/T_neu)] = True
	return sequence

def decode_sequence(sequence):
	if type(sequence) == dict:
		delta_t_out = {}
		for k in sequence:
			spike_times = np.where(sequence[k])
			if len(spike_times[0])>1:
				delta_t = (spike_times[0][1] - spike_times[0][0])*T_neu
				delta_t_out[k] = delta_t
			else:
				delta_t_out[k] = None
		return delta_t_out
	else:
		spike_times = np.where(sequence)
		if len(spike_times[0])>1:
			delta_t = (spike_times[0][1] - spike_times[0][0])*T_neu
			return delta_t
		return None


sequence_time =0.45 #ms
sequence_length = int(sequence_time/T_neu)

#Define neuron class:
class neuron(object):
	def __init__(self):
		#Membrane potential
		self.V = V_reset
		self.sequence = np.zeros(sequence_length, dtype=bool)
		self.synapse = {}
	def spike(self, time_step, neuron):
		neuron.sequence[time_step + int(T_syn/T_neu)] = True



def generate_constant_value(x):
	return encode_sequence(encode_value(x))



#To produce a chronogram identical to Ryad paper, pass the recall time as an argument to the recall fcn to shift output?
class inverting_memory(object):
	''' An inverting memory is used to store a value temporarily in a network and retrieve it at a later stage
		Note that the inverting memory is intended to be called in a sequence input-recall
		If two inputs are sent before recalling the value, the inputs will be summed
		and if the network recall pathway is called when no input has been set the network will return a 1'''
	def __init__(self):
		self.input_ = neuron()
		self.first = neuron()
		self.last = neuron()
		self.acc = neuron()
		self.output = neuron()
	def input(self, sequence):
		delta_T_in = decode_sequence(sequence)
		self.input_.sequence[0] = True
		self.input_.sequence[int(delta_T_in/T_neu)]=True
		self.first.sequence[int(T_syn/T_neu)] = True
		self.last.sequence[int((T_syn + delta_T_in)/T_neu)] = True
		self.acc.V += w_acc/tau_m*(delta_T_in - T_min)
		if self.acc.V >= V_t:
			self.acc.V = V_reset
	def recall(self):
		delta_T_out = (V_t - self.acc.V)*tau_m/w_acc
		self.acc.V = V_reset
		self.output.sequence[np.where(self.last.sequence)[0]+int((2*T_syn+T_neu)/T_neu)] = True
		self.acc.sequence[np.where(self.last.sequence)[0]+int((T_syn + delta_T_out)/T_neu)] = True 
		self.output.sequence[np.where(self.last.sequence)[0]+int((2*T_syn+delta_T_out+T_neu)/T_neu)] = True
		return self.output.sequence
	def plot(self):
		f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=True)
		ax1.vlines(np.where(self.input_.sequence), 0,1, color='b')
		ax1.set_title('Input')
		ax2.vlines(np.where(self.first.sequence), 0,1, color='b')
		ax2.set_title('First')
		ax3.vlines(np.where(self.last.sequence), 0,1, color='b')
		ax3.set_title('Last')
		ax4.vlines(np.where(self.acc.sequence), 0,1, color='b')
		ax4.set_title('Acc')
		ax5.vlines(np.where(self.output.sequence), 0,1, color='b')
		ax5.set_title('Output')
		f.suptitle("Inverting Memory Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		plt.show()

#invt = inverting_memory()  #create inverting memory object
#invt.input(encode_sequence(encode_value(.1))) #encode value and input
#recall_sequence = invt.recall() #call recall function
#print decode_value(decode_sequence(recall_sequence))
#decodes value from output spike sequence, prints .9!






class memory(object):
	''' An memory is used to store a value temporarily in a network and retrieve it at a later stage
		Note that the inverting memory is intended to be called in a sequence input-recall
		If two inputs are sent before recalling the value, the inputs will be summed
		and if the network recall pathway is called when no input has been set the network will return a 1'''
	def __init__(self):
		self.input_ = neuron()
		self.first = neuron()
		self.last = neuron()
		self.acc = neuron()
		self.acc2 = neuron()
		self.ready = neuron()
		self.output = neuron()
	def input(self, sequence):
		delta_T_in = decode_sequence(sequence)
		self.input_.sequence[0] = True
		self.input_.sequence[int(delta_T_in/T_neu)] = True
		self.first.sequence[int((T_syn + T_neu)/T_neu)] = True
		if (delta_T_in) == None:
			delta_T_in = T_min
		self.last.sequence[int((T_syn + T_neu + delta_T_in)/T_neu)] = True
		self.acc.sequence[int((2*T_syn + T_max + T_neu)/T_neu)] = True
		self.ready.sequence[int((3*T_syn + T_max + 2*T_neu)/T_neu)] = True
		self.acc2.V += w_acc/tau_m*(T_max - delta_T_in + T_syn + T_neu)
		if self.acc2.V >= V_t:
			self.acc2.V = V_reset
	def recall(self):
		self.output.sequence[int((T_syn + T_neu)/T_neu)] = True
		tacc2 = (V_t - self.acc2.V)*tau_m/w_acc + T_syn + T_neu
		self.acc2.sequence[int(tacc2/T_neu)] = True
		self.output.sequence[int((tacc2 + T_syn + T_neu)/T_neu)] = True 
		return self.output.sequence
	def plot(self):
		f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, sharey=True)
		ax1.vlines(np.where(self.input_.sequence), 0,1, color='b')
		ax1.set_title('Input')
		ax2.vlines(np.where(self.first.sequence), 0,1, color='b')
		ax2.set_title('First')
		ax3.vlines(np.where(self.last.sequence), 0,1, color='b')
		ax3.set_title('Last')
		ax4.vlines(np.where(self.acc.sequence), 0,1, color='b')
		ax4.set_title('Acc')
		ax5.vlines(np.where(self.acc2.sequence), 0,1, color='b')
		ax5.set_title('Acc 2')
		ax6.vlines(np.where(self.output.sequence), 0,1, color='b')
		ax6.set_title('Output')
		f.suptitle("Memory Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		plt.show()

#mem = memory()
#mem.input(encode_sequence(encode_value(.1)))
#mem.recall()
#mem.plot()

class signed_memory(object):
	''' This network receives a positive or negative input and stores it
		When the recall pathway is activated, a dictionary with the positive and negative sequences is returned'''
	def __init__(self):
		self.input_plus = neuron()
		self.input_minus = neuron()
		self.ready_plus = neuron()
		self.ready_minus = neuron()
		self.output_plus = neuron()
		self.output_minus = neuron()
		self.ready = neuron()
		self.memory = memory()
	def input_positive(self, sequence):
		delta_T_in = decode_sequence(sequence)
		self.input_plus.sequence[0] = True
		self.input_plus.sequence[int(delta_T_in/T_neu)]=True
		self.ready_plus.V += 0.5*w_e
		self.memory.input(sequence)
		self.ready.sequence[int((4*T_syn + T_max + 3*T_neu)/T_neu)] = True
	def input_negative(self, sequence):
		delta_T_in = decode_sequence(sequence)
		self.input_minus.sequence[0] = True
		self.input_minus.sequence[int(delta_T_in/T_neu)]=True
		self.ready_minus.V += 0.5*w_e
		self.memory.input(sequence)
		self.ready.sequence[int((4*T_syn + T_max + 3*T_neu)/T_neu)] = True		
	def recall(self):
		self.ready_plus.V += 0.5*w_e
		self.ready_minus.V += 0.5*w_e
		if self.ready_plus.V >= V_t:
			self.ready_plus.sequence[int((T_syn + T_neu)/T_neu)] = True
			self.ready_plus.V = V_reset
			self.ready_minus.V += 0.5*w_i
			self.output_plus.sequence = np.concatenate((np.zeros(int(3*T_syn + 3*T_neu), dtype=bool), self.memory.recall()))
			return {'+':self.output_plus.sequence, '-':np.zeros(len(self.output_plus.sequence), dtype=bool)}
		if self.ready_minus.V >= V_t:
			self.ready_minus.sequence[int((T_syn + T_neu)/T_neu)] = True
			self.ready_minus.V = V_reset
			self.ready_plus.V += 0.5*w_i
			self.output_minus.sequence = np.concatenate((np.zeros(int(3*T_syn + 3*T_neu), dtype=bool), self.memory.recall()))		
			return {'+':np.zeros(len(self.output_minus.sequence), dtype=bool), '-':self.output_minus.sequence}
	def plot(self):
		f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, sharex=True, sharey=True)
		ax1.vlines(np.where(self.input_plus.sequence), 0,1, color='b')
		ax1.set_title('Input +')
		ax2.vlines(np.where(self.input_minus.sequence), 0,1, color='b')
		ax2.set_title('Input -')
		ax3.vlines(np.where(self.ready_plus.sequence), 0,1, color='b')
		ax3.set_title('Ready +')
		ax4.vlines(np.where(self.ready_minus.sequence), 0,1, color='b')
		ax4.set_title('Ready -')
		ax5.vlines(np.where(self.output_plus.sequence), 0,1, color='b')
		ax5.set_title('Output +')
		ax6.vlines(np.where(self.output_minus.sequence), 0,1, color='b')
		ax6.set_title('Output -')
		ax7.vlines(np.where(self.ready.sequence), 0,1, color='b')
		ax7.set_title('Ready')
		f.suptitle("Signed Memory Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		plt.show()

#sgn = signed_memory()
#sgn.input_positive(encode_sequence(encode_value(.3)))
#sgn.recall()
#sgn.plot()

class synchronizer(object):
	def __init__(self, N):
		self.N = N
		self.inputs = []
		self.sync = neuron()
		self.inputs = [memory() for i in xrange(N)]
		self.input_number = 0
	def input(self, sequence):
		self.inputs[self.input_number].input(sequence)
		self.sync.V += w_e/self.N
		if self.sync.V >= V_t:
			out = {}
			for i in xrange(len(self.inputs)):
				out[i] = self.inputs[i].recall()
			return  out
		self.input_number += 1
	def plot(self):
		f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, sharey=True)
		ax1.vlines(np.where(self.inputs[0].recall()), 0,1, color='b')
		ax1.set_title('Input 1')
		ax2.vlines(np.where(self.inputs[1].recall()), 0,1, color='b')
		ax2.set_title('Input 2')
		ax3.vlines(np.where(self.last.sequence), 0,1, color='b')
		ax3.set_title('Last')
		ax4.vlines(np.where(self.acc.sequence), 0,1, color='b')
		ax4.set_title('Acc')
		ax5.vlines(np.where(self.acc2.sequence), 0,1, color='b')
		ax5.set_title('Acc 2')
		ax6.vlines(np.where(self.output.sequence), 0,1, color='b')
		ax6.set_title('Output')
		f.suptitle("Memory Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		plt.show()

class minimum(object):
	'''Input sequences need to be synchronized'''
	def __init__(self):
		self.input1 = neuron()
		self.input2 = neuron()
		self.smaller1 = neuron()
		self.smaller2 = neuron()
		self.output = neuron()
	def input(self, sequence1, sequence2):
		delta_T_in1 = decode_sequence(sequence1)
		self.input1.sequence[0] = True
		self.input1.sequence[int(delta_T_in1/T_neu)]=True
		delta_T_in2 = decode_sequence(sequence2)
		self.input2.sequence[0] = True
		self.input2.sequence[int(delta_T_in2/T_neu)]=True
		for i in xrange(max(len(sequence1), len(sequence2))):
			if i < len(sequence1):
				if sequence1[i]:
					self.smaller1.V += 0.5*w_e
					self.output.V += 0.5*w_e
			if i < len(sequence2):
				if sequence2[i]:
					self.smaller2.V += 0.5*w_e
					self.output.V += 0.5*w_e
			if self.smaller1.V >= V_t:
				self.smaller1.sequence[int(i + (T_syn+T_neu)/T_neu)] = True
				self.output.V += 0.5*w_e
				self.smaller2.V += 0.5*w_i
				self.smaller1.V = V_reset
			if self.smaller2.V >= V_t:
				self.smaller2.sequence[int(i + (T_syn+T_neu)/T_neu)] = True
				self.output.V += 0.5*w_e
				self.smaller1.V += 0.5*w_i
				self.smaller2.V = V_reset
			if self.output.V >= V_t:
				self.output.sequence[int((2*T_syn + 2*T_neu)/T_neu + i)] = True
				self.output.V = V_reset
		return {'smaller1':self.smaller1.sequence, 'smaller2': self.smaller2.sequence, 'output':self.output.sequence}
	def plot(self):
		f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=True)
		ax1.vlines(np.where(self.input1.sequence), 0,1, color='b')
		ax1.set_title('Input 1')
		ax2.vlines(np.where(self.input2.sequence), 0,1, color='b')
		ax2.set_title('Input 2')
		ax3.vlines(np.where(self.smaller1.sequence), 0,1, color='b')
		ax3.set_title('Smaller 1')
		ax4.vlines(np.where(self.smaller2.sequence), 0,1, color='b')
		ax4.set_title('Smaller 2')
		ax5.vlines(np.where(self.output.sequence), 0,1, color='b')
		ax5.set_title('Output')
		
		f.suptitle("Minimum Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		plt.show()
		
		
#min = minimum()
#min.input(encode_sequence(encode_value(.1)), encode_sequence(encode_value(.5)))
#min.plot()
		
class maximum(object):
	'''Input sequences need to be synchronized'''
	def __init__(self):
		self.input1 = neuron()
		self.input2 = neuron()
		self.larger1 = neuron()
		self.larger2 = neuron()
		self.output = neuron()
	def input(self, sequence1, sequence2):
		delta_T_in1 = decode_sequence(sequence1)
		self.input1.sequence[0] = True
		self.input1.sequence[int(delta_T_in1/T_neu)]=True
		delta_T_in2 = decode_sequence(sequence2)
		self.input2.sequence[0] = True
		self.input2.sequence[int(delta_T_in2/T_neu)]=True
		for i in xrange(max(len(sequence1), len(sequence2))):
			if i < len(sequence1):
				if sequence1[i]:
					self.larger2.V += 0.5*w_e
					self.output.V += 0.5*w_e
			if i < len(sequence2):
				if sequence2[i]:
					self.larger1.V += 0.5*w_e
					self.output.V += 0.5*w_e
			if self.larger1.V >= V_t:
				self.larger1.sequence[int(i + (T_syn+T_neu)/T_neu)] = True
				self.larger2.V += w_i
				self.larger1.V = V_reset
			if self.larger2.V >= V_t:
				self.larger2.sequence[int(i + (T_syn+T_neu)/T_neu)] = True
				self.larger1.V += w_i
				self.larger2.V = V_reset
			if self.output.V >= V_t:
				self.output.sequence[int((T_syn + T_neu)/T_neu + i)] = True
				self.output.V = V_reset
		return {'larger1':self.larger1.sequence, 'larger2': self.larger2.sequence, 'output':self.output.sequence}
	def plot(self):
		f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=True)
		ax1.vlines(np.where(self.input1.sequence), 0,1, color='b')
		ax1.set_title('Input 1')
		ax2.vlines(np.where(self.input2.sequence), 0,1, color='b')
		ax2.set_title('Input 2')
		ax3.vlines(np.where(self.larger1.sequence), 0,1, color='b')
		ax3.set_title('Larger 1')
		ax4.vlines(np.where(self.larger2.sequence), 0,1, color='b')
		ax4.set_title('Larger 2')
		ax5.vlines(np.where(self.output.sequence), 0,1, color='b')
		ax5.set_title('Output')
		
		f.suptitle("Maximum Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		plt.show()
		

#maxi = maximum()
#ret_seq = maxi.input(encode_sequence(encode_value(0.5)), 	encode_sequence(encode_value(0.3)))
#maxi.plot()


class subtract(object):
	def __init__(self):
		self.input1 = neuron()
		self.input2 = neuron()
		self.sync1 = neuron()
		self.sync2 = neuron()
		self.inb1 = neuron()
		self.inb2 = neuron()
		self.output_plus = neuron()
		self.output_minus = neuron()
	def input(self, sequence1, sequence2):
		self.sync1.synapse = {'output_plus': np.zeros(sequence_length, dtype=bool), 'inb1':np.zeros(sequence_length, dtype=bool), 'output_minus':np.zeros(sequence_length, dtype=bool), 'inb2':np.zeros(sequence_length, dtype=bool)}
		self.sync2.synapse = {'output_plus': np.zeros(sequence_length, dtype=bool), 'inb1':np.zeros(sequence_length, dtype=bool), 'output_minus':np.zeros(sequence_length, dtype=bool), 'inb2':np.zeros(sequence_length, dtype=bool)}
 		self.output_plus.synapse = {'output_plus': np.zeros(sequence_length, dtype=bool), 'inb2':np.zeros(sequence_length, dtype=bool)}
		self.output_minus.synapse = {'output_minus': np.zeros(sequence_length, dtype=bool), 'inb1':np.zeros(sequence_length, dtype=bool)}
		
		delta_T_in1 = decode_sequence(sequence1)
		self.input1.sequence[0] = True
		self.input1.sequence[int(delta_T_in1/T_neu)]=True
		delta_T_in2 = decode_sequence(sequence2)
		self.input2.sequence[0] = True
		self.input2.sequence[int(delta_T_in2/T_neu)]=True
		
		for i in xrange(2*int(T_max/T_neu)):
			if i < len(sequence1):
				if sequence1[i]:
					self.sync1.V += .5*w_e
			if i < len(sequence2):
				if sequence2[i]:
					self.sync2.V += .5*w_e
			if self.sync1.V >= V_t:
				self.sync1.synapse['output_plus'][int(i+(T_min+3*T_syn+2*T_neu)/T_neu)] = True
				self.sync1.synapse['inb1'][int(i+(T_syn+T_neu)/T_neu)] = True
				self.sync1.synapse['output_minus'][int(i+(3*T_syn+3*T_neu)/T_neu)] = True
				self.sync1.synapse['inb2'][int(i+(T_syn+T_neu)/T_neu)] = True
				self.sync1.V = V_reset
			if self.sync1.synapse['output_plus'][i]:
			 	self.output_plus.V += w_e
			if self.sync1.synapse['inb1'][i]:
				self.inb1.V += w_e
			if self.sync1.synapse['output_minus'][i]:
				self.output_minus.V += w_e
			if self.sync1.synapse['inb2'][i]:
				self.inb2.V += w_i
			if self.sync2.V >= V_t:
				self.sync2.synapse['output_minus'][int(i+(T_min+3*T_syn+2*T_neu)/T_neu)] = True
				self.sync2.synapse['inb1'][int(i+(T_syn+T_neu)/T_neu)] = True
				self.sync2.synapse['output_plus'][int(i+(3*T_syn+3*T_neu)/T_neu)] = True
				self.sync2.synapse['inb2'][int(i+(T_syn+T_neu)/T_neu)] = True
				self.sync2.V = V_reset
			if self.sync2.synapse['inb1'][i]:
				self.inb1.V += w_i
			if self.sync2.synapse['output_plus'][i]:
				self.output_plus.V += w_e
			if self.sync2.synapse['output_minus'][i]:
				self.output_minus.V += w_e
			if self.sync2.synapse['inb2'][i]:
				self.inb2.V += w_e
			if self.inb1.V >= V_t:
				self.inb1.sequence[int(i+(T_syn+T_neu)/T_neu)] = True
				self.inb1.V = V_reset
			if self.inb1.sequence[i]:
				self.output_plus.V += 2*w_i
			if self.inb2.V >= V_t:
				self.inb2.sequence[int(i+(T_syn+T_neu)/T_neu)] = True
				self.inb2.V = V_reset
			if self.inb2.sequence[i]:
				self.output_minus.V += 2*w_i
			if self.output_plus.V >= V_t:
				self.output_plus.synapse['output_plus'][int(i+(T_neu)/T_neu)] = True
				self.output_plus.synapse['inb2'][int(i+(T_syn+T_neu)/T_neu)] = True
				self.output_plus.V = V_reset
			if self.output_plus.synapse['inb2'][i]:
				self.inb2.V += .5*w_e
			if self.output_minus.V >= V_t:
				self.output_minus.synapse['output_minus'][int(i+(T_neu)/T_neu)] = True
				self.output_minus.synapse['inb1'][int(i+(T_syn+T_neu)/T_neu)] = True
				self.output_minus.V = V_reset
			if self.output_minus.synapse['inb1'][i]:
				self.inb1.V += .5*w_e
		return {'+':self.output_plus.synapse['output_plus'], '-':self.output_minus.synapse['output_minus']}
	def plot(self):
		f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, sharex=True, sharey=True)
		self.sync1.sequence = self.sync1.synapse['output_plus']+self.sync1.synapse['inb1']+self.sync1.synapse['output_minus']+self.sync1.synapse['inb2']
		self.sync2.sequence = self.sync2.synapse['output_plus']+self.sync2.synapse['inb1']+self.sync2.synapse['output_minus']+self.sync2.synapse['inb2']
		self.output_plus.sequence = self.output_plus.synapse['output_plus']
		self.output_minus.sequence = self.output_minus.synapse['output_minus']
		ax1.vlines(np.where(self.input1.sequence), 0,1, color='b')
		ax1.set_title('Input 1')
		ax2.vlines(np.where(self.input2.sequence), 0,1, color='b')
		ax2.set_title('Input 2')
		ax3.vlines(np.where(self.sync1.sequence), 0,1, color='b')
		ax3.set_title('Sync 1')
		ax4.vlines(np.where(self.sync2.sequence), 0,1, color='b')
		ax4.set_title('Sync 2')
		ax5.vlines(np.where(self.inb1.sequence), 0,1, color='b')
		ax5.set_title('Inb 1')
		ax6.vlines(np.where(self.inb2.sequence), 0,1, color='b')
		ax6.set_title('Inb 2')
		ax7.vlines(np.where(self.output_plus.sequence), 0,1, color='b')
		ax7.set_title('Output +')
		ax8.vlines(np.where(self.output_minus.sequence), 0,1, color='b')
		ax8.set_title('Output -')
		
		f.suptitle("Subtraction Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		plt.show()
		
#sub = subtract()
#subans = sub.input(encode_sequence(encode_value(0.9)), 	encode_sequence(encode_value(0.3)))
#print decode_value(decode_sequence(subans))
#sub.plot()

#use: lincomb_obj = lincomb(N) where N is the number of values to be summed
#lincomb_obj = input_plus(coeff, sequence) for positive values of sequence where coeff is the coefficient
#use input_minus for negative values of sequence (irrespective of the sign of the coeff in either case)
class lincomb(object):
	def __init__(self, N):
		self.N = N
		self.sync = neuron()
		self.input_number = 0
		self.acc1_plus = neuron()
		self.acc1_minus = neuron()
		self.inter_plus = neuron()
		self.inter_minus = neuron()
		self.acc2_plus = neuron()
		self.acc2_minus = neuron()
		self.Sync2 = synchronizer(2)
		self.Subs = subtract()
		self.output_plus = neuron()
		self.output_minus = neuron()
	def input_plus(self, coeff, sequence):
		delta_T_in = decode_sequence(sequence)
		if coeff >= 0:
			self.acc1_plus.V += abs(coeff)*w_acc/tau_m*(delta_T_in - T_min)
		else:
			self.acc1_minus.V += abs(coeff)*w_acc/tau_m*(delta_T_in - T_min)
		self.sync.V += w_e/self.N
		if self.sync.V >= V_t:
			tacc1_p = (V_t - self.acc1_plus.V)*tau_m/w_acc + T_syn + T_neu
			tacc1_m = (V_t - self.acc1_minus.V)*tau_m/w_acc + T_syn + T_neu
			self.acc1_plus.sequence[int(tacc1_p/T_neu)] = True
			self.acc1_minus.sequence[int(tacc1_m/T_neu)] = True
			self.acc1_plus.V = V_reset
			self.acc1_minus.V = V_reset
			self.sync.V = V_reset
			self.inter_plus.sequence[int((tacc1_p+T_syn+T_neu)/T_neu)] = True
			self.inter_minus.sequence[int((tacc1_m+T_syn+T_neu)/T_neu)] = True

			tacc2_p = (V_t - self.acc2_plus.V)*tau_m/w_acc + T_syn + T_neu
			tacc2_m = (V_t - self.acc2_minus.V)*tau_m/w_acc + T_syn + T_neu
			self.acc2_plus.sequence[int((tacc2_p+T_syn+T_min+T_neu)/T_neu)] = True
			self.acc2_minus.sequence[int((tacc2_m+T_syn+T_min+T_neu)/T_neu)] = True
			self.inter_plus.sequence[np.where(self.acc2_plus.sequence)[0]+int((T_syn+T_neu)/T_neu)] = True
			self.inter_minus.sequence[np.where(self.acc2_minus.sequence)[0]+int((T_syn+T_neu)/T_neu)] = True
			self.Sync2.input(self.inter_plus.sequence)
			sync_out = self.Sync2.input(self.inter_minus.sequence)
			diff = self.Subs.input(sync_out[0], sync_out[1])
			self.output_plus.sequence = diff['+']
			self.output_minus.sequence = diff['-']
			self.input_number += 1
			return {'+':self.output_plus.sequence, '-':self.output_minus.sequence}
			
			
	def input_minus(self, coeff, sequence):
		delta_T_in = decode_sequence(sequence)
		if coeff >= 0:
			self.acc1_minus.V += abs(coeff)*w_acc/tau_m*(delta_T_in - T_min)
		else:
			self.acc1_plus.V += abs(coeff)*w_acc/tau_m*(delta_T_in - T_min)
		self.sync.V += w_e/self.N
		if self.sync.V >= V_t:
			tacc1_p = (V_t - self.acc1_plus.V)*tau_m/w_acc + T_syn + T_neu
			tacc1_m = (V_t - self.acc1_minus.V)*tau_m/w_acc + T_syn + T_neu
			self.acc1_plus.sequence[int(tacc1_p/T_neu)] = True
			self.acc1_minus.sequence[int(tacc1_m/T_neu)] = True
			self.acc1_plus.V = V_reset
			self.acc1_minus.V = V_reset
			self.sync.V = V_reset
			self.inter_plus.sequence[int((tacc1_p+T_syn+T_neu)/T_neu)] = True
			self.inter_minus.sequence[int((tacc1_m+T_syn+T_neu)/T_neu)] = True

			tacc2_p = (V_t - self.acc2_plus.V)*tau_m/w_acc + T_syn + T_neu
			tacc2_m = (V_t - self.acc2_minus.V)*tau_m/w_acc + T_syn + T_neu
			self.acc2_plus.sequence[int((tacc2_p+T_syn+T_min+T_neu)/T_neu)] = True
			self.acc2_minus.sequence[int((tacc2_m+T_syn+T_min+T_neu)/T_neu)] = True
			self.inter_plus.sequence[np.where(self.acc2_plus.sequence)[0]+int((T_syn+T_neu)/T_neu)] = True
			self.inter_minus.sequence[np.where(self.acc2_minus.sequence)[0]+int((T_syn+T_neu)/T_neu)] = True
			self.Sync2.input(self.inter_plus.sequence)
			sync_out = self.Sync2.input(self.inter_minus.sequence)
			diff = self.Subs.input(sync_out[0], sync_out[1])
			self.output_plus.sequence = diff['+']
			self.output_minus.sequence = diff['-']
			self.input_number += 1
			return {'+':self.output_plus.sequence, '-':self.output_minus.sequence}
			
	def plot(self):
		f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, sharey=True)
		ax1.vlines(np.where(self.acc1_plus.sequence), 0,1, color='b')
		ax1.set_title('acc1 +')
		ax2.vlines(np.where(self.acc1_minus.sequence), 0,1, color='b')
		ax2.set_title('acc1 -')
		ax3.vlines(np.where(self.acc2_plus.sequence), 0,1, color='b')
		ax3.set_title('acc2 +')
		ax4.vlines(np.where(self.acc2_minus.sequence), 0,1, color='b')
		ax4.set_title('acc2 -')
		ax5.vlines(np.where(self.inter_plus.sequence), 0,1, color='b')
		ax5.set_title('Inter +')
		ax6.vlines(np.where(self.inter_minus.sequence), 0,1, color='b')
		ax6.set_title('Inter -')
		f.suptitle("Linear Combination Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		axes.set_xlim([0,len(self.inter_plus.sequence)])
		plt.show()
			
		
#lin = lincomb(2)
#lin.input_plus(.1, encode_sequence(encode_value(.5)))
#seq =  lin.input_minus(.1, encode_sequence(encode_value(.3)))
#print decode_value(decode_sequence(seq))
#lin.plot()


#gets the logarithm of the input for 0 < x < 1
class log(object):
	def __init__(self):
		self.first = neuron()
		self.last = neuron()
		self.acc = neuron()
		self.output = neuron()
	def input(self, sequence):
		delta_T_in = decode_sequence(sequence)
		self.first.sequence[int((T_syn + T_neu)/T_neu)] = True
		if (delta_T_in) == None:
			delta_T_in = T_min
		self.last.sequence[int((T_syn + T_neu + delta_T_in)/T_neu)] = True
		tacc = .1*math.log(T_cod/(delta_T_in-T_min))
		self.acc.sequence[np.where(self.last.sequence)[0]+int((tacc+T_syn+T_neu)/T_neu)] = True
		self.output.sequence[np.where(self.acc.sequence)[0]+int((T_syn+T_min+T_neu)/T_neu)] = True
		self.output.sequence[np.where(self.last.sequence)[0]+int((2*T_syn+T_neu)/T_neu)] = True
		return {'+':None, '-':self.output.sequence}
	def plot(self):
		f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
		ax1.vlines(np.where(self.first.sequence), 0,1, color='b')
		ax1.set_title('First')
		ax2.vlines(np.where(self.last.sequence), 0,1, color='b')
		ax2.set_title('Last')
		ax3.vlines(np.where(self.acc.sequence), 0,1, color='b')
		ax3.set_title('Acc')
		ax4.vlines(np.where(self.output.sequence), 0,1, color='b')
		ax4.set_title('Output')

		f.suptitle("Logarithm Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		axes.set_xlim([0,len(self.inter_plus.sequence)])
		plt.show()

#getlog = log()
#spike_in = encode_sequence(encode_value(.8))
#ans = getlog.input(spike_in)
#print decode_value(decode_sequence(ans))

#tau_f changed to .1 to produce the real answer instead of a proportional one.
class exponential(object):
	def __init__(self):
		self.first = neuron()
		self.last = neuron()
		self.acc = neuron()
		self.output = neuron()
	def input(self, sequence):
		delta_T_in = decode_sequence(sequence)
		self.first.sequence[int((T_syn + T_neu)/T_neu)] = True
		if (delta_T_in) == None:
			delta_T_in = T_min
		self.last.sequence[int((T_syn + T_neu + delta_T_in)/T_neu)] = True
		tacc = T_cod*(math.exp((delta_T_in - T_min)/ .1))
		self.acc.sequence[np.where(self.last.sequence)[0]+int((tacc+T_syn+T_neu)/T_neu)] = True
		self.output.sequence[np.where(self.acc.sequence)[0]+int((T_syn+T_min+T_neu)/T_neu)] = True
		self.output.sequence[np.where(self.last.sequence)[0]+int((2*T_syn+T_neu)/T_neu)] = True
		return {'+':None, '-':self.output.sequence}
	def plot(self):
		f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)
		ax1.vlines(np.where(self.first.sequence), 0,1, color='b')
		ax1.set_title('First')
		ax2.vlines(np.where(self.last.sequence), 0,1, color='b')
		ax2.set_title('Last')
		ax3.vlines(np.where(self.acc.sequence), 0,1, color='b')
		ax3.set_title('Acc')
		ax4.vlines(np.where(self.output.sequence), 0,1, color='b')
		ax4.set_title('Output')

		f.suptitle("Exponential Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		plt.show()

		
#getexp = exponential()
#spike_in = encode_sequence(encode_value(.8))
#ans = getexp.input(spike_in)
#print decode_value(decode_sequence(ans))
#getexp.plot()

class multiplier(object):
	def __init__(self):
		self.first1 = neuron()
		self.last1 = neuron()
		self.first2 = neuron()
		self.last2 = neuron()
		self.acclog1 = neuron()
		self.acclog2 = neuron()
		self.sync = neuron()
		self.accexp = neuron()
		self.output = neuron()
	def input(self, sequence1, sequence2):
		self.last1.synapse = {'acclog1': np.zeros(sequence_length, dtype=bool), 'sync':np.zeros(sequence_length, dtype=bool)}
		self.last2.synapse = {'acclog2': np.zeros(sequence_length, dtype=bool), 'sync':np.zeros(sequence_length, dtype=bool)}
		self.acclog2.synapse = {'accexp': np.zeros(sequence_length, dtype=bool), 'output':np.zeros(sequence_length, dtype=bool)}
		delta_T_in1 = decode_sequence(sequence1)
		self.first1.sequence[int((T_syn + T_neu)/T_neu)] = True
		if (delta_T_in1) == None:
			delta_T_in1 = T_min
		self.last1.sequence[int((T_syn + T_neu + delta_T_in1)/T_neu)] = True
		
		delta_T_in2 = decode_sequence(sequence2)
		self.first2.sequence[int((T_syn + T_neu)/T_neu)] = True
		if (delta_T_in2) == None:
			delta_T_in2 = T_min
		self.last2.sequence[int((T_syn + T_neu + delta_T_in2)/T_neu)] = True
	
		for i in xrange(2*int(T_max/T_neu)):
			if self.last1.sequence[i]:
				self.sync.V += .5*w_e
				
			if self.last2.sequence[i]:
				self.sync.V += .5*w_e
			
			if self.sync.V >= V_t:
				self.sync.sequence[i+int((3*T_syn+T_neu)/T_neu)] = True
				tacclog1 = .1*math.log(T_cod/(delta_T_in1-T_min))
				tacclog2 = .1*math.log(T_cod/(delta_T_in2-T_min))
				self.acclog1.sequence[i+int((2*T_syn + T_neu + tacclog1)/T_neu)] = True
				self.sync.V = V_reset
				self.acclog2.sequence[np.where(self.acclog1.sequence)[0]+int((tacclog2+T_syn+T_neu)/T_neu)] = True
				taccexp = T_cod*math.exp(T_neu*(np.where(self.sync.sequence)[0]-np.where(self.acclog2.sequence)[0])/.1)	
				self.accexp.sequence[np.where(self.acclog2.sequence)[0]+int((taccexp+T_syn+T_neu) / T_neu)] = True
				self.output.sequence[np.where(self.acclog2.sequence)[0]+int((T_syn)/T_neu)] = True
				self.output.sequence[np.where(self.acclog2.sequence)[0]+int((taccexp+T_syn+T_min+T_neu)/T_neu)] = True
		return self.output.sequence
		
	def plot(self):
		f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, sharex=True, sharey=True)
		input1 = self.first1.sequence+self.last1.sequence
		input2 = self.first2.sequence+self.last2.sequence
		ax1.vlines(np.where(input1), 0,1, color='b')
		ax1.set_title('Input 1')
		ax2.vlines(np.where(input2), 0,1, color='b')
		ax2.set_title('Input 2')
		ax3.vlines(np.where(self.acclog1.sequence), 0,1, color='b')
		ax3.set_title('Acc Log 1')
		ax4.vlines(np.where(self.acclog2.sequence), 0,1, color='b')
		ax4.set_title('Acc Log 2')
		ax5.vlines(np.where(self.sync.sequence), 0,1, color='b')
		ax5.set_title('Sync')
		ax6.vlines(np.where(self.accexp.sequence), 0,1, color='b')
		ax6.set_title('Acc Exp')
		ax7.vlines(np.where(self.output.sequence), 0,1, color='b')
		ax7.set_title('Output')
	
		
		f.suptitle("Multiplication Spike Chronogram")
		axes = plt.gca()
		axes.set_ylim([0,1])
		plt.show()
#mult = multiplier()
#seq = mult.input(encode_sequence(encode_value(.2)), encode_sequence(encode_value(.8)))
#print decode_value(decode_sequence(seq))
#print np.where(mult.acclog1.sequence)
#mult.plot()

class integrator(object):
	def __init__(self):
		
		self.const = neuron()
		self.input_plus_ = neuron()
		self.input_minus_ = neuron()
		self.start = neuron()
		self.output_plus = neuron()
		self.output_minus = neuron()
#load the integrator with an initial condition x0
	def init(self, x0):
		self.accumulate = lincomb(2)
		delta_T_in = encode_value(x0)
		if (delta_T_in) == None:
			delta_T_in = T_min
		self.const.sequence[0] = True
		self.const.sequence[int(delta_T_in/T_neu)] = True
		self.start.sequence[0] = True
		self.start.sequence[int(T_min/T_neu)] = True
		
		self.accumulate.input_plus(1, self.const.sequence)
		init_out = self.accumulate.input_plus(1, self.start.sequence)
		self.output_plus.sequence = init_out['+']
		print "output init"
		print decode_value(decode_sequence(self.output_plus.sequence))
		self.output_plus.V = V_t
		
	def input_plus(self, sequence, dt):
		self.accumulate = lincomb(2)
		delta_T_in = decode_sequence(sequence)
		if (delta_T_in) == None:
			delta_T_in = T_min
		self.input_plus_.sequence[0] = True
		self.input_plus_.sequence[int(delta_T_in/T_neu)] = True
		if self.output_plus.V >= V_t:
			self.accumulate.input_plus(1, self.output_plus.sequence)
			self.output_plus.V = V_reset
		if self.output_minus.V >= V_t:
			self.accumulate.input_minus(1, self.output_minus.sequence)
			self.output_minus.V = V_rest
		output = self.accumulate.input_plus(dt, sequence)
		print np.where(output)
		self.output_plus.sequence = output['+']
		self.output_minus.sequence = output['-']
		if decode_sequence(self.output_plus.sequence) == None:
			self.output_minus.V = V_t
		if decode_sequence(self.output_minus.sequence) == None:
			self.output_plus.V = V_t
		
		return output
		
	def input_minus(self, sequence, dt):
		delta_T_in = decode_sequence(sequence)
		if (delta_T_in) == None:
			delta_T_in = T_min
		self.input_plus_.sequence[0] = True
		self.input_plus_.sequence[int(delta_T_in/T_neu)] = True
		if self.output_plus.V >= V_t:
			self.accumulate.input_plus(1, self.output_plus.sequence)
			self.output_plus.V = V_reset
		if self.output_minus.V >= V_t:
			self.accumulate.input_minus(1, self.output_minus.sequence)
			self.output_minus.V = V_reset
		output = self.accumulate.input_minus(dt, sequence)
		
		self.output_plus.sequence = output['+']
		self.output_minus.sequence = output['-']
		if decode_sequence(self.output_plus.sequence) == None:
			self.output_minus.V = V_t
		if decode_sequence(self.output_minus.sequence) == None:
			self.output_plus.V = V_t
		
		return output
			
#integrate = integrator()
#integrate.init(.1)
#integrate.input_plus(encode_sequence(encode_value(.3)), .1)
#out = integrate.input_plus(encode_sequence(encode_value(.4)), .1)
#print decode_value(decode_sequence(out))
