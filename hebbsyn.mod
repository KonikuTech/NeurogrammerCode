NEURON {
	POINT_PROCESS hebbsyn
	RANGE tau, e, i
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau = 0.1 (ms) <1e-9,1e9>
	e = 0	(mV)
}

ASSIGNED {
	v (mV)
	i (nA)
}

STATE {
	g (uS)
}

INITIAL {
	g=0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g*(v - e)
}

DERIVATIVE state {
	g' = -g/tau
}

NET_RECEIVE(weight1 (uS), weight2 (uS), spikeno) {
	INITIAL {
		spikeno = 0
		}
	if (spikeno == 0) {
		g = g + weight1
		spikeno = spikeno + 1
	}
	else if (spikeno == 1){
		g = g + weight2
	}
		
}
