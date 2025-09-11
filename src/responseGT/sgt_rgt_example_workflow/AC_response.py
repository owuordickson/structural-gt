
#==========================================================================
#0: imports
#==========================================================================
#Used for core functionality
import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
#Used for reading certain networks
import gsd.hoomd
import freud
#Just used for plotting/testing
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from StructuralGT.networks import Network
import time
import pandas as pd
#==========================================================================
#1: methods
#==========================================================================

def make_incidence_matrix_from_edgelist(Edge_list, N_vertices, N_edges): #makes an incidence matrix from a list of edges in O(n) time

        #We first initalize lists, to which we will append the row, column, and value of the non-zero elements that will fill our sparse C array
	Crows = []
	Ccols = []
	Cvals = []
	
        #Appending non-zero entries and their row/col data for each edge in the list. Edges are considered directed (+1/-1), but direction does not matter
	for i in range(len(Edge_list)):
		Crows.append(i)
		Ccols.append(int(Edge_list[i,0]))
		Cvals.append(-1)
		Crows.append(i)
		Ccols.append(int(Edge_list[i,1]))
		Cvals.append(1)
		#It is faster to append each element/coord to a list and then make a sparse array than it is to make a sparse array and then update each element

	C=csc_array((Cvals,(Crows,Ccols)),shape=(N_edges,N_vertices), dtype = "complex")

	return C


def AC_reposnse(C, R_list, L_list, C_list, RL_list, omega, ua_list, va_list):

	"""
	C: The incidence matrix of the network, where rows are directed edges and columns are vertices
	R_list: array of resistance for each EDGE,
	L_list: array of inductance for each EDGE.
	C_list: array of capacitance of each NODE that is NOT given an applied potential. nodes are taken to be capactiors connected to a grounded potential (0).
	RL_list: array of resistance between each NODE that is NOT given an applied potential and the ground, a "leakage" resistance.
	omega: angular frequency of applied alternating potential
	ua_list: array applied potentials
	va_list: array of nodes with the corresponding applied potential

	From my testing on square-lattice networks, this method has time complexity of O(n^~1.4).
	Time complexity might increase significantly in networks with higher average node degree.
	"""
	N = (C[0].shape)[0] #Number of vertices in the graph	
	va_list = np.array(va_list,dtype=int) #numpy array of vertices that have a forced potential, casting to int in case given as float
	Na=int(len(va_list))  #number of vertices in va_list
	Nb=int(N-Na) #number of vertices in vb_list

	sparseY = diags(1/(R_list+1j*omega*L_list))#diags(1/(rho+1j*omega*inductance)*np.ones(len(edges))) #Y=(R+iwL)^-1, addmittance matrix
	CT = C.T #transpose of incidence matrix
	sparseD = CT @ sparseY @ C #sparse version of the dynamical martrix

	isA_list = np.zeros(N) #auxiliary array where the value stored at an index is 1 if that index is in va_list
	for vert in va_list:
                isA_list[vert] = 1
	vb_list=[]
	for i in range(N): #constructing vb_list in O(n) time
                if isA_list[i]==0:
                        vb_list.append(i)

	A_index_list = np.zeros(N)
	for i in range(len(va_list)):
                A_index_list[va_list[i]] = i
	B_index_list = np.zeros(N)
	for i in range(len(vb_list)):
                B_index_list[vb_list[i]] = i
                
	sparseD_coo = sparseD.tocoo()
	sparseD_as_list= np.column_stack((sparseD_coo.row, sparseD_coo.col, sparseD_coo.data))

	aaRows = []
	aaCols = []
	aaVals = []

	bbRows = []
	bbCols = []
	bbVals = []

	baRows = []
	baCols = []
	baVals = []

	for elem in sparseD_as_list:
                x = int(np.real(elem[0]))
                y = int(np.real(elem[1]))
                val = elem[2]

                if isA_list[x]==1:
                        if isA_list[y]==1: #Daa
                                sp_x = int(A_index_list[x])
                                sp_y = int(A_index_list[y])
                                aaRows.append(sp_x)
                                aaCols.append(sp_y)
                                aaVals.append(val)
                else:
                        if isA_list[y]==0: #Dbb
                                sp_x = int(B_index_list[x])
                                sp_y = int(B_index_list[y])
                                bbRows.append(sp_x)
                                bbCols.append(sp_y)
                                bbVals.append(val)
                        else: #Dba
                                sp_x = int(B_index_list[x])
                                sp_y = int(A_index_list[y])
                                baRows.append(sp_x)
                                baCols.append(sp_y)
                                baVals.append(val)

        #Creating the sub-matrices for the Dynamical matrix, Daa, Dbb, Dab
	D_aa=csc_array((aaVals,(aaRows,aaCols)),shape=(Na,Na), dtype = "complex")
	D_ba=csc_array((baVals,(baRows,baCols)),shape=(Nb,Na), dtype = "complex")
	D_bb=csc_array((bbVals,(bbRows,bbCols)),shape=(Nb,Nb), dtype = "complex")


	#Keed to solve the equation -Dba ua = (Dbb - alpha I) ub
	#LHS = p, RHS = Q.ub
	#calculating p and Q:
	p= - D_ba @ ua_list
	Q = D_bb - diags(1j*omega*C_list) - diags(1/RL_list)

	#solving for u_b
	ub_list = spsolve(Q, p)

	#calculting potential response
	pot_res = np.zeros(N, dtype = "complex")
	
	#splicing the a and b components of the response back into a single list
	for i in range(len(va_list)):
                pot_res[va_list[i]] = ua_list[i]

	for i in range(len(vb_list)):
                pot_res[vb_list[i]] = ub_list[i]	

	#calculating current response
	cur_res = sparseY @ C @ pot_res

	return(pot_res, cur_res)


def plot_vert(vertpos, pot_res, edgelist, cur_res, phase_labels=None):
# Convert to numpy arrays
	vertpos = np.array(vertpos)
	pot_res = np.array(pot_res)
	cur_res = np.array(cur_res)
    
    # Create figure and axis
	fig, ax = plt.subplots(figsize=(9, 9))
    
    # Determine plot ranges
	x_min, x_max = vertpos[:, 0].min(), vertpos[:, 0].max()
	y_min, y_max = vertpos[:, 1].min(), vertpos[:, 1].max()
	x_pad = (x_max - x_min) * 0.1
	y_pad = (y_max - y_min) * 0.1
	ax.set_xlim(x_min - x_pad, x_max + x_pad)
	ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    # Normalize magnitudes for vertices and edges separately
	pot_mags = np.abs(pot_res)
	pot_mags_normalized = (pot_mags - pot_mags.min()) / (pot_mags.max() - pot_mags.min() + 1e-10)
	pot_phases = np.angle(pot_res) % (2 * np.pi)
    
	cur_mags = np.abs(cur_res)
	cur_mags_normalized = (cur_mags - cur_mags.min()) / (cur_mags.max() - cur_mags.min() + 1e-10)
	cur_phases = np.angle(cur_res) % (2 * np.pi)
    
    # Convert complex numbers to colors (HSV where H=phase, S=1, V=normalized magnitude)
	def complex_to_rgb(phases, mags):
		h = phases / (2 * np.pi)
		s = mags
		v = np.ones_like(h)
		hsv = np.stack([h, s, v], axis=-1)
		return mcolors.hsv_to_rgb(hsv)
    
	vertex_colors = complex_to_rgb(pot_phases, np.array([x for x in pot_mags_normalized]) ) #pot_mags_normalized
	edge_colors = complex_to_rgb(cur_phases, np.array([1 for x in cur_mags_normalized])) #cur_mags_normalized
    
    # Plot edges
	edge_segments = np.array([[vertpos[int(i)], vertpos[int(j)]] for i, j in edgelist])
	edge_collection = LineCollection(edge_segments, colors="0", linewidths=1*cur_mags_normalized) #colors=edge_colors
	#ax.add_collection(edge_collection)
    
    # Plot vertices
	ax.scatter(vertpos[:, 0], vertpos[:, 1], c=vertex_colors, s=30*pot_mags_normalized, edgecolors='none', zorder=3) #s=50*pot_mags_normalized
    
    # Create color wheel legend
	ax_color = fig.add_axes([0.75, 0.15, 0.15, 0.15], projection='polar') #this vector is x-position, y-position, width, height of colorwheel
    
    # Create color wheel using bars instead of fill_between
	n_angles = 360
	theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
	width = 2 * np.pi / n_angles
	radii = np.ones(n_angles)
    
    # Create HSV colors for the wheel
	h = theta / (2 * np.pi)
	s = np.ones_like(h)
	v = np.ones_like(h)
	hsv = np.stack([h, s, v], axis=-1)
	colors = mcolors.hsv_to_rgb(hsv)
    
    # Plot the color wheel using bar plot
	#bars = ax_color.bar(theta, radii, width=width, bottom=0, color=colors)
    
    # Add magnitude indicator (radius)
	#ax_color.plot([0, 0], [0, 1], 'k-', lw=1)  # magnitude indicator
	#ax_color.text(0, 1.1, '|z|', ha='center', va='center')
    
    # Add phase angle labels
	if phase_labels is None:
		phase_labels = {
			#0: '0',
			#np.pi/2: r'$\pi/2$',
			#np.pi: r'$\pi$',
			#3*np.pi/2: r'$3\pi/2$'
		}
    
	for angle, label in phase_labels.items():
		ax_color.text(angle, 1.3, label, ha='center', va='center')
    
	ax_color.set_xticks([])
	ax_color.set_yticks([])
	ax_color.set_theta_direction(-1)
	ax_color.set_theta_offset(np.pi/2)
	ax_color.set_frame_on(False)
    
	plt.tight_layout()
	plt.show()

def plot_edge(vertpos, pot_res, edgelist, cur_res, phase_labels=None):
# Convert to numpy arrays
	vertpos = np.array(vertpos)
	pot_res = np.array(pot_res)
	cur_res = np.array(cur_res)
    
    # Create figure and axis
	fig, ax = plt.subplots(figsize=(9, 9))
    
    # Determine plot ranges
	x_min, x_max = vertpos[:, 0].min(), vertpos[:, 0].max()
	y_min, y_max = vertpos[:, 1].min(), vertpos[:, 1].max()
	x_pad = (x_max - x_min) * 0.1
	y_pad = (y_max - y_min) * 0.1
	ax.set_xlim(x_min - x_pad, x_max + x_pad)
	ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    # Normalize magnitudes for vertices and edges separately
	pot_mags = np.abs(pot_res)
	pot_mags_normalized = (pot_mags - pot_mags.min()) / (pot_mags.max() - pot_mags.min() + 1e-10)
	pot_phases = np.angle(pot_res) % (2 * np.pi)
    
	cur_mags = np.abs(cur_res)
	cur_mags_normalized = (cur_mags - cur_mags.min()) / (cur_mags.max() - cur_mags.min() + 1e-10)
	cur_phases = np.angle(cur_res) % (2 * np.pi)
    
    # Convert complex numbers to colors (HSV where H=phase, S=1, V=normalized magnitude)
	def complex_to_rgb(phases, mags):
		h = phases / (2 * np.pi)
		s = mags
		v = np.ones_like(h)
		hsv = np.stack([h, s, v], axis=-1)
		return mcolors.hsv_to_rgb(hsv)
    
	vertex_colors = complex_to_rgb(pot_phases, np.array([x for x in pot_mags_normalized]) ) #pot_mags_normalized
	edge_colors = complex_to_rgb(np.mod(2*cur_phases,2*np.pi), np.array([x for x in cur_mags_normalized])) #cur_mags_normalized
    
    # Plot edges
	edge_segments = np.array([[vertpos[int(i)], vertpos[int(j)]] for i, j in edgelist])
	edge_collection = LineCollection(edge_segments, colors=edge_colors, linewidths=3*cur_mags_normalized) #colors=edge_colors
	ax.add_collection(edge_collection)
    
    # Plot vertices
	#ax.scatter(vertpos[:, 0], vertpos[:, 1], c=vertex_colors, s=10*pot_mags_normalized, edgecolors='none', zorder=3) #s=50*pot_mags_normalized
    
    # Create color wheel legend
	ax_color = fig.add_axes([0.75, 0.15, 0.15, 0.15], projection='polar') #this vector is x-position, y-position, width, height of colorwheel
    
    # Create color wheel using bars instead of fill_between
	n_angles = 360
	theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
	width = 2 * np.pi / n_angles
	radii = np.ones(n_angles)
    
    # Create HSV colors for the wheel
	h = theta / (2 * np.pi)
	s = np.ones_like(h)
	v = np.ones_like(h)
	hsv = np.stack([h, s, v], axis=-1)
	colors = mcolors.hsv_to_rgb(hsv)
    
    # Plot the color wheel using bar plot
	#bars = ax_color.bar(theta, radii, width=width, bottom=0, color=colors)
    
    # Add magnitude indicator (radius)
	#ax_color.plot([0, 0], [0, 1], 'k-', lw=1)  # magnitude indicator
	#ax_color.text(0, 1.1, '|z|', ha='center', va='center')
    
    # Add phase angle labels
	if phase_labels is None:
		phase_labels = {
			#0: '0',
			#np.pi/2: r'$\pi/2$',
			#np.pi: r'$\pi$',
			#3*np.pi/2: r'$3\pi/2$'
		}
    
	for angle, label in phase_labels.items():
		ax_color.text(angle, 1.3, label, ha='center', va='center')
    
	ax_color.set_xticks([])
	ax_color.set_yticks([])
	ax_color.set_theta_direction(-1)
	ax_color.set_theta_offset(np.pi/2)
	ax_color.set_frame_on(False)
    
	plt.tight_layout()
	plt.show()



def plot_all(vertpos, pot_res, edgelist, cur_res, phase_labels=None):
# Convert to numpy arrays
	vertpos = np.array(vertpos)
	pot_res = np.array(pot_res)
	cur_res = np.array(cur_res)
    
    # Create figure and axis
	fig, ax = plt.subplots(figsize=(16.4/2, 10.7/2)) #last ordered pair is aspect ratio
    
    # Determine plot ranges
	x_min, x_max = vertpos[:, 0].min(), vertpos[:, 0].max()
	y_min, y_max = vertpos[:, 1].min(), vertpos[:, 1].max()
	x_pad = (x_max - x_min) * 0.1
	y_pad = (y_max - y_min) * 0.1
	ax.set_xlim(x_min - x_pad, x_max + x_pad)
	ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    # Normalize magnitudes for vertices and edges separately
	pot_mags = np.abs(pot_res)
	pot_mags_normalized = (pot_mags - pot_mags.min()) / (pot_mags.max() - pot_mags.min() + 1e-10)
	pot_phases = np.angle(pot_res) % (2 * np.pi)
    
	cur_mags = np.abs(cur_res)
	cur_mags_normalized = (cur_mags - cur_mags.min()) / (cur_mags.max() - cur_mags.min() + 1e-10)
	cur_phases = np.angle(cur_res) % (2 * np.pi)
    
    # Convert complex numbers to colors (HSV where H=phase, S=1, V=normalized magnitude)
	def complex_to_rgb(phases, mags):
		h = phases / (2 * np.pi)
		s = mags
		v = np.ones_like(h)
		hsv = np.stack([h, s, v], axis=-1)
		return mcolors.hsv_to_rgb(hsv)
    
	vertex_colors = complex_to_rgb(pot_phases, np.array([x for x in pot_mags_normalized]) )
	edge_colors = complex_to_rgb(cur_phases, np.array([1 for x in cur_mags_normalized])) 
    
    # Plot edges
	edge_segments = np.array([[vertpos[int(i)], vertpos[int(j)]] for i, j in edgelist])
	edge_collection = LineCollection(edge_segments, colors="0", linewidths=5*cur_mags_normalized) #might need to change linewidths for your usage #can set colors=edge_colors to show phase of current response
	ax.add_collection(edge_collection)
    
    # Plot vertices
	ax.scatter(vertpos[:, 0], vertpos[:, 1], c=vertex_colors, s=10*pot_mags_normalized, edgecolors='none', zorder=3) #s is size variable, might need to change for your usage
    
    # Create color wheel legend
	ax_color = fig.add_axes([0.75, 0.15, 0.15, 0.15], projection='polar') #this vector is x-position, y-position, width, height of colorwheel
    
    # Create color wheel using bars instead of fill_between
	n_angles = 360
	theta = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
	width = 2 * np.pi / n_angles
	radii = np.ones(n_angles)
    
    # Create HSV colors for the wheel
	h = theta / (2 * np.pi)
	s = np.ones_like(h)
	v = np.ones_like(h)
	hsv = np.stack([h, s, v], axis=-1)
	colors = mcolors.hsv_to_rgb(hsv)
    
    # Plot the color wheel using bar plot
	#bars = ax_color.bar(theta, radii, width=width, bottom=0, color=colors) #uncomment to show colorwheel
    
    # Add magnitude indicator (radius)
	#ax_color.plot([0, 0], [0, 1], 'k-', lw=1)  # magnitude indicator
	#ax_color.text(0, 1.1, '|z|', ha='center', va='center')
    
    # Add phase angle labels
	if phase_labels is None:
		phase_labels = {
			#0: '0',
			#np.pi/2: r'$\pi/2$',
			#np.pi: r'$\pi$',
			#3*np.pi/2: r'$3\pi/2$'
		}
    
	for angle, label in phase_labels.items():
		ax_color.text(angle, 1.3, label, ha='center', va='center')
    
	ax_color.set_xticks([])
	ax_color.set_yticks([])
	ax_color.set_theta_direction(-1)
	ax_color.set_theta_offset(np.pi/2)
	ax_color.set_frame_on(False)
    
	plt.tight_layout()
	plt.show()



#==========================================
#2: Importing Files
#==========================================


edges = pd.read_csv('edgeList.csv', header=None, index_col=False).to_numpy()
vertpos = pd.read_csv('vertexPositions.csv', header=None, index_col=False).to_numpy()
n_vertices=len(vertpos)


#flips vertically to have same orientation as inital image
y_coords, x_coords = zip(*vertpos)
neg_y_coords = [y * -1 for y in y_coords]
vertpos=np.array(list(zip(x_coords, neg_y_coords)))
n_vertices=len(vertpos)


#==========================================
#3: Example Usage
#==========================================

#obtaining compatibility matrix
C=make_incidence_matrix_from_edgelist(np.array(edges), n_vertices, len(edges))


##making the imposed potential -- THIS IS WHAT YOU NEED TO CHANGE TO GET DIFFERENT RESPONSES ON THE SAME SYSTEM, here we just apply a top-bottom potential
given_potential_fraction = 0.05
given_potential_magnitude = 100
num_selected = int(given_potential_fraction * n_vertices)
given_potential_list = np.zeros(int(2*num_selected)) #Ultimately this is what gets passed on, the other code in this block just makes this a top-bottom potential
vertex_list = np.zeros(int(2*num_selected)) #Ultimately this is what gets passed on, the other code in this block just makes this a top-bottom potential
# Sort vertices by y-position
sorted_vertices = sorted(enumerate(vertpos), key=lambda x: x[-1][-1])#x[-1][-1])  # Sort by y-coordinate
# Select the top and bottom vertices
top_vertices = sorted_vertices[-num_selected:]  # Top f% (highest y-values)
bottom_vertices = sorted_vertices[:num_selected]  # Bottom f% (lowest y-values)
top_list = [idx for idx, _ in top_vertices]
bottom_list = [idx for idx, _ in bottom_vertices]
for i in range(len(top_list)):
        given_potential_list[i] = given_potential_magnitude
        vertex_list[i] = top_list[i]
for i in range(len(bottom_list)):
        given_potential_list[len(top_list)+i] = -given_potential_magnitude
        vertex_list[len(top_list)+i] = bottom_list[i]


#example parameters, set to very large/small numbers for DC response
given_potential_frequency = 0.000000000001
resistivity=1
capacitance=0.000000000001
inductance=0.00000000000000000001
leak_resistivity=1000000000

rlist = resistivity*np.ones(len(edges))
llist = inductance*np.ones(len(edges))
clist = capacitance*np.ones(n_vertices-2*num_selected)
rllist = leak_resistivity*np.ones(n_vertices-2*num_selected)


#running methods
start = time.time()
output = AC_reposnse(C, rlist, llist, clist, rllist, given_potential_frequency, given_potential_list, vertex_list)
end = time.time()
print("runtime:", end-start)


#The first two options plot only the vertices or the edges, if that is what you care about
#plot_vert(vertpos, output[0], edges, output[1], phase_labels=None)
#plot_edge(vertpos, output[0], edges, output[1], phase_labels=None)
plot_all(vertpos, output[0], edges, output[1], phase_labels=None)


df = pd.DataFrame(output[0])

# Export the DataFrame to a CSV file without header and index
df.to_csv('vertexPotentials.csv', header=False, index=False)

df = pd.DataFrame(output[1])

# Export the DataFrame to a CSV file without header and index
df.to_csv('edgeCurrents.csv', header=False, index=False)














    
      
