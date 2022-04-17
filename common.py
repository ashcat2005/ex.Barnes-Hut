# 2-D Barnes-Hut Algorithm

from copy import deepcopy
from numpy import array, ones, empty, random, sqrt
from numpy.linalg import norm
import matplotlib.pyplot as plt


##### Simulation Parameters #########################################################

# Gravitational constant
G = 1.e-5

# Discrete time step.
dt = 1.e-2 

# Theta-criterion of the Barnes-Hut algorithm.
theta = 0.3

#####################################################################################

class Node:
    '''
    A node object will represent a body (if node.child is None)
    or an abstract node of the quad-tree if it has node.child attributes.
    '''
    def __init__(self, m, position, momentum):
        '''
        Creates a child-less node using the arguments
        .mass : scalar
        .position : NumPy array  with the coordinates [x,y]
        .momentum : NumPy array  with the components [px,py]
        '''
        self.m = m
        self.m_pos = m * position
        self.momentum = momentum
        self.child = None
    
    def position(self):
        '''
        Returns the physical coordinates of the node.
        '''
        return self.m_pos / self.m
        
    def reset_location(self):
        '''
        Resets the position of the node to the 0th-order quadrant.
        The size of the quadrant is reset to the value 1.0
        '''
        self.size = 1.0
        # The relative position inside the 0th-order quadrat is equal
        # to the current physical position.
        self.relative_position = self.position().copy()
        
    def place_into_quadrant(self):
        '''
        Places the node into next order quadrant.
        Returns the quadrant number according to the labels defined in the
        documentation.
        '''
        # The next order quadrant will have half the side of the current quadrant
        self.size = 0.5 * self.size
        return self.subdivide(1) + 2*self.subdivide(0)

    def subdivide(self, i):
        '''
        Places the node node into the next order quadrant along the direction i
        and re-calculates the relative_position of the node inside this quadrant.
        '''
        self.relative_position[i] *= 2.0
        if self.relative_position[i] < 1.0:
            quadrant = 0
        else:
            quadrant = 1
            self.relative_position[i] -= 1.0
        return quadrant



def add(body, node):
    '''
    Defines the quad-tree by introducing a body and locating it
    according to three conditions (see documentation for details).
    Returns the updated node containing the body.
    '''
    smallest_quadrant = 1.e-4 # Lower limit for the side-size of the quadrants
    
    # Case 1. If node does not contain a body, the body is put in here
    new_node = body if node is None else None
    
    if node is not None and node.size > smallest_quadrant:
        # Case 3. If node is an external node, then the new body can not
        # be put in there. We have to verify if it has .child attribute
        if node.child is None:
            new_node = deepcopy(node)
            # Subdivide the node creating 4 children
            new_node.child = [None for i in range(4)]
            # Place the body in the appropiate quadrant
            quadrant = node.place_into_quadrant()
            new_node.child[quadrant] = node
        # Case 2. If node is an internal node, it already has .child attribute
        else:
            new_node = node

        # For cases 2 and 3, it is needed to update the mass and the position
        # of the node
        new_node.m += body.m
        new_node.m_pos += body.m_pos
        # Add the new body into the appropriate quadrant.
        quadrant = body.place_into_quadrant()
        new_node.child[quadrant] = add(body, new_node.child[quadrant])
    return new_node


def distance_between(node1, node2):
    '''
    Returns the distance between node1 and node2.
    '''
    d12 = node1.position() - node2.position()
    return sqrt(d12.dot(d12))


def gravitational_force(node1, node2):
    '''
    Returns the gravitational force that node1 exerts on node2.
    A short distance cutoff is introduced in order to avoid numerical
    divergences in the gravitational force.
    '''
    cutoff_dist = 1.e-4
    d12 = node1.position() - node2.position()
    d = sqrt(d12.dot(d12))
    if d < cutoff_dist:
        # Returns no Force to prevent divergences!
        return array([0., 0.])
    else:
        # Gravitational force
        return G*node1.m*node2.m*(d12)/d**3


def force_on(body, node, theta):
    '''
    # Barnes-Hut algorithm: usage of the quad-tree. This function computes
    # the net force on a body exerted by all bodies in node "node".
    # Note how the code is shorter and more expressive than the human-language
    # description of the algorithm.
    '''
    # 1. If the current node is an external node,
    #    calculate the force exerted by the current node on b.
    if node.child is None:
        return gravitational_force(node,body)

    # 2. Otherwise, calculate the ratio s/d. If s/d < θ, treat this internal
    #    node as a single body, and calculate the force it exerts on body b.
    if node.size < distance_between(node,body)*theta:
        return gravitational_force(node,body)

    # 3. Otherwise, run the procedure recursively on each child.
    return sum(force_on(body, c, theta) for c in node.child if c is not None)


def verlet(bodies, root, theta, dt):
    '''
    Velocity-Verlet method for time evolution.
    '''
    for body in bodies:
        force = force_on(body, root, theta)
        body.momentum += 0.5*force*dt
        body.m_pos += body.momentum*dt
        body.momentum += 0.5*force_on(body, root, theta)*dt
        

def random_generate(N, center, ini_radius):
    '''
    Randomly generate the system of N particles.
    Returns
    - Positions
    - Momenta
    '''
    # We will generate K=2*N random particles from which we will chose
    # only N-bodies for the system
    K = 2*N
    random.seed(413)
    positions = empty([N,2])
    momenta = empty([N,2])
    mass = 1.0
    # x-, y- positions are initialized inside a square with
    # a side of length = 2*ini_radius.
    posx = random.random(K) *2.*ini_radius + center[0]-ini_radius
    posy = random.random(K) *2.*ini_radius + center[1]-ini_radius
    i=0
    j=0
    #Loop until complete the random N bodies or use the K generated bodies
    while i<K and j<N:
        position = array([posx[i],posy[i]])
        r = position - center
        norm_r = sqrt(r.dot(r))
        if norm_r < ini_radius:
            positions[j] = position
            momenta[j] = mass*0.01*array([-r[1], r[0]])*(norm_r+0.5*ini_radius)/ini_radius
            j+=1
        i+=1
    return mass, positions, momenta
    
    
def system_init(N, center, ini_radius):
    '''
    This function initialize the N-body system by randomly defining
    the position vectors fo the bodies and creating the corresponding
    objects of the Node class
    '''
    bodies = []
    mass, positions, momenta = random_generate(N, center, ini_radius)
    for i in range(N):
       bodies.append(Node(mass, positions[i], momenta[i]))
    return bodies



def evolve(bodies, n, center, ini_radius, img_step, image_folder='images/'):
    '''
    This function evolves the system in time using the Verlet algorithm 
    and the Barnes-Hut quad-tree
    '''
    # Limits for the axes in the plot
    axis_limit = 1.1*ini_radius
    lim_inf = [center[0]-axis_limit, center[1]-axis_limit]
    lim_sup = [center[0]+axis_limit, center[1]+axis_limit]
    
    # Principal loop over n time iterations.
    for i in range(n+1):
        # The quad-tree is recomputed at each iteration.
        root = None
        for body in bodies:
            body.reset_location()
            root = add(body, root)

        # Evolution using the Verlet method
        verlet(bodies, root, theta, dt)

        # Write the image files
        if i%img_step==0:
            print("Writing image at time {0}".format(i))
            plot_bodies(bodies, i//img_step, lim_inf, lim_sup, image_folder)


def plot_bodies(bodies, i, lim_inf, lim_sup, image_folder='images/'):
    '''
    Writes an image file with the current position of the bodies
    '''
    fig,ax = plt.subplots(figsize=(10,10), facecolor='black')
    ax.set_xlim([lim_inf[0], lim_sup[0]])
    ax.set_ylim([lim_inf[1], lim_sup[1]])
    ax.set_facecolor('black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    for body in bodies:
        pos = body.position()
        ax.scatter(pos[0], pos[1], marker='.', color='lightcyan')
    plt.savefig(image_folder+'bodies_{0:06}.png'.format(i))
    plt.close()


def create_video(image_folder='images/', video_name='my_video.mp4'):
    '''
    Creates a .mp4 video using the stored files images
    '''
    from os import listdir
    import moviepy.video.io.ImageSequenceClip
    fps = 15
    image_files = [image_folder+img for img in sorted(listdir(image_folder)) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)



def create_avi_video(image_folder='images/', video_name = 'video.avi'):
    '''
    Creates a .avi video using the stored files images
    '''
    import cv2
    from os import listdir
    from os.path import join
    images = [img for img in listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    for image in images:
        video.write(cv2.imread(join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()






if __name__=="__main__":
    '''
    Example of a randomly generated N-body system to be evolved using
    the Barnes-Hut Algorithm and the Verlet method
    '''
    import time

    # Number of bodies.
    N = 500

    # Location of the center of the distribution
    center = array([0.5, 0.5]) 

    # Initial radius of the distribution
    ini_radius = 10.
    
    # Number of time-iterations executed by the program.
    n = 2000

    # Frequency at which .PNG images are written.
    img_step = 20

    # Folder to save the images
    image_folder = 'imagesBH/'

    # Name of the generated video
    video_name = str(N)+'bodiesE.mp4'

    # Main 
    start = time.time()
    bodies = system_init(N, center, ini_radius)
    print(f'\nEl número total de cuerpos es: {len(bodies):.0f}\n')
    evolve(bodies, n, center, ini_radius, img_step, image_folder)
    end = time.time()
    total_time = end - start
    print(f'\nPara {N:.0f} partiículas, el tiempo de computo fue {total_time:.2f}')

    create_video(image_folder, video_name)