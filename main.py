from common import *
import time


######### MAIN PROGRAM ########################################################

# Number of bodies.
N = 100

# Location of the center of the distribution
center = array([0.5, 0.5]) 

# Initial radius of the distribution
ini_radius = 10.

# Number of time-iterations executed by the program.
n = 3000

# Frequency at which .PNG images are written.
img_step = 50

# Folder to save the images
image_folder = 'imagesBH/'

# Name of the generated video
video_name = str(N)+'bodies.mp4'



# Main 
start = time.time()
bodies = system_init(N, center, ini_radius)
print(f'\nEl número total de cuerpos es: {len(bodies):.0f}\n')
evolve(bodies, n, center, ini_radius, img_step, image_folder)
end = time.time()

total_time = end - start
print(f'\nPara {N:.0f} partículas, el tiempo de computo fue {total_time:.2f}')

create_video(image_folder, video_name)