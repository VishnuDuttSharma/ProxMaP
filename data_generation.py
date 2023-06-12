#import sys
#sys.path.append('/home/vishnuds/ai2thor')

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from helper_v3 import *

from tqdm import tqdm

## Variables
g_count = 0 # Global counter for file names
inp_dir = 'inp_data/' # Input data directory
gt_dir = 'gt_data/' # Ground Truth data directory

descfile = 'updated_description_ang0.csv' # Description/metadata file name

## Data generation
# Creating placeholder dataframe
df = pd.DataFrame(columns=['Filename',
                           'FloorName',
                           'pos_x',
                           'pos_y',
                           'pos_z',
                           'ang_x',
                           'ang_y',
                           'ang_z',
                           'free_perc'])

# Initiating contorller
controller = BotController(init_scene=f'FloorPlan201')

# Iterating over the living rooms
for fl_num in range(201, 231):
    controller.controller.reset(scene=f'FloorPlan{fl_num}')
    ## Dictionary to save IDs for the third-camera parties
    controller.extra_cameras = {}
    ## Calculate the intrinsic camers
    controller.get_intrinsic_matric()
    
    ## Adding cameras to the sides of the robot
    # controller.add_top_camera()
    controller.add_side_cameras()
    
    # Finding all reacheble positions in the living room
    positions = controller.get_reacheble_pos()
    print(f'Reachable positions: {len(positions)}')

    # Finding the colors for floor and ceiling for clean removal
    floor_color = [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'floor' in x['name'].lower()]
    floor_color += [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'rug' in x['name'].lower()]
    ceiling_color = [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'ceiling' in x['name'].lower()]
    ceiling_color += [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'ceilinglight' in x['name'].lower()]
    
    # Creating a map from objects name to color
    obj_to_color_map = {n.lower(): list(c) for n,c in controller.controller.last_event.object_id_to_color.items() if '|' not in n}
    
    # Iterating over reachable positions
    for pos in  tqdm(positions):
        # Rotating in steps of 45 at each location
        for yaw in range(0,360,45):#range(0,360,30):
            # Moving/Teleporting the robot to the desired pose
            orient = {'x': 0, 'y':yaw, 'z': 0}
            event = controller.controller.step(
                        action="Teleport",
                        position=pos,
                        rotation=orient
                    )
            
            # Updating location of the side cameras
            controller.update_side_cameras()

            # getting point clouds from all cameras
            pc_list = controller.get_all_point_clouds()
            
            # Getting occupancy map from the center camera
            main_pcd = pc_list[0]
            inp_occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)
            
            # Getting occupancy map from the left camera
            main_pcd = pc_list[1]
            left_occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)
            
            # Getting occupancy map from the right camera
            main_pcd = pc_list[2]
            right_occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)
            
            # Creating the ground truth occupancy map as the combination of all three occupancy maps
            gt_map = np.maximum(np.abs(inp_occ_map), np.maximum(np.abs(left_occ_map), np.abs(right_occ_map))) * np.sign(inp_occ_map + left_occ_map + right_occ_map)

            # Finding out the percentage of free and occupaied space in occupancy maps
            free_perc = 100*np.sum(gt_map == 0)/gt_map.size
            occ_perc = 100*np.sum(inp_occ_map != 0)/inp_occ_map.size
            
            # Filling in information for this data point into a dataframe
            file_name = f'FP{fl_num}_{g_count}'
            desc = {}
            desc['Filename'] = file_name
            desc['FloorName'] = f'FloorPlan{fl_num}'
            desc['pos_x'] = pos['x']
            desc['pos_y'] = pos['y']
            desc['pos_z'] = pos['z']
            desc['ang_x'] = orient['x']
            desc['ang_y'] = orient['y']
            desc['ang_z'] = orient['z']
            desc['free_perc'] = free_perc
            desc['occ_perc'] = occ_perc
            
            # Saving the input and ground truth occupancy maps
            np.save(inp_dir+file_name+'.npy', arr=inp_occ_map)
            np.save(gt_dir+file_name+'.npy', arr=gt_map)
            
            # Stacking the information into the main description/metadata dataframe
            df = df.append(desc, ignore_index=True)     
            
            # Increamenting the filename counter
            g_count += 1

# Saving the descrption file
df.to_csv(descfile)
