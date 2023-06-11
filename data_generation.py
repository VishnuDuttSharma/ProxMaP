#import sys
#sys.path.append('/home/vishnuds/ai2thor')

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from helper_v3 import *

from tqdm import tqdm


g_count = 0
inp_dir = 'inp_data/'
inp_seg_dir = 'inp_seg_data/'
gt_dir = 'gt_data/'
descfile = 'updated_description_ang0.csv'

object_superlist = {
    'ceiling' : ['ceiling', 'ceilinglight'],
    'chairs' : ['armchair', 'chair', 'sofa', 'ottoman', 'stool'],
    'tables' : ['coffeetable', 'desk', 'diningtable', 'sidetable'],
    'floor' : ['floor', 'rug'],
    'walls' : ['blinds', 'door', 'painting', 'wall', 'wallcabinet', 'wallcabinetbody', 'window']
}

class_list = ['chairs', 'tables', 'floor', 'walls', 'ceiling']

df = pd.DataFrame(columns=['Filename',
                           'FloorName',
                           'pos_x',
                           'pos_y',
                           'pos_z',
                           'ang_x',
                           'ang_y',
                           'ang_z',
                           'free_perc'])

controller = BotController(init_scene=f'FloorPlan201')
for fl_num in range(201, 231):
    controller.controller.reset(scene=f'FloorPlan{fl_num}')
    ## Dictionary to save IDs for the third-camera parties
    controller.extra_cameras = {}
    ## Calculate the intrinsic camers
    controller.get_intrinsic_matric()

    # controller.add_top_camera()
    controller.add_side_cameras()

    positions = controller.get_reacheble_pos()
    print(f'Reachable positions: {len(positions)}')
    floor_color = [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'floor' in x['name'].lower()]
    floor_color += [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'rug' in x['name'].lower()]
    ceiling_color = [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'ceiling' in x['name'].lower()]
    ceiling_color += [x['color'] for x in controller.controller.last_event.metadata['colors'] if 'ceilinglight' in x['name'].lower()]

    obj_to_color_map = {n.lower(): list(c) for n,c in controller.controller.last_event.object_id_to_color.items() if '|' not in n}

    for pos in  tqdm(positions):
        for yaw in range(0,360,45):#range(0,360,30):
            orient = {'x': 0, 'y':yaw, 'z': 0}

            event = controller.controller.step(
                        action="Teleport",
                        position=pos,
                        rotation=orient
                    )

            controller.update_side_cameras()
            pc_list = controller.get_all_point_clouds()

            main_pcd = pc_list[0]
            inp_occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)

            main_pcd = pc_list[1]
            left_occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)

            main_pcd = pc_list[2]
            right_occ_map = get_occ_map(main_pcd, floor_color, ceiling_color)

            gt_map = np.maximum(np.abs(inp_occ_map), np.maximum(np.abs(left_occ_map), np.abs(right_occ_map))) * np.sign(inp_occ_map + left_occ_map + right_occ_map)

            seg_pcd = controller.get_seg_point_cloud(object_superlist, obj_to_color_map, class_list)
            seg_map_6ch =  get_seg_occ_map(seg_pcd, max_pts=255, class_list=class_list)

            free_perc = 100*np.sum(gt_map == 0)/gt_map.size
            occ_perc = 100*np.sum(inp_occ_map != 0)/inp_occ_map.size

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

            np.save(inp_dir+file_name+'.npy', arr=inp_occ_map)
            np.save(gt_dir+file_name+'.npy', arr=gt_map)
            np.save(inp_seg_dir+file_name+'.npy', arr=seg_map_6ch)

            df = df.append(desc, ignore_index=True)

            g_count += 1

df.to_csv(descfile)
