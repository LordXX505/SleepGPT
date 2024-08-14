import glob
import os
Root_path = '/data/shhs_raw/polysomnography/edfs'
shhs1_root_path = os.path.join(Root_path, 'shhs1')
shhs2_root_path = os.path.join(Root_path, 'shhs2')
shhs1_data_path_list = sorted(glob.glob(shhs1_root_path + '/*'))
shhs2_data_path_list = sorted(glob.glob(shhs2_root_path + '/*'))
all_data_path_list = shhs1_data_path_list + shhs2_data_path_list
print(len(all_data_path_list))
anna_root_path = '/data/shhs_raw/polysomnography/annotations-events-profusion'
anna_shhs1_path = os.path.join(anna_root_path, 'shhs1')
anna_shhs2_path = os.path.join(anna_root_path, 'shhs2')
anna_shhs1_path_list = sorted(glob.glob(anna_shhs1_path + '/*'))
anna_shhs2_path_list = sorted(glob.glob(anna_shhs2_path + '/*'))
all_anno_path_list = anna_shhs1_path_list + anna_shhs2_path_list

store_path = '/mnt/myvol/data/shhs'
process_items = glob.glob(os.path.join(store_path, '*', 'success'))
process_items = [item.split('/')[-2] for item in process_items]
for items in all_data_path_list:
    base_name = os.path.basename(items).split('.')[0]
    if base_name not in process_items:
        print(items)
