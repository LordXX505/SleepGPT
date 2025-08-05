import os
from glob import glob


def main():
    root_path = "/Volumes/T7/data/sleep-edf-database-expanded-1.0.0/sleep-cassette"
    out_path = os.path.join(root_path, 'usleep')
    regex_edf = 'SC4[0-9][0-9]*'
    all_items = glob(os.path.join(root_path, regex_edf))
    print(f'all_items: {all_items}')
    name_dict = {}
    for item in all_items:
        item_name = os.path.basename(item).split('-')[0]
        if item_name in name_dict:
            name_dict[item_name].append(item)
        else:
            name_dict[item_name] = [item]

    names = name_dict.keys()
    for ns in names:
        store_path = os.path.join(out_path, ns)
        os.makedirs(store_path, exist_ok=True)


if __name__ == '__main__':
    main()