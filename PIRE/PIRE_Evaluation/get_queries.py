from os import listdir
import os
import shutil


oxford_queries = ["all_souls_000013", "all_souls_000026", "oxford_002985", "all_souls_000051", "oxford_003410", "ashmolean_000058", "ashmolean_000000", "ashmolean_000269", "ashmolean_000007", "ashmolean_000305", "balliol_000051", "balliol_000187", "balliol_000167", "balliol_000194", "oxford_001753", "bodleian_000107", "oxford_002416", "bodleian_000108", "bodleian_000407", "bodleian_000163", "christ_church_000179", "oxford_002734", "christ_church_000999", "christ_church_001020", "oxford_002562", "cornmarket_000047", "cornmarket_000105", "cornmarket_000019", "oxford_000545", "cornmarket_000131", "hertford_000015", "oxford_001752", "oxford_000317", "hertford_000027", "hertford_000063", "keble_000245", "keble_000214", "keble_000227", "keble_000028", "keble_000055", "magdalen_000078", "oxford_003335", "magdalen_000058", "oxford_001115", "magdalen_000560", "pitt_rivers_000033", "pitt_rivers_000119", "pitt_rivers_000153", "pitt_rivers_000087", "pitt_rivers_000058", "radcliffe_camera_000519", "oxford_002904", "radcliffe_camera_000523", "radcliffe_camera_000095", "bodleian_000132"]
paris_queries = ["paris_defense_000605", "paris_defense_000331", "paris_defense_000216", "paris_defense_000056", "paris_defense_000254", "paris_general_002985", "paris_general_001729", "paris_eiffel_000266", "paris_general_002645", "paris_general_002391", "paris_invalides_000355", "paris_invalides_000072", "paris_invalides_000490", "paris_invalides_000229", "paris_invalides_000360", "paris_louvre_000081", "paris_louvre_000135", "paris_louvre_000050", "paris_louvre_000035", "paris_louvre_000139", "paris_moulinrouge_000667", "paris_moulinrouge_000868", "paris_moulinrouge_000657", "paris_moulinrouge_000794", "paris_moulinrouge_000004", "paris_museedorsay_000527", "paris_museedorsay_000012", "paris_museedorsay_000897", "paris_museedorsay_000564", "paris_museedorsay_000878", "paris_notredame_000256", "paris_notredame_000965", "paris_notredame_000390", "paris_general_003117", "paris_notredame_000581", "paris_pantheon_000466", "paris_pantheon_000520", "paris_pantheon_000232", "paris_pantheon_000547", "paris_pantheon_000339", "paris_pompidou_000432", "paris_pompidou_000444", "paris_pompidou_000252", "paris_pompidou_000471", "paris_pompidou_000636", "paris_sacrecoeur_000162", "paris_sacrecoeur_000417", "paris_sacrecoeur_000237", "paris_sacrecoeur_000586", "paris_sacrecoeur_000437", "paris_triomphe_000369", "paris_triomphe_000016", "paris_triomphe_000135", "paris_triomphe_000149", "paris_defense_000038"]

def generate_path(dataset:str)->str:
    return os.path.join(os.path.abspath(os.getcwd()), dataset)
    # path = os.path.abspath(os.getcwd())
    # path = os.path.join(path, 'cnnimageretrieval-pytorch')
    # path = os.path.join(path, 'data')
    # path = os.path.join(path, 'test')
    # path = os.path.join(path, dataset)
    # path = os.path.join(path, 'jpg')
    # return path
    

def search_name(file_name, name_list):
    for name in name_list:
        if file_name in name:
            return name
    
    return file_name

def copy_file(file_name, load_path, save_path):
    print(file_name)
    load_filepath = os.path.join(load_path, file_name)
    save_filepath = os.path.join(save_path, file_name)

    try:
        shutil.copyfile(load_filepath, save_filepath)
        # img = cv2.imread(load_filepath)
        # cv2.imwrite(save_filepath, img)
    except:
        print('\tWrite Error-%s' %file_name)

def copy_dataset_img(load_path, save_path, query_imgs):
    files = [f for f in listdir(load_path)]
    for file in query_imgs:
        copy_file(search_name(file, files), load_path, save_path)


load_oxford = generate_path('oxford5k')
save_oxford = os.path.join(os.path.abspath(os.getcwd()), 'Oxford_Queries')
load_paris = generate_path('paris6k')
save_paris = os.path.join(os.path.abspath(os.getcwd()), 'Paris_Queries')

copy_dataset_img(load_oxford, save_oxford, oxford_queries)
copy_dataset_img(load_paris, save_paris, paris_queries)








