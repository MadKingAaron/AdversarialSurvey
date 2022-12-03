# Initial Setup
 Delete `.keep` files from empty directories
 -
 1. Remove all `.keep` files from `./oxford5k`, `./paris6k`, `./Paris_Queries`, `./Oxford_Queries`, `./PIRE/img_input`, `./PIRE/img_output`, `./cnnimageretrieval-pytorch/queries/oxford5k`, `./cnnimageretrieval-pytorch/queries/paris6k`, `./cnnimageretrieval-pytorch/vecs/oxford5k`, and `./cnnimageretrieval-pytorch/vecs/paris6k`

 Download and Extract Dataset Images
 -
 1. Download and extract the [Oxford5k](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) and [Paris6k](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/)
 2. Place all of the image files for each of dataset in the folder `./<dataset>`
	- Where `<dataset>` is either `oxford5k` or `paris6k`

 Setup [PIRE](https://github.com/liuzrcc/PIRE)
 -

 1. Install PyTorch 1.0.0 and Python 3.6
 2. If not already created, create folder in root of PIRE folder called `img_output`
 3. Insert images to generate adversarial images for in `img_input` folder

 Get query images
 -
 1. In the root of the repository, run the script, `get_queries.py`, to obtain the adversarial images for both the [Oxford5k](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) and [Paris6k](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) datasets
 	- If the script fails, copy the query images for the dataset you wish to generate adversarial images for copy then to the `img_input` folder
	 	- The files can be found in `cnnimageretrieval-pytorch/data/test/<dataset>/jpg`, where `<dataset>` is either `oxford5k` or `paris6k`
		- For Oxford5k, these are the image queries: `["all_souls_000013", "all_souls_000026", "oxford_002985", "all_souls_000051", "oxford_003410", "ashmolean_000058", "ashmolean_000000", "ashmolean_000269", "ashmolean_000007", "ashmolean_000305", "balliol_000051", "balliol_000187", "balliol_000167", "balliol_000194", "oxford_001753", "bodleian_000107", "oxford_002416", "bodleian_000108", "bodleian_000407", "bodleian_000163", "christ_church_000179", "oxford_002734", "christ_church_000999", "christ_church_001020", "oxford_002562", "cornmarket_000047", "cornmarket_000105", "cornmarket_000019", "oxford_000545", "cornmarket_000131", "hertford_000015", "oxford_001752", "oxford_000317", "hertford_000027", "hertford_000063", "keble_000245", "keble_000214", "keble_000227", "keble_000028", "keble_000055", "magdalen_000078", "oxford_003335", "magdalen_000058", "oxford_001115", "magdalen_000560", "pitt_rivers_000033", "pitt_rivers_000119", "pitt_rivers_000153", "pitt_rivers_000087", "pitt_rivers_000058", "radcliffe_camera_000519", "oxford_002904", "radcliffe_camera_000523", "radcliffe_camera_000095", "bodleian_000132"]`
		- For Paris6k, these are the image queries: `["paris_defense_000605", "paris_defense_000331", "paris_defense_000216", "paris_defense_000056", "paris_defense_000254", "paris_general_002985", "paris_general_001729", "paris_eiffel_000266", "paris_general_002645", "paris_general_002391", "paris_invalides_000355", "paris_invalides_000072", "paris_invalides_000490", "paris_invalides_000229", "paris_invalides_000360", "paris_louvre_000081", "paris_louvre_000135", "paris_louvre_000050", "paris_louvre_000035", "paris_louvre_000139", "paris_moulinrouge_000667", "paris_moulinrouge_000868", "paris_moulinrouge_000657", "paris_moulinrouge_000794", "paris_moulinrouge_000004", "paris_museedorsay_000527", "paris_museedorsay_000012", "paris_museedorsay_000897", "paris_museedorsay_000564", "paris_museedorsay_000878", "paris_notredame_000256", "paris_notredame_000965", "paris_notredame_000390", "paris_general_003117", "paris_notredame_000581", "paris_pantheon_000466", "paris_pantheon_000520", "paris_pantheon_000232", "paris_pantheon_000547", "paris_pantheon_000339", "paris_pompidou_000432", "paris_pompidou_000444", "paris_pompidou_000252", "paris_pompidou_000471", "paris_pompidou_000636", "paris_sacrecoeur_000162", "paris_sacrecoeur_000417", "paris_sacrecoeur_000237", "paris_sacrecoeur_000586", "paris_sacrecoeur_000437", "paris_triomphe_000369", "paris_triomphe_000016", "paris_triomphe_000135", "paris_triomphe_000149", "paris_defense_000038"]`

Setting up [Evaluation Script](https://github.com/filipradenovic/cnnimageretrieval-pytorch)
-
 1. Install PyTorch 1.0.0 and Python 3.6 if you have not done so already
 2. Place all image queries in `./cnnimageretrieval-pytorch/queries/<dataset>`
	- Replace `<dataset>` with either `oxford5k` or `paris6k`
 3. In order to run a set of baseline tests, from the `cnnimageretrieval-pythorch` folder run the script `run_eval.py` 
	 - Make sure to run the script as `python run_eval.py --gpu` if you going to run the script with GPU
	 - **MAKE SURE TO RUN THE EVALUATION SCRIPT BEFORE PROCEEDING. THE SCRIPT WILL DOWNLOAD ALL NECESSARY MODELS AND DATASETS THE FIRST TIME IT IS RUN.**
	  	- **WARNING: DOWNLOADS WILL TAKE UP 60GB+**


# Generate Adversarial Queries
 1. In the root director of PIRE folder, run `python gen_pire.py -T "<iterations>" -gpu_id "0" -cnnmodel "<model>" -in_dir "./img_input/" -out_dir "./img_output/" -p True`
	 - Replace `<iterations>` with the number of iterations desired to run PIRE on
	 - Replace `<model>` with either `gem` or `imagenet-res101`
	 - Make sure to set `-gpu_id "-1"` if you going to run the script with CPU
 2. The generated adversarial images will be in the `img_output` folder

# Evaluation Script

Testing PIRE with Evaluation Script
-
 1. Insert generated adversarial queries into the respective folders for the dataset, `./cnnimageretrieval-pytorch/queries/<dataset>`
	 - Replace `<dataset>` with corresponding dataset, i.e. `oxford5k` or `paris6k`
 2. Run evaluation script as stated in the previous section with the adversarial images replacing the old images<br />


> Evaluation of PIRE created by Aaron Jimenez and Laboni Sarker

