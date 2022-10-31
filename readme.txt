1. put all of the image folders(00000, 00001, ......) into data directory

2. put train_label.txt, test_label.txt and test_desired_age.txt into 0610131_src directory(root directory)

3. python3 train.py --split_data True

parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--split_data", type=bool, default=False, help="split the data into training set and testing set")


4. python3 output.py --split_data True

parser.add_argument("--model_id", type=int, default=50, help="choose a model from save/trained_model")
parser.add_argument("--split_data", type=bool, default=False, help="split the data into training set and testing set")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

5. the generated images are in save/gen_imgs dir