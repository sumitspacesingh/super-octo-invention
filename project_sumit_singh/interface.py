<<<<<<< HEAD
# replace MyCustomModel with the name of your model
from model import MyCustomModel as TheModel

# change my_descriptively_named_train_function to 
# the function inside train.py that runs the training loop.  
from train import my_descriptively_named_train_function as the_trainer

# change cryptic_inf_f to the function inside predict.py that
# can be called to generate inference on a single image/batch.
from predict import cryptic_inf_f as the_predictor

# change UnicornImgDataset to your custom Dataset class.
from dataset import UnicornImgDataset as TheDataset

# change unicornLoader to your custom dataloader
from dataset import unicornLoader as the_dataloader
=======
# replace MyCustomModel with the name of your model
from model import MyCustomModel as TheModel

# change my_descriptively_named_train_function to 
# the function inside train.py that runs the training loop.  
from train import my_descriptively_named_train_function as the_trainer

# change cryptic_inf_f to the function inside predict.py that
# can be called to generate inference on a single image/batch.
from predict import cryptic_inf_f as the_predictor

# change UnicornImgDataset to your custom Dataset class.
from dataset import UnicornImgDataset as TheDataset

# change unicornLoader to your custom dataloader
from dataset import unicornLoader as the_dataloader
>>>>>>> 1b2eee19662f137d68b5f7b6707c034db86c15f8
