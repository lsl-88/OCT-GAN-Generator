from Dataset import CNV, DME, Drusen
from Processing import Preprocess
from Inference.Inference import LR_Inference, SR_Inference
from Model import DCGAN, EDSR
from glob import glob
import os.path as op
from sklearn.utils import shuffle

data_dir = '..\\Data'
save_dir = '..\\Processed_data'

# Load the data
CNV_data_list = CNV(data_dir).load()
CNV(data_dir).print_stats

DME_data_list = DME(data_dir).load()
DME(data_dir).print_stats

Drusen_data_list = Drusen(data_dir).load()
Drusen(data_dir).print_stats

# Process the data
Preprocess(data_list=CNV_data_list, data_dir=data_dir, save_dir=save_dir, condition='CNV').full_data()
Preprocess(data_list=DME_data_list, data_dir=data_dir, save_dir=save_dir, condition='DME').full_data()
Preprocess(data_list=Drusen_data_list, data_dir=data_dir, save_dir=save_dir, condition='Drusen').full_data()

# Train the DCGAN and EDSR model (CNV)
CNV_processed_data = glob(op.join(save_dir, 'CNV', 'OCT', '*'))
CNV_processed_data = shuffle(CNV_processed_data)

model = DCGAN(name='Run_1', condition='CNV').create_model()
model.fit(train_data=CNV_processed_data, batch_size=8, epochs=2000)

model = EDSR(name='Run_1', condition='CNV').create_model()
model.fit(dataset=CNV_processed_data, batch_size=8, epochs=2000)

# Train the DCGAN and EDSR model (DME)
DME_processed_data = glob(op.join(save_dir, 'DME', 'OCT', '*'))
DME_processed_data = shuffle(DME_processed_data)

model = DCGAN(name='Run_2', condition='DME').create_model()
model.fit(train_data=DME_processed_data, batch_size=8, epochs=2000)

model = EDSR(name='Run_2', condition='DME').create_model()
model.fit(dataset=DME_processed_data, batch_size=8, epochs=2000)

# Train the DCGAN and EDSR model (Drusen)
drusen_processed_data = glob(op.join(save_dir, 'Drusen', 'OCT', '*'))
drusen_processed_data = shuffle(drusen_processed_data)

model = DCGAN(name='Run_3', condition='Drusen').create_model()
model.fit(train_data=drusen_processed_data, batch_size=8, epochs=2000)

model = EDSR(name='Run_3', condition='Drusen').create_model()
model.fit(dataset=drusen_processed_data, batch_size=8, epochs=2000)

# Perform inference using the generative model and EDSR model
LR_Inference(name='Run_1', condition='CNV').generate_images(num_images=1000)
SR_Inference(name='Run_1', condition='CNV').generate_images(num_images=1000)

LR_Inference(name='Run_2', condition='DME').generate_images(num_images=1000)
SR_Inference(name='Run_2', condition='DME').generate_images(num_images=1000)

LR_Inference(name='Run_3', condition='Drusen').generate_images(num_images=1000)
SR_Inference(name='Run_3', condition='Drusen').generate_images(num_images=1000)