from cnn2snn import check_model_compatibility, quantize, convert
from tensorflow import keras
import argparse
import json, os, shutil, sys

# parse arguments (--metadata FILE is passed in)
parser = argparse.ArgumentParser(description='Custom deploy block demo')
parser.add_argument('--metadata', type=str)
args = parser.parse_args()

# load the metadata.json file
with open(args.metadata) as f:
    metadata = json.load(f)

print('Copying files to build directory...')
input_dir = metadata['folders']['input']
output_dir = metadata['folders']['output']

# create a build directory, the input / output folders are on network storage so might be very slow
build_dir = '/tmp/build'
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)
os.makedirs(build_dir)

# copy in the data from input folder
os.system('cp -r ' + input_dir + '/* ' + build_dir)
shutil.unpack_archive(os.path.join(build_dir, 'trained.savedmodel.zip'), build_dir, 'zip')
os.system('ls -l ' + build_dir)

print('Copying files to build directory OK')
print('')

model = keras.models.load_model(os.path.join(build_dir, 'saved_model'))
check_model_compatibility(model, input_is_image=False)

model_quantized = quantize(model, 8, 4, 4) 



# How many epochs we will fine tune the model with QAT
FINE_TUNE_EPOCHS = 30
model_quantized.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0000045),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model_quantized.fit(train_dataset_batch,
                epochs=FINE_TUNE_EPOCHS,
                verbose=2,
                validation_data=validation_dataset_batch,
                callbacks=callbacks,
                class_weight=None
            )

model_akida = convert(model_quantized, input_is_image=False)
model_akida.summary()
# remove everything besides quantized model and akida_model
os.system('rm -r ' + build_dir)
model_quantized.save(os.path.join(build_dir, 'saved_model_keras_quantized'))
model_akida.save(os.path.join(build_dir, 'akida_model.fbz'))

shutil.make_archive(os.path.join(output_dir, 'akida_deployment'), 'zip', build_dir)
