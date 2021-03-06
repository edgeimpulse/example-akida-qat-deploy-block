import json, os, shutil, logging, argparse

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.disable(logging.WARNING)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

from cnn2snn import check_model_compatibility, quantize, convert
import akida
from ascii_graph import Pyasciigraph

import training, profiling

# parse arguments (--metadata FILE is passed in)
parser = argparse.ArgumentParser(description='Custom deploy block demo')
parser.add_argument('--metadata', type=str)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--fine_tune_epochs', type=int, default=60)
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
shutil.unpack_archive(os.path.join(build_dir, 'trained.h5.zip'), build_dir, 'zip')
print('Copying files to build directory OK')
print('')

print('Loading model and checking compatibility...')
model = keras.models.load_model(os.path.join(build_dir, 'model.h5'))

compatible = check_model_compatibility(model, input_is_image=False)

if not compatible:
    exit(1)
print('Loading model and checking compatibility OK')
print('')

print('Loading and splitting dataset...')
X_train, X_test, Y_train, Y_test, _ = training.split_and_shuffle_data('npy', len(metadata['classes']),
                                                         None, 'classification', 3, '/data')

train_dataset = training.get_dataset_standard(X_train, Y_train)
validation_dataset = training.get_dataset_standard(X_test, Y_test)

input_shape = metadata['tfliteModels'][0]['details']['inputs'][0]['shape'][1:]

train_dataset = train_dataset.map(training.get_reshape_function(input_shape), tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.map(training.get_reshape_function(input_shape), tf.data.experimental.AUTOTUNE)

BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)
print('Loading and splitting dataset OK')
print('')

def profile_model(keras_model, description):
    print(f'Profiling {description} model...')
    report, accuracy, f1, loss = profiling.evaluate(keras_model, validation_dataset, Y_test, len(metadata['classes']))
    print(f'Accuracy: {accuracy}')
    print(f'F1 score: {f1}')
    print('')
    return accuracy

accuracy_float = profile_model(model, 'floating point')

print('Performing post-training quantization...')
model_quantized = quantize(model,
                           weight_quantization=4,
                           activ_quantization=4,
                           input_weight_quantization=8)
print('Performing post-training quantization OK')
print('')

accuracy_quantized = profile_model(model_quantized, 'post-training quantized')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  mode='max',
                                                  verbose=1,
                                                  min_delta=0,
                                                  patience=10,
                                                  restore_best_weights=True)

print('Fine-tuning to recover accuracy...')
print(f'Using learning rate {args.learning_rate}')
model_quantized.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model_quantized.fit(train_dataset,
                epochs=args.fine_tune_epochs,
                verbose=2,
                validation_data=validation_dataset,
                callbacks=[early_stopping]
            )

print('Fine-tuning to recover accuracy OK')
print('')

accuracy_quantized_trained = profile_model(model_quantized, 'quantization-aware trained')

stats = [('Float', accuracy_float),
         ('Quantized', accuracy_quantized),
         ('QAT', accuracy_quantized_trained)]

graph = Pyasciigraph(line_length=50)
for line in graph.graph('Comparison of model accuracy', stats):
    print(line)

print(f'Float:                   {accuracy_float}')
print(f'Quantized:               {accuracy_quantized}')
print(f'QAT:                     {accuracy_quantized_trained}')
diff = accuracy_quantized_trained - accuracy_float
if diff > 0:
    symbol = '+'
else:
    symbol = ''
print(f'Difference (float->QAT): {symbol}{diff}')
print('')

print('Converting to Akida model...')
print('')
model_akida = convert(model_quantized, input_is_image=True)

model_akida.map(akida.AKD1000())

model_akida.summary()
print('Converting to Akida model OK')
print('')

print('Saving models...')
# remove everything besides quantized model and akida_model
os.system('rm -r ' + build_dir)
model_quantized.save(os.path.join(build_dir, 'saved_model_keras_quantized'))
model_akida.save(os.path.join(build_dir, 'akida_model.fbz'))

shutil.make_archive(os.path.join(output_dir, 'akida_deployment'), 'zip', build_dir)
print('Saving models OK...')
