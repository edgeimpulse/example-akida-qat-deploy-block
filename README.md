# Akida Custom Deploy Block

This repository provides a Custom Deploy Block that will convert and quantize deep learning models trained with Edge Impulse for use with the [Akida inferencing library](https://doc.brainchipinc.com/user_guide/akida.html#inference), which supports Akida PCIe and RPi development kits, as well as software simulation of spiking neural nets.

This repostitory offers an early look at our partnership work with Brainchip - with sensor input, preprocessing blocks, and full integration with Edge Impulse tools coming soon!

# Using this repository

1. Chooose your favorite Edge Impulse image classification project. If you don't already have one, learn how to build your first ML model with Edge Impulse [here](https://docs.edgeimpulse.com/docs/tutorials/image-classification)

2. Delete any existing learn block, create a new `Classification (Keras)` block. Within the block, use Expert Mode to define a model architecture that is compatible with Akida. For guidance on model compatibility, see [the MetaTF documentation](https://doc.brainchipinc.com/api_reference/cnn2snn_apis.html#cnn2snn.check_model_compatibility).

3. Add the custom deploy blocks present in this repository to your account. To do this just install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) and run the following commands in the `meta_tf_deploy` directory.

```
edge-impulse-blocks init
edge-impulse-blocks push
```

Follow the prompts and select `Deployment Block` when prompted.

After pushing the deployment block, navigate to [Uploading your block to edge impulse](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/building-deployment-blocks#3.-uploading-the-deployment-block-to-edge-impulse) and follow the instructions to enable permissions for the deployment. **IMPORTANT** Enable `mount training block under /data`, as this allows direct access to the training data, which is used to perform quantization-aware fine tuning of the model.

4. Deploy your project. During deployment the model will be quantized for Akida inferencing, and then fine tuned in order to retain model performance. Observe the `trained.fbz` model artifact. This is your serialized akida model. The training and testing datasets are additionally provided as a test data source. Refer to MetaTF documentation [here](https://doc.brainchipinc.com/user_guide/akida.html#inference) for information on how to run your newly trained spiking NN.

5. If you wish to experiment with different values for learning rate and the number of epochs for fine tuning you can configure the `--learning_rate` and `--fine_tune_epochs` flags in the block configuration in your Edge Impulse Organization.