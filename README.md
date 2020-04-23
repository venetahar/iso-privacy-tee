# iso-privacy-tee
Hosts experiments for Private Inference on Trusted Execution Environments

This repository contains the necessary logic to train and save a Keras model which is later used for evaluation.

In order to perform the evaluation on trusted hardware you can use the following library: 

https://github.com/capeprivacy/tf-trusted

You can follow the instructions in the README file for setting that project up. It's important to note that you need an older version of bazel for
it to build correctly  (I am using 0.24.1 and it seems to work).

To perform private inference do the following steps:

MALARIA Dataset:
1. Copy the contents of the ```tf_trusted_code``` directory to the ```tf_trusted/tf_trusted_custom_op/```
```cp -R tf_trusted_code/* $TF_TRUSTED_INSALL_PATH$/tf_trusted_custom_op/```
2. ```cd $TF_TRUSTED_INSALL_PATH$/tf_trusted_custom_op/```
3. Run: ```python3 tf_trusted_script.py```
4. Once step 3 is finished, run ```python3 tee_evaluation.py``` to check the reported accuracy.