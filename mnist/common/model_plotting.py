from tensorflow.keras import utils
from tensorflow.keras.models import load_model


new_model = load_model('../models/alice_conv_model')
utils.plot_model(
    new_model, to_file='model.png', show_shapes=True, show_layer_names=True,
    rankdir='TB', expand_nested=True, dpi=96)
