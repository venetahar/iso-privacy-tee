from malaria.common.conv_pool_model import ConvPoolModel
from mnist.common.conv_model import ConvModel
from mnist.common.fully_connected_model import FullyConnectedModel

FULLY_CONNECTED_MODEL_TYPE = 'FullyConnected3'
CONV_MODEL_TYPE = 'Conv1'
CONV_POOL_MODEL_TYPE = 'Conv2Pool2'


class ModelFactory:
    """
    A Model Factory.
    """

    @staticmethod
    def create_model(model_type, input_shape, num_classes):
        """
        Returns a model of the appropriate type.
        :param model_type: The model type.
        :param input_shape: The input shape.
        :param num_classes: The number of classes.
        :return: An instantiated model.
        """
        if model_type == FULLY_CONNECTED_MODEL_TYPE:
            model = FullyConnectedModel(input_shape, dense_units=[128, 128], num_classes=num_classes).model
        elif model_type == CONV_MODEL_TYPE:
            model = ConvModel(input_shape, kernel_size=5, stride=2, out_channels=5, avg_pool_size=2,
                              dense_units=[100], num_classes=10).model
        elif model_type == CONV_POOL_MODEL_TYPE:
            model = ConvPoolModel(input_shape=input_shape, kernel_sizes=[5, 5], channels=[36, 36],
                                  stride=1, avg_pool_sizes=[2, 2], dense_units=[72], num_classes=2).model
        else:
            raise ValueError("Invalid model_type provided. ")

        print(model)
        return model
