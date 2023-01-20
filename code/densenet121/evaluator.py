import abc
from collections.abc import Iterable
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.metrics import SparseCategoricalCrossentropy as CE
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy as TopKAcc


class IEvaluator(object):

    def __init__(self, model, dataset, nsteps):
        self.model = model
        self.dataset = dataset
        self.nsteps = int(nsteps)
        assert self.nsteps > 0

        # Check dataset
        assert isinstance(self.dataset, Iterable), \
            ("class Evaluator: def __init__(...): "
             "Dataset: {}, - is not iterable".format(self.dataset))

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Image2LabelEvaluator(IEvaluator):

    def __init__(self, model, dataset, nsteps):
        super(Image2LabelEvaluator, self).__init__(model, dataset, nsteps)

        # Check model is tf.keras.Model
        assert isinstance(self.model, tf.keras.Model), ("class Image2LabelEvaluator: def __init__(...): "
                                                        "Model: {}, - is not tf.keras.Model".format(self.model))

        # Check dataset is tf.data.Dataset
        assert isinstance(self.dataset, tf.data.Dataset), ("class Image2LabelEvaluator: def __init__(...): "
                                                           "Dataset: {}, - is not tf.data.Dataset".format(self.dataset))

    def extract_inputs(self, inputs):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == 2
        assert isinstance(inputs[0], tf.Tensor) and isinstance(inputs[1], tf.Tensor)
        assert inputs[1].dtype == tf.int32

        return inputs

    def __call__(self, *args, **kwargs):
        top1 = TopKAcc(k=1, name="eval_top1_accuracy")
        top5 = TopKAcc(k=5, name="eval_top5_accuracy")
        ce = CE(from_logits=False, name="eval_cross_entropy")

        top1.reset_state()
        top5.reset_state()
        ce.reset_state()
        i = 0
        for inputs in tqdm(self.dataset, total=self.nsteps):
            # Extract data.
            inp_img_t, label_t = self.extract_inputs(inputs)

            # Inference model.
            pred_t = tf.nn.softmax(self.model(inp_img_t, training=False),
                                   axis=-1)

            # Accumulate Top-N (1,5) accuracy.
            top1.update_state(label_t, pred_t)
            top5.update_state(label_t, pred_t)
            ce.update_state(label_t, pred_t)

            # Update index.
            i = i + 1
            if i >= self.nsteps:
                break

        return top1.result() * 100.0, top5.result() * 100.0, ce.result()
