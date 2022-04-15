from model_wrappers.model_wrapper import ModelWrapper
from model_wrappers.tensorflow_model_wrappers.tensorflow_wrapper import TensorFlowWrapper
from model_wrappers.tensorflow_model_wrappers.numpy_tensorflow_wrapper import NumpyTensorFlowWrapper
from model_wrappers.tensorflow_model_wrappers.csr_matrix_tensorflow_wrapper import CsrMatrixTensorFlowWrapper
from model_wrappers.tensorflow_model_wrappers.sequential_rnn_wrapper import SequentialRnnWrapper
from model_wrappers.tensorflow_model_wrappers.subclassed_rnn_wrapper import SubclassedRnnWrapper
from model_wrappers.tensorflow_model_wrappers.residual_rnn_wrapper import ResidualRnnWrapper
from model_wrappers.tensorflow_model_wrappers.nltk_rnn_wrapper import NltkRnnWrapper
from model_wrappers.tensorflow_model_wrappers.nltk_tfidf_wrapper import NltkTfidfWrapper
from model_wrappers.tensorflow_model_wrappers.attention_wrapper import AttentionWrapper
from model_wrappers.tensorflow_model_wrappers.isaac_rnn_wrapper import IsaacRnnWrapper
from model_wrappers.tensorflow_model_wrappers.isaac_attention_wrapper import IsaacAttentionWrapper

wrappers = {
    'sequential_rnn': SequentialRnnWrapper,
    'subclassed_rnn': SubclassedRnnWrapper,
    'residual_rnn': ResidualRnnWrapper,
    'nltk_rnn': NltkRnnWrapper,
    'nltk_tfidf': NltkTfidfWrapper,
    'attention': AttentionWrapper,
    'isaac_rnn': IsaacRnnWrapper,
    'isaac_attention': IsaacAttentionWrapper
}
