import pandas as pd
import mlflow
from lib_mlflow import *
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler, CIFAR100_Handler, EuroSAT_Handler, PCAM_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10, get_CIFAR100, get_EuroSAT, get_PCAM
from nets import Net, MNIST_Net, SVHN_Net, CIFAR10_Net, ResNet
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
                             LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
                             KMeansSampling, KCenterGreedy, BALDDropout, \
                             AdversarialBIM, AdversarialDeepFool, OracleUncertainty

batch_size=128
params = {'MNIST':
              {'n_epoch': 10,
               'num_classes': 10,
               'train_args':{'batch_size': 64},
               'test_args':{'batch_size': 1000},
               'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
          'FashionMNIST':
              {'n_epoch': 10,
               'num_classes': 10,
               'train_args':{'batch_size': 64},
               'test_args':{'batch_size': 1000},
               'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
          'SVHN':
              {'n_epoch': 20,
               'num_classes': 10,
               'train_args':{'batch_size': batch_size},
               'test_args':{'batch_size': 1000},
#                'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
               'optimizer_args':{'lr': 0.005, 'momentum': 0.9}}, # SGD
          'CIFAR10':
              {'n_epoch': 20,
               'num_classes': 10,
               'train_args':{'batch_size': batch_size},
               'test_args':{'batch_size': 1000},
#                'optimizer_args':{'lr': 0.05, 'momentum': 0.3}}
               'optimizer_args':{'lr': 0.005, 'momentum': 0.9}}, # SGD
#                'optimizer_args':{'lr': 0.01}} # Adam
          'CIFAR100':
              {'n_epoch': 20,
               'num_classes': 100,
               'train_args':{'batch_size': batch_size},
               'test_args':{'batch_size': 1000},
#                'optimizer_args':{'lr': 0.05, 'momentum': 0.3}}
               'optimizer_args':{'lr': 0.005, 'momentum': 0.9}}, # SGD
          'EuroSAT':
              {'n_epoch': 20,
               'num_classes': 10,
               'train_args':{'batch_size': batch_size},
               'test_args':{'batch_size': 1000},
#                'optimizer_args':{'lr': 0.05, 'momentum': 0.3}}
               'optimizer_args':{'lr': 0.001, 'momentum': 0.9}}, # SGD
          'PCAM':
              {'n_epoch': 20,
               'num_classes': 2,
               'train_args':{'batch_size': batch_size},
               'test_args':{'batch_size': 1000},
#                'optimizer_args':{'lr': 0.05, 'momentum': 0.3}}
               'optimizer_args':{'lr': 0.001, 'momentum': 0.9}}, # SGD
          }

def get_handler(name):
    if name == 'MNIST':
        return MNIST_Handler
    elif name == 'FashionMNIST':
        return MNIST_Handler
    elif name == 'SVHN':
        return SVHN_Handler
    elif name == 'CIFAR10':
        return CIFAR10_Handler
    elif name == 'CIFAR100':
        return CIFAR100_Handler
    elif name == 'EuroSAT':
        return EuroSAT_Handler
    elif name == 'PCAM':
        return PCAM_Handler

def get_dataset(name, root):
    handler_name = name
    if name == 'MNIST':
         return get_MNIST(get_handler(handler_name), root)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(get_handler(handler_name), root)
    elif name == 'SVHN':
        return get_SVHN(get_handler(handler_name), root)
    elif name == 'CIFAR10':
        return get_CIFAR10(get_handler(handler_name), root)
    elif name == 'CIFAR100':
        return get_CIFAR100(get_handler(handler_name), root)
    elif name == 'EuroSAT':
        return get_EuroSAT(get_handler(handler_name), root)
    elif name == 'PCAM':
        return get_PCAM(get_handler(handler_name), root)
    else:
        raise NotImplementedError

def get_net(name, device):
    curr_params = pd.json_normalize(params[name], sep='_')
    curr_params = curr_params.to_dict(orient='records')[0]
    del curr_params["n_epoch"]
    print(curr_params)
    retry(lambda: mlflow.log_params(curr_params))
    if name == 'MNIST':
        return Net(MNIST_Net, params[name], device)
        # return Net(ResNet, params[name], device)
    elif name == 'FashionMNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'SVHN':
#         return Net(SVHN_Net, params[name], device)
        return Net(ResNet, params[name], device)
    elif name == 'CIFAR10':
        # return Net(CIFAR10_Net, params[name], device)
        return Net(ResNet, params[name], device)
    elif name in ['CIFAR100', "EuroSAT", "PCAM"]:
        return Net(ResNet, params[name], device)
    else:
        raise NotImplementedError

def get_params(name):
    return params[name]

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "LeastConfidence":
        return LeastConfidence
    elif name == "OracleUncertainty":
        return OracleUncertainty
    elif name == "MarginSampling":
        return MarginSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "LeastConfidenceDropout":
        return LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        return MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "KMeansSampling":
        return KMeansSampling
    elif name == "KCenterGreedy":
        return KCenterGreedy
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialBIM":
        return AdversarialBIM
    elif name == "AdversarialDeepFool":
        return AdversarialDeepFool
    else:
        raise NotImplementedError

# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)

def get_confidences_file(root, dataset):
    if dataset not in ["CIFAR10", "CIFAR100", "SVHN", "EuroSAT", "PCAM"]:
        raise RuntimeError(f"Dataset {dataset} not supported for oracle uncertainty")
    return os.path.join(root, f"data/{dataset}/confidences.pkl")