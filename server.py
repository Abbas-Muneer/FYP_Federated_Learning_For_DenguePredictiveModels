from omegaconf import DictConfig
from model import Net, test
import torch
from collections import OrderedDict

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):

        return {'lr': config.lr, 'momentum': config.momentum,
                'local_epochs': config.local_epochs}
    
    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    def evaluate_fn(server_round: int, parameters, config):

        model = Net(num_classes)

        device = torch.device("cpu")

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy, prec, rec, f1 = test(model, testloader, device)

        print(f"→ Evaluation after Round")
        print(f"   Loss      : {loss:.4f}")
        print(f"   Accuracy  : {accuracy:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   F1-Score  : {f1:.4f}")


        return loss, {"accuracy": accuracy, "precision": prec, "recall": rec, "f1_score": f1}
        

        
    return evaluate_fn