from pathlib import Path
import pickle
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import psutil
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
import time
import flwr as fl
from model import test


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    # 1️ Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    # 2️ Prepare Dataset
    trainloaders, validationloaders, testloader = prepare_dataset(cfg.num_clients, cfg.batch_size)

    # 3️ Define Clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    # 4️ Define FL Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.00001,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader)
    )


    # 5️ Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 2, 'num_gpus': 0.0},
    )


    # 6️ Save Training Results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'

    results = {"history": history, "notes": "Performance testing results"}
    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    # 7️ Extract Final Accuracy & Loss
    final_accuracy = history.metrics_centralized['accuracy'][-1][1]  # Latest accuracy
    final_loss = history.losses_centralized[-1][1]  # Latest loss
    final_prec = history.metrics_centralized["precision"][-1][1]
    final_rec = history.metrics_centralized["recall"][-1][1]
    final_f1 = history.metrics_centralized["f1_score"][-1][1]

    print(f"FINAL_ACCURACY: {final_accuracy}")
    print(f"FINAL_LOSS: {final_loss}")
    print(f"FINAL_PRECISION: {final_prec:.4f}")
    print(f"FINAL_RECALL: {final_rec:.4f}")
    print(f"FINAL_F1_SCORE: {final_f1:.4f}")

    
if __name__ == "__main__":
    main()
