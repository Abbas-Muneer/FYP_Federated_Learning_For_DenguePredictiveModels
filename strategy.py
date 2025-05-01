import flwr as fl
import numpy as np

class ContributionTrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_contributions = {}

    def aggregate_fit(self, server_round, results, failures):
     # Call the parent method to aggregate updates
        aggregated_weights = super().aggregate_fit(server_round, results, failures)

        if not results:
            return aggregated_weights

    # Calculating contributions for each client
        self.client_contributions = {}
        for idx, (client_weights, fit_metrics) in enumerate(results):
            # Convert client_weights to numpy arrays if they are in tensor form
            if hasattr(client_weights, 'weights'):  
                client_weights = client_weights.weights  # Get the actual weights
        
            #
            if isinstance(client_weights, tuple):
                client_weights = [w.numpy() if hasattr(w, 'numpy') else w for w in client_weights]

            # 
            if isinstance(aggregated_weights, tuple):
                aggregated_weights = [w.numpy() if hasattr(w, 'numpy') else w for w in aggregated_weights]

             # Calculate magnitude of updates for each client
            client_update = [
                np.linalg.norm(cw - aw)
                for cw, aw in zip(client_weights, aggregated_weights)
            ]
            contribution = np.sum(client_update)
            self.client_contributions[idx + 1] = contribution

        return aggregated_weights
    

    def print_client_contributions(self):
        """Prints client contributions in a clean format."""
        print("\nClient Contributions (based on update magnitudes):")
        total_contribution = sum(self.client_contributions.values())
        for client_id, contribution in self.client_contributions.items():
            normalized_contribution = (contribution / total_contribution) * 100
            print(f"Client {client_id}: {normalized_contribution:.2f}%")
    
    def evaluate(self, server_round, parameters):
        # Call the parent evaluate method
        eval_result = super().evaluate(server_round, parameters)

        # Log contributions after evaluation
        print("\nClient Contributions (based on update magnitudes):")
        #for client_id, contribution in self.client_contributions.items():
            #print(f"Client {client_id}: {contribution:.4f}")

        return eval_result
