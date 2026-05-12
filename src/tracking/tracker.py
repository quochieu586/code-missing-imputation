class FCSTracker:
    def __init__(self):
        self.history = []
        
    def record(self, iteration, variable, params, imputed_values):
        self.history.append({
            "iteration": iteration,
            "variable": variable,
            "params": params,
            "imputed_values": imputed_values.copy()
        })