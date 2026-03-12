import torch

from ml_planner.placenav.topomap import load_topomap


class PlaceRecognition:
    ACTION_TO_COMMAND = {
        'roadside': 0,
        'straight': 1,
        'left': 2,
        'right': 3,
    }

    def __init__(
        self,
        weight_path,
        topomap_path,
        delta=5.0,
        window_lower=-2,
        window_upper=10,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(weight_path, map_location=self.device)
        self.model.eval()

        topomap = load_topomap(topomap_path, self.device)
        self.node_ids = topomap['node_ids']
        self.actions = topomap['actions']
        self.feature_matrix = topomap['feature_matrix']

        self.delta = float(delta)
        self.window_lower = int(window_lower)
        self.window_upper = int(window_upper)
        self.lambda1 = None
        self.belief = None
        self.transition_matrix = self._build_transition_matrix(len(self.node_ids))

    def _build_transition_matrix(self, num_nodes):
        transition = torch.zeros((num_nodes, num_nodes), device=self.device, dtype=torch.float32)
        for src in range(num_nodes):
            lower = max(0, src + self.window_lower)
            upper = min(num_nodes, src + self.window_upper)
            if upper <= lower:
                transition[src, src] = 1.0
                continue
            transition[src, lower:upper] = 1.0

        row_sums = transition.sum(dim=1, keepdim=True).clamp_min(1.0)
        return transition / row_sums

    def _compute_scores(self, query_feature):
        return self.feature_matrix @ query_feature

    def _initialize_belief(self, scores):
        score_quantiles = torch.quantile(scores, torch.tensor([0.025, 0.975], device=self.device))
        score_span = (score_quantiles[1] - score_quantiles[0]).clamp_min(1e-6)
        self.lambda1 = torch.log(torch.tensor(self.delta, device=self.device)) / score_span

        belief = torch.exp(self.lambda1 * (scores - score_quantiles[1]))
        self.belief = belief / belief.sum().clamp_min(1e-12)

    def _update_belief(self, scores):
        obs_likelihood = torch.exp(self.lambda1 * scores)
        self.belief = self.transition_matrix.transpose(0, 1) @ self.belief
        self.belief = self.belief * obs_likelihood
        self.belief = self.belief / self.belief.sum().clamp_min(1e-12)

    def get_recognition(self, image_tensor):
        image_tensor = image_tensor.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(image_tensor)

        query_feature = output.squeeze(0).flatten()
        scores = self._compute_scores(query_feature)

        if self.belief is None:
            self._initialize_belief(scores)
        else:
            self._update_belief(scores)

        best_idx = int(torch.argmax(self.belief).item())
        action = self.actions[best_idx]
        if action not in self.ACTION_TO_COMMAND:
            raise ValueError(f'Unsupported action in topomap: {action}')

        return self.ACTION_TO_COMMAND[action]
