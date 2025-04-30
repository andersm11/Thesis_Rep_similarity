import torch.nn as nn

def petrosian_fd_batch(windows_np):  
    # windows_np shape: (N_windows, window_length)
    diff = np.diff(windows_np, axis=1)
    zero_cross = (diff[:, :-1] * diff[:, 1:] < 0).sum(axis=1)
    N = windows_np.shape[1]
    numerator = np.log10(N)
    denominator = numerator + np.log10(N / (N + 0.4 * zero_cross))
    return numerator / denominator

class FractalFeatureLayer(nn.Module):
    def __init__(self, input_length, num_scales=5, fd_func=petrosian_fd_batch):
        super().__init__()
        self.input_length = input_length
        self.num_scales = num_scales
        self.fd_func = fd_func
        self.window_sizes = self._compute_window_sizes()
        self.strides = [w // 2 for w in self.window_sizes]

    def _compute_window_sizes(self):
        return [max(4, int(self.input_length // (2 ** i))) for i in range(self.num_scales)]

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        all_fd = []

        x_cpu = x.cpu().numpy()

        for w, s in zip(self.window_sizes, self.strides):
            n_windows = max(1, (T - w) // s + 1)
            fd_vals = np.zeros((B, C, n_windows), dtype=np.float32)

            for b in range(B):
                for c in range(C):
                    signal = x_cpu[b, c]
                    # Use NumPy strided window extraction
                    windows = np.lib.stride_tricks.sliding_window_view(signal, w)[::s]
                    if windows.shape[0] > n_windows:
                        windows = windows[:n_windows]
                    fd = self.fd_func(windows)
                    fd_vals[b, c, :windows.shape[0]] = fd

            all_fd.append(torch.from_numpy(fd_vals))

        fd_out = torch.cat(all_fd, dim=2)  # (B, C, total_windows)
        return fd_out.to(x.device)

# Example: wrap your dataset in a DataLoader
train_loader = DataLoader(train_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

FL = FractalFeatureLayer(input_length=264)
FL_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FL = FL.to(FL_device)

all_fd_train = []
all_fd_test = []
# Helper to extract and store FD features
def extract_fd(loader):
    all_fd = []
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            data, labels,_ = batch  # (B, C, T), (B,)
            data = data.to(FL_device)
            fd = FL.forward(data)  # (B, C, total_windows)
            #fd_flat = fd.view(fd.size(0), -1)  # (B, C * total_windows)
            all_fd.append(fd.cpu())
            all_labels.append(labels.cpu())
            print(f"Processed batch {i}")
    features = torch.cat(all_fd, dim=0)       # (N, F)
    labels = torch.cat(all_labels, dim=0)     # (N,)
    return features, labels

frac_train, labels_train = extract_fd(train_loader)
frac_test, labels_test = extract_fd(test_loader)
min_val = frac_train.min(dim=0, keepdim=True).values
max_val = frac_train.max(dim=0, keepdim=True).values
range_val = max_val - min_val + 1e-6
frac_train_norm = (frac_train - min_val) / range_val
frac_test_norm = (frac_test - min_val) / range_val