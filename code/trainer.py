import torch
import torch.nn as nn
from typing import Optional
from datetime import datetime
import os
from logger import logger
from torch.utils.data.dataset import Dataset
from scipy.linalg import eigh

from setup import init_trainining_results
from vae_model import Db_vae
from datasets.data_utils import DataLoaderTuple, DatasetOutput
import utils
from dataset import (
    make_hist_loader,
    make_train_and_valid_loaders,
    concat_datasets,
    sample_dataset,
)

from torchvision.utils import make_grid
from matplotlib import pyplot as plt

# New imports for Chow-Liu
import numpy as np
import networkx as nx
from sklearn.metrics import mutual_info_score
import tqdm


class Trainer:
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        hist_size: int,
        z_dim: int,
        alpha: float,
        num_bins: int,
        max_images: int,
        debias_type: str,
        device: str,
        lr: float = 0.001,
        eval_freq: int = 10,
        optimizer=torch.optim.Adam,
        load_model: bool = False,
        run_folder: Optional[str] = None,
        custom_encoding_layers: Optional[nn.Sequential] = None,
        custom_decoding_layers: Optional[nn.Sequential] = None,
        path_to_model: Optional[str] = None,
        config: Optional[dict] = None,
        bins: float = 100,
        smoothing_fac: float = 1e-4,
        temperature: float = 4.0,
        k: int = 20,
        **kwargs,
    ):
        """Wrapper class which trains a model."""
        init_trainining_results(config)
        self.epochs = epochs
        self.load_model = load_model
        self.z_dim = z_dim
        self.path_to_model = path_to_model
        self.batch_size = batch_size
        self.hist_size = hist_size
        self.alpha = alpha
        self.num_bins = num_bins
        self.debias_type = debias_type
        self.device = device
        self.eval_freq = eval_freq
        self.run_folder = run_folder

        self.k = k
        self.bins = bins
        self.smoothing_fac = smoothing_fac
        self.temperature = temperature

        self.config = config

        new_model: Db_vae = Db_vae(
            z_dim=z_dim,
            hist_size=hist_size,
            alpha=alpha,
            num_bins=num_bins,
            device=self.device,
        ).to(device=self.device)

        self.model = self.init_model()

        self.optimizer = optimizer(params=self.model.parameters(), lr=lr)

        train_loaders: DataLoaderTuple
        valid_loaders: DataLoaderTuple

        train_loaders, valid_loaders = make_train_and_valid_loaders(
            batch_size=batch_size, max_images=max_images, **kwargs
        )

        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders

    def init_model(self):
        # If model is loaded from file-system
        if self.load_model:
            if self.path_to_model is None:
                logger.error(
                    "Path has not been set.",
                    next_step="Model will not be initialized.",
                    tip="Set a path_to_model in your config.",
                )
                raise Exception

            if not os.path.exists(f"results/{self.path_to_model}"):
                logger.error(
                    f"Can't find model at results/{self.path_to_model}.",
                    next_step="Model will not be initialized.",
                    tip=f"Check if the directory results/{self.path_to_model} exists.",
                )
                raise Exception

            logger.info(f"Initializing model from {self.path_to_model}")
            return Db_vae.init(self.path_to_model, self.device, self.z_dim).to(
                self.device
            )

        # Model is newly initialized
        logger.info(
            f"Creating new model with the following parameters:\n"
            f"z_dim: {self.z_dim}\n"
            f"hist_size: {self.hist_size}\n"
            f"alpha: {self.alpha}\n"
            f"num_bins: {self.num_bins}\n"
        )

        return Db_vae(
            z_dim=self.z_dim,
            hist_size=self.hist_size,
            alpha=self.alpha,
            num_bins=self.num_bins,
            device=self.device,
        ).to(device=self.device)

    def train(self, epochs: Optional[int] = None):
        # Optionally use passed epochs
        epochs = self.epochs if epochs is None else epochs

        # Start training and validation cycle
        for epoch in range(epochs):
            epoch_start_t = datetime.now()
            logger.info(f"Starting epoch: {epoch+1}/{epochs}")

            self._update_sampling_histogram(epoch)

            # Training
            train_loss, train_acc = self._train_epoch()
            epoch_train_t = datetime.now() - epoch_start_t
            logger.info(f"epoch {epoch+1}/{epochs}::Training done")
            logger.info(
                f"epoch {epoch+1}/{epochs} => train_loss={train_loss:.2f}, train_acc={train_acc:.2f}"
            )

            # Validation
            logger.info("Starting validation")
            val_loss, val_acc = self._eval_epoch(epoch)
            epoch_val_t = datetime.now() - epoch_start_t
            logger.info(f"epoch {epoch+1}/{epochs}::Validation done")
            logger.info(
                f"epoch {epoch+1}/{epochs} => val_loss={val_loss:.2f}, val_acc={val_acc:.2f}"
            )

            # Print reconstruction
            valid_data = concat_datasets(
                self.valid_loaders.faces.dataset,
                self.valid_loaders.nonfaces.dataset,
                proportion_a=0.5,
            )
            self.print_reconstruction(self.model, valid_data, epoch, self.device)

            # Save model and scores
            self._save_epoch(epoch, train_loss, val_loss, train_acc, val_acc)

        logger.success(f"Finished training on {epochs} epochs.")

    def print_reconstruction(self, model, data, epoch, device, n_rows=4, save=True):
        # TODO: Add annotation
        model.eval()
        n_samples = n_rows**2

        images = sample_dataset(data, n_samples).to(device)

        recon_images = model.recon_images(images)

        fig = plt.figure(figsize=(16, 8))

        fig.add_subplot(1, 2, 1)
        grid = make_grid(images.reshape(n_samples, 3, 64, 64), n_rows)
        plt.imshow(grid.permute(1, 2, 0).cpu())

        utils.remove_frame(plt)

        fig.add_subplot(1, 2, 2)
        grid = make_grid(recon_images.reshape(n_samples, 3, 64, 64), n_rows)
        plt.imshow(grid.permute(1, 2, 0).cpu())

        utils.remove_frame(plt)

        if save:
            fig.savefig(
                f"results/{self.config.run_folder}/reconstructions/epoch={epoch}",
                bbox_inches="tight",
            )
            plt.close()
        else:
            return fig

    def _save_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
    ):
        """Writes training and validation scores to a csv, and stores a model to disk."""
        if not self.run_folder:
            logger.warning(
                "`--run_folder` could not be found.",
                "The program will continue, but won't save anything",
                "Double-check if --run_folder is configured.",
            )
            return

        # Write epoch metrics
        path_to_results = f"results/{self.run_folder}/training_results.csv"
        with open(path_to_results, "a") as wf:
            wf.write(f"{epoch}, {train_loss}, {val_loss}, {train_acc}, {val_acc}\n")

        # Write model to disk
        path_to_model = f"results/{self.run_folder}/model.pt"
        torch.save(self.model.state_dict(), path_to_model)

        logger.save(f"Stored model and results at results/{self.run_folder}")

    def visualize_bias(
        self, probs, data_loader, all_labels, all_index, epoch, n_rows=3
    ):
        # TODO: Add annotation
        n_samples = n_rows**2

        highest_probs = probs.argsort(descending=True)[:n_samples]
        lowest_probs = probs.argsort()[:n_samples]

        highest_imgs = utils.sample_idxs_from_loader(
            all_index[highest_probs], data_loader, 1
        )
        worst_imgs = utils.sample_idxs_from_loader(
            all_index[lowest_probs], data_loader, 1
        )

        img_list = (highest_imgs, worst_imgs)
        titles = ("Highest weights", "Lowest weights")
        fig = plt.figure(figsize=(16, 16))

        for i in range(2):
            ax = fig.add_subplot(1, 2, i + 1)
            grid = make_grid(img_list[i].reshape(n_samples, 3, 64, 64), n_rows)
            plt.imshow(grid.permute(1, 2, 0).cpu())
            ax.set_title(titles[i], fontdict={"fontsize": 30})

            utils.remove_frame(plt)

        path_to_results = f"results/{self.config.run_folder}/bias_probs/epoch={epoch}"
        logger.save(f"Saving a bias probability figure in {path_to_results}")

        fig.savefig(path_to_results, bbox_inches="tight")
        plt.close()

    def _eval_epoch(self, epoch):
        """Calculates the validation error of the model."""
        face_loader, nonface_loader = self.valid_loaders

        self.model.eval()
        avg_loss = 0
        avg_acc = 0

        all_labels = torch.tensor([], dtype=torch.long).to(self.device)
        all_preds = torch.tensor([], dtype=torch.long).to(self.device)
        all_idxs = torch.tensor([], dtype=torch.long).to(self.device)

        count = 0

        with torch.no_grad():
            for i, (face_batch, nonface_batch) in enumerate(
                zip(face_loader, nonface_loader)
            ):
                images, labels, idxs = utils.concat_batches(face_batch, nonface_batch)

                images = images.to(self.device)
                labels = labels.to(self.device)
                idxs = idxs.to(self.device)
                pred_logits, loss = self.model.forward(images, labels)

                # Convert logits to predicted labels
                pred_probs = torch.sigmoid(pred_logits)
                pred_labels = (pred_probs >= 0.5).long()

                loss = loss.mean()
                acc = utils.calculate_accuracy(labels, pred_labels)

                avg_loss += loss.item()
                avg_acc += acc

                all_labels = torch.cat((all_labels, labels))
                all_preds = torch.cat((all_preds, pred_labels))
                all_idxs = torch.cat((all_idxs, idxs))

                count = i

        best_faces, worst_faces, best_other, worst_other = (
            utils.get_best_and_worst_predictions(all_labels, all_preds, self.device)
        )
        self.visualize_best_and_worst(
            self.valid_loaders,
            all_labels,
            all_idxs,
            epoch,
            best_faces,
            worst_faces,
            best_other,
            worst_other,
        )

        return avg_loss / (count + 1), avg_acc / (count + 1)

    def _train_epoch(self):
        """Trains the model for one epoch."""
        face_loader, nonface_loader = self.train_loaders

        self.model.train()

        avg_loss: float = 0
        avg_acc: float = 0
        count: int = 0

        for i, (face_batch, nonface_batch) in enumerate(
            zip(face_loader, nonface_loader)
        ):
            images, labels, _ = utils.concat_batches(face_batch, nonface_batch)
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            pred_logits, loss = self.model.forward(images, labels)

            # Convert logits to predicted labels
            pred_probs = torch.sigmoid(pred_logits)
            pred_labels = (pred_probs >= 0.5).long()

            # Calculate the gradient, and clip at 5
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()

            # Calculate metrics
            acc = utils.calculate_accuracy(labels, pred_labels)
            avg_loss += loss.item()
            avg_acc += acc

            if i % self.eval_freq == 0:
                logger.info(f"Training: batch:{i} accuracy:{acc}")

            count = i

        return avg_loss / (count + 1), avg_acc / (count + 1)

    def _update_sampling_histogram(self, epoch: int):
        """Updates the data loader for faces to be proportional to how challenging each image is, if
        debias_type is not 'none'.
        """
        hist_loader = make_hist_loader(
            self.train_loaders.faces.dataset, self.batch_size
        )

        if self.debias_type != "none":
            self._update_histogram(hist_loader, epoch)
        else:
            self.train_loaders.faces.sampler.weights = torch.ones(
                len(self.train_loaders.faces.sampler.weights)
            )

    def _update_histogram(self, data_loader, epoch):
        """Updates the histogram of `self.model` based on the specified debias_type."""
        logger.info(f"Updating weight histogram using method: {self.debias_type}")

        if self.debias_type == "chow-liu":
            probs, all_labels, all_index = (
                self.get_training_sample_probabilities_chow_liu(data_loader)
            )
        elif self.debias_type == "full_gaussian":
            probs, all_labels, all_index = (
                self.get_training_sample_probabilities_full_gaussian(data_loader)
            )
        elif self.debias_type in ["max", "max5", "gaussian"]:
            self.model.means = torch.Tensor().to(self.device)
            self.model.std = torch.Tensor().to(self.device)

            all_labels = torch.tensor([], dtype=torch.long).to(self.device)
            all_index = torch.tensor([], dtype=torch.long).to(self.device)

            with torch.no_grad():
                for batch in data_loader:
                    images, labels, index, _ = batch
                    images, labels, index = (
                        images.to(self.device),
                        labels.to(self.device),
                        index.to(self.device),
                    )

                    all_labels = torch.cat((all_labels, labels))
                    all_index = torch.cat((all_index, index))

                    if self.debias_type in ["max", "max5"]:
                        self.model.build_means(images)
                    elif self.debias_type == "gaussian":
                        self.model.build_histo(images)

            if self.debias_type == "max":
                probs = self.model.get_histo_max()
            elif self.debias_type == "max5":
                probs = self.model.get_histo_max5()
            elif self.debias_type == "gaussian":
                probs = self.model.get_histo_gaussian()
        else:
            logger.error(
                "Unsupported debias_type!",
                next_step="The program will now close",
                tip="Set --debias_type to 'max', 'max5', 'gaussian', 'full_gaussian' or 'chow-liu'.",
            )
            raise Exception()

        # Apply the calculated probabilities to the sampler weights
        if self.debias_type != "none":
            self.train_loaders.faces.sampler.weights = probs
        else:
            self.train_loaders.faces.sampler.weights = torch.ones(
                len(self.train_loaders.faces.sampler.weights)
            )

        # Call visualize_bias for the relevant debiasing methods
        if self.debias_type in ["chow-liu", "full_gaussian"]:
            self.visualize_bias(probs, data_loader, all_labels, all_index, epoch)

    def get_training_sample_probabilities_chow_liu(
        self,
        loader,
    ):
        """
        Calculates training sample probabilities based on the Chow-Liu algorithm.
        Returns:
            training_sample_p (torch.Tensor): Normalized probabilities for sampling.
            all_labels (torch.Tensor): All labels collected during processing.
            all_index (torch.Tensor): All indices collected during processing.
        """
        self.model.eval()  # Set the model to evaluation mode
        mu_list = []
        all_labels = torch.tensor([], dtype=torch.long).to(self.device)
        all_index = torch.tensor([], dtype=torch.long).to(self.device)

        latent_dim = self.z_dim
        bins = self.bins
        smoothing_fac = self.smoothing_fac
        temperature = self.temperature

        count = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc="Calculating latent mu"):
                images, labels, index, _ = batch
                images = images.to(
                    self.device, non_blocking=True
                )  # Move images to the appropriate device
                _, mu_batch, _ = self.model.encode(
                    images
                )  # Get the latent mean for the batch
                mu_list.append(mu_batch.cpu().numpy())  # Move to CPU and store
                all_labels = torch.cat((all_labels, labels))
                all_index = torch.cat((all_index, index))
                count += self.batch_size
                # Optionally, add a debug mode to break early
                # if count > 2500:
                #     print(f"Breaking early...")
                #     break

        # Concatenate mu from all batches
        mu = np.concatenate(mu_list, axis=0)

        # Step 1: Discretize each latent dimension into bins
        discretized_mu = np.zeros_like(mu, dtype=int)
        bin_edges = []
        for i in range(latent_dim):
            min_val, max_val = mu[:, i].min(), mu[:, i].max()
            edges = np.linspace(min_val, max_val, bins + 1)
            bin_edges.append(edges)
            # Digitize and clamp the indices to stay within the valid range
            discretized_mu[:, i] = np.clip(
                np.digitize(mu[:, i], edges) - 1, 0, bins - 1
            )

        # Step 2: Calculate pairwise mutual information to build the Chow-Liu Tree
        mutual_info_matrix = np.zeros((latent_dim, latent_dim))
        for i in range(latent_dim):
            for j in range(i + 1, latent_dim):
                mi = mutual_info_score(discretized_mu[:, i], discretized_mu[:, j])
                mutual_info_matrix[i, j] = mi
                mutual_info_matrix[j, i] = mi

        # Build maximum spanning tree based on mutual information
        G = nx.Graph()
        for i in range(latent_dim):
            for j in range(i + 1, latent_dim):
                G.add_edge(i, j, weight=mutual_info_matrix[i, j])
        mst = nx.maximum_spanning_tree(G)

        # Step 3: Orient the MST and create pairwise histograms
        pairwise_histograms = {}
        root_node = list(mst.nodes())[0]  # Choose a root node

        # Orient the tree using BFS
        queue = [root_node]
        visited = set()
        directed_edges = []  # To store directed edges
        while queue:
            node = queue.pop(0)
            visited.add(node)
            for neighbor in mst.neighbors(node):
                if neighbor not in visited:
                    directed_edges.append((node, neighbor))
                    queue.append(neighbor)

        # Create pairwise histograms for directed edges
        for edge in directed_edges:
            i, j = edge
            hist_2d, _, _ = np.histogram2d(
                discretized_mu[:, i], discretized_mu[:, j], bins=bins
            )
            pairwise_histograms[(i, j)] = (
                hist_2d / hist_2d.sum()
            )  # Normalize to probabilities

        # Create marginal histogram for the root node
        marginal_histogram_root, _ = np.histogram(
            discretized_mu[:, root_node], bins=bins
        )
        marginal_histogram_root = (
            marginal_histogram_root / marginal_histogram_root.sum()
        )  # Normalize

        # Step 4: Calculate log probabilities for each sample using the directed tree
        log_training_sample_p = np.zeros(mu.shape[0])

        for sample_idx in range(mu.shape[0]):
            current_sample = discretized_mu[sample_idx]

            # Start with the root node's marginal probability
            root_bin = current_sample[root_node]
            log_p_sample = np.log(marginal_histogram_root[root_bin] + smoothing_fac)

            # Traverse directed edges
            for parent, child in directed_edges:
                parent_bin = current_sample[parent]
                child_bin = current_sample[child]
                # Ensure indices are clamped within the valid range
                parent_bin = np.clip(parent_bin, 0, bins - 1)
                child_bin = np.clip(child_bin, 0, bins - 1)
                conditional_prob = pairwise_histograms[(parent, child)][
                    parent_bin, child_bin
                ]
                log_p_sample += np.log(conditional_prob + smoothing_fac)

            # Adjust with temperature and accumulate
            log_training_sample_p[sample_idx] = -log_p_sample / temperature

        # Step 5: Convert log probabilities to normalized probabilities
        training_sample_p = np.exp(
            log_training_sample_p - np.max(log_training_sample_p)
        )
        training_sample_p /= training_sample_p.sum()

        return (
            torch.tensor(training_sample_p, dtype=torch.float32).to(self.device),
            all_labels,
            all_index,
        )

    def get_training_sample_probabilities_full_gaussian(
    self,
    loader,
    ):
        from scipy.stats import multivariate_normal

        latent_dim = self.z_dim  # Use the latent dimension from the model

        smoothing_fac = self.smoothing_fac
        temperature = self.temperature
        k = self.k

        self.model.eval()
        mu_list = []
        std_list = []
        all_labels = torch.tensor([], dtype=torch.long)
        all_index = torch.tensor([], dtype=torch.long)

        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc="Calculating latent mu and sigma"):
                images, labels, index, _ = batch
                images = images.to(self.device, non_blocking=True)
                _, mu_batch, std_batch = self.model.encode(images)
                mu_list.append(mu_batch.cpu().numpy())
                std_list.append(std_batch.cpu().numpy())
                all_labels = torch.cat((all_labels, labels.cpu()))
                all_index = torch.cat((all_index, index.cpu()))

        mu = np.concatenate(mu_list, axis=0)
        std = np.concatenate(std_list, axis=0)

        N = mu.shape[0]
        global_mu = np.mean(mu, axis=0)
        mu_centered = mu - global_mu

        # Compute covariance matrix using the sample covariance formula
        covariance = (mu_centered.T @ mu_centered) / N

        # Incorporate the mean of the variances into the covariance matrix
        avg_std2 = np.mean(std ** 2, axis=0)
        covariance += np.diag(avg_std2)

        # Enforce symmetry
        covariance = (covariance + covariance.T) / 2

        # Stabilize covariance matrix with regularization
        reg_term = 1e-6  # You may adjust this value as needed
        covariance += np.eye(latent_dim) * reg_term

        # Ensure covariance matrix is positive semidefinite
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Set negative eigenvalues to zero
        eigenvalues[eigenvalues < 0] = 0

        # Reconstruct the covariance matrix
        covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Optional low-rank approximation
        if k is not None and k < latent_dim:
            # Keep only the top k eigenvalues
            idx = np.argsort(eigenvalues)[::-1][:k]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Ensure covariance is symmetric
        covariance = (covariance + covariance.T) / 2

        # Use scipy.stats.multivariate_normal to compute log probabilities
        try:
            rv = multivariate_normal(
                mean=global_mu, cov=covariance, allow_singular=True
            )
            log_probs = rv.logpdf(mu)
        except ValueError as e:
            print(f"An error occurred: {e}")
            # Increase regularization if necessary
            reg_term = 1e-4
            covariance += np.eye(latent_dim) * reg_term
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues[eigenvalues < 0] = 0
            covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            covariance = (covariance + covariance.T) / 2
            rv = multivariate_normal(
                mean=global_mu, cov=covariance, allow_singular=True
            )
            log_probs = rv.logpdf(mu)

        # Compute energies (negative log probabilities)
        energies = -log_probs

        # Subtract max energy to prevent numerical overflow
        max_energy = np.max(energies)
        energies -= max_energy

        # Compute probabilities by exponentiating energies over temperature
        probs = np.exp(energies / temperature)

        # Add smoothing to avoid zeros
        probs += smoothing_fac

        # Normalize to sum to 1
        probs /= np.sum(probs)

        # Debugging: Log probability statistics
        logger.info(
            f"Probabilities stats: min={probs.min():.6e}, max={probs.max():.6e}, "
            f"mean={probs.mean():.6e}, std={probs.std():.6e}"
        )

        return (
            torch.tensor(probs, dtype=torch.float32).to(self.device),
            all_labels.to(self.device),
            all_index.to(self.device),
        )



    def sample(self, n_rows=4):
        n_samples = n_rows**2
        sample_images = self.model.sample(n_samples=n_samples)

        plt.figure(figsize=(n_rows * 2, n_rows * 2))
        grid = make_grid(sample_images.reshape(n_samples, 3, 64, 64), n_rows)
        plt.imshow(grid.permute(1, 2, 0).cpu())

        utils.remove_frame(plt)
        plt.show()

        return

    def reconstruction_samples(self, n_rows=4):
        valid_data = concat_datasets(
            self.valid_loaders.faces.dataset,
            self.valid_loaders.nonfaces.dataset,
            proportion_a=0.5,
        )
        fig = self.print_reconstruction(
            self.model, valid_data, 0, self.device, save=False
        )

        fig.show()

        return

    def visualize_best_and_worst(
        self,
        data_loaders,
        all_labels,
        all_indices,
        epoch,
        best_faces,
        worst_faces,
        best_other,
        worst_other,
        n_rows=4,
        save=True,
    ):
        # TODO: Add annotation
        n_samples = n_rows**2

        fig = plt.figure(figsize=(16, 16))

        sub_titles = ["Best faces", "Worst faces", "Best non-faces", "Worst non-faces"]
        for i, indices in enumerate((best_faces, worst_faces, best_other, worst_other)):
            labels_subset = all_labels[indices]
            indices_subset = all_indices[indices]
            images = utils.sample_idxs_from_loaders(
                indices_subset, data_loaders, labels_subset[0]
            )

            ax = fig.add_subplot(2, 2, i + 1)
            grid = make_grid(images.reshape(n_samples, 3, 64, 64), n_rows)
            plt.imshow(grid.permute(1, 2, 0).cpu())
            ax.set_title(sub_titles[i], fontdict={"fontsize": 30})

            utils.remove_frame(plt)

        if save:
            fig.savefig(
                f"results/{self.config.run_folder}/best_and_worst/epoch:{epoch}",
                bbox_inches="tight",
            )
            plt.close()
        else:
            return fig

    def best_and_worst(self, n_rows=4):
        """Calculates the validation error of the model."""
        face_loader, nonface_loader = self.valid_loaders

        self.model.eval()
        avg_loss = 0
        avg_acc = 0

        all_labels = torch.tensor([], dtype=torch.long).to(self.device)
        all_preds = torch.tensor([], dtype=torch.long).to(self.device)
        all_idxs = torch.tensor([], dtype=torch.long).to(self.device)

        count = 0

        with torch.no_grad():
            for i, (face_batch, nonface_batch) in enumerate(
                zip(face_loader, nonface_loader)
            ):
                images, labels, idxs = utils.concat_batches(face_batch, nonface_batch)

                images = images.to(self.device)
                labels = labels.to(self.device)
                idxs = idxs.to(self.device)
                pred, loss = self.model.forward(images, labels)

                loss = loss.mean()
                acc = utils.calculate_accuracy(labels, pred)

                avg_loss += loss.item()
                avg_acc += acc

                all_labels = torch.cat((all_labels, labels))
                all_preds = torch.cat((all_preds, pred))
                all_idxs = torch.cat((all_idxs, idxs))

                count = i

        best_faces, worst_faces, best_other, worst_other = (
            utils.get_best_and_worst_predictions(all_labels, all_preds, self.device)
        )
        fig = self.visualize_best_and_worst(
            self.valid_loaders,
            all_labels,
            all_idxs,
            0,
            best_faces,
            worst_faces,
            best_other,
            worst_other,
            save=False,
        )

        fig.show()

        return
