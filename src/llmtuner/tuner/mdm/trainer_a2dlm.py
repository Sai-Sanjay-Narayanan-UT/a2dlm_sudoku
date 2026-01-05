import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from transformers import Trainer
from torch import nn
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.nn.functional as F

logger = get_logger(__name__)

class CustomDiffusionTrainer(Trainer):
    def __init__(
        self,
        diff_args,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.diff_args = diff_args
        print(self.diff_args)
        
        # A2DLM: Augmented vocabulary setup
        self.vocab_size = self.tokenizer.vocab_size
        self.augmented_vocab_size = 2 * self.vocab_size + 1  # regular + important + mask
        self.mask_index = 2 * self.vocab_size  # mask token at the end
        
        # A2DLM hyperparameters (can be overridden in diff_args)
        self.kappa = getattr(self.diff_args, 'kappa', 0.5)
        self.sigma_noise = getattr(self.diff_args, 'sigma_noise', 0.1)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes diffusion loss.
        """
        final_loss = self.inner_forward(model, inputs)
        return final_loss
    
    def q_sample_y(self, y_0, t, t_prime, maskable_mask):
        """A2DLM version of q_sample with stochastic per-token masking.
        
        Args:
            y_0: Tokens in augmented space [batch, seq_len]
            t: Discrete timestep in [0, T-1]
            t_prime: Normalized time in [0, 1], where t_prime = (t+1)/T
            maskable_mask: Boolean mask, True for solution tokens
            
        Returns:
            y_t: Noised sequence
            t: Timestep tensor
            t_mask: Boolean mask of masked positions
        """
        batch_size, seq_len = y_0.shape
        
        # Compute fraction of regular tokens (r)
        r = self._compute_regular_fraction(y_0)  # [batch]
        
        # Determine if each token is important
        is_important = (y_0 >= self.vocab_size) & (y_0 < 2 * self.vocab_size)  # [batch, seq_len]
        
        # Expand t_prime and r for broadcasting
        if t_prime.ndim == 1:
            t_prime_expanded = t_prime.unsqueeze(-1)  # [batch, 1]
        else:
            t_prime_expanded = t_prime
        r_expanded = r.unsqueeze(-1)  # [batch, 1]
        
        # Compute beta means for regular and important tokens
        # beta_{0,t'} = t' + kappa * t' * (1-t') * (1-r)
        # beta_{1,t'} = t' - kappa * t' * (1-t') * r
        beta_mean_regular = t_prime_expanded + self.kappa * t_prime_expanded * (1 - t_prime_expanded) * (1 - r_expanded)
        beta_mean_important = t_prime_expanded - self.kappa * t_prime_expanded * (1 - t_prime_expanded) * r_expanded
        
        # Expand to [batch, seq_len]
        beta_mean_regular_expanded = beta_mean_regular.expand(-1, seq_len)
        beta_mean_important_expanded = beta_mean_important.expand(-1, seq_len)
        
        # Select appropriate beta_mean for each token
        beta_mean = torch.where(is_important, beta_mean_important_expanded, beta_mean_regular_expanded)
        
        # Sample masking probabilities from Gaussian
        variance = self.sigma_noise ** 2 * t_prime_expanded * (1 - t_prime_expanded)
        std = torch.sqrt(variance)
        noise = torch.randn_like(beta_mean)
        beta_sampled = beta_mean + std * noise
        beta_sampled = torch.clamp(beta_sampled, 0.0, 1.0)
        
        # Apply stochastic masking
        u = torch.rand_like(y_0, dtype=torch.float)
        t_mask = (u < beta_sampled) & maskable_mask
        y_t = y_0.masked_fill(t_mask, self.mask_index)
        
        return y_t, t, t_mask

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        rewrite prediction_step for eval loss
        """
        model.eval()
        labels = inputs['input_ids'].masked_fill(inputs['src_mask'].bool(), self.tokenizer.pad_token_id)
        with torch.no_grad():
            # import pdb; pdb.set_trace();
            final_loss = self.inner_forward(model, inputs)
            if prediction_loss_only:
                preds = None
            else:
                preds = self.generate_samples(inputs)
        # ignore the source part when calculating metric and saving
        preds = preds.masked_fill(inputs['src_mask'].bool(), self.tokenizer.pad_token_id)
        return final_loss, preds, labels
    
    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))

    def inner_forward(
        self,
        model,
        inputs
    ):
        """A2DLM forward pass with augmented vocabulary and 3-term loss."""
        x = inputs["input_ids"]  # Original tokens [batch, seq_len]
        src_mask = inputs["src_mask"].bool()
        batch_size = x.size(0)
        num_timesteps = self.diff_args.diffusion_steps

        # Sample timestep
        t = torch.randint(0, num_timesteps, (batch_size, ), device=x.device)
        t_prime = (t + 1).float() / num_timesteps  # Normalized time in [0, 1]
        
        # Convert x to y (augmented space)
        importance_mask = self.create_sequence_mask(x)  # [batch, seq_len]
        y_0 = x + importance_mask.long() * self.vocab_size  # Offset important tokens
        
        # Compute fraction of regular tokens (ground truth r)
        r = self._compute_regular_fraction(y_0)  # [batch]
        
        # Forward diffusion: y_0 -> y_t
        y_t, t, loss_mask = self.q_sample_y(y_0, t, t_prime, maskable_mask=~src_mask)
        
        # Model forward pass (model should output augmented_vocab_size logits)
        attention_mask = torch.ones_like(y_t)
        logits = model(y_t, t, attention_mask=attention_mask)  # [batch, seq_len, augmented_vocab_size]
        
        # Subs parameterization: set mask_index logit to -inf and renormalize
        logits[:, :, self.mask_index] = float('-inf')  # Suppress mask token (use -1e4 for fp16 compatibility)
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # Renormalize
        
        # Get log probabilities
        log_p_theta = F.log_softmax(logits, dim=-1)  # [batch, seq_len, augmented_vocab_size]
        p_y_theta = log_p_theta.exp()
        
        # Gather log prob for ground truth tokens
        log_p_theta_y0 = torch.gather(log_p_theta, dim=-1, index=y_0.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
        
        # Compute beta values and f_t coefficients (using ground truth r)
        t_prime_expanded = t_prime.unsqueeze(-1)  # [batch, 1]
        r_expanded = r.unsqueeze(-1)  # [batch, 1]
        
        # For discrete time, s' = t' - dt' where dt' = 1/T
        dt_prime = 1.0 / num_timesteps
        s_prime_expanded = torch.clamp(t_prime_expanded - dt_prime, min=0.0)  # Clamp to avoid negative time
        
        # Beta means at time t' (ground truth r)
        beta_0_t = t_prime_expanded + self.kappa * t_prime_expanded * (1 - t_prime_expanded) * (1 - r_expanded)
        beta_1_t = t_prime_expanded - self.kappa * t_prime_expanded * (1 - t_prime_expanded) * r_expanded
        
        # Beta means at time s' (ground truth r)
        beta_0_s = s_prime_expanded + self.kappa * s_prime_expanded * (1 - s_prime_expanded) * (1 - r_expanded)
        beta_1_s = s_prime_expanded - self.kappa * s_prime_expanded * (1 - s_prime_expanded) * r_expanded
        
        # Clamp to avoid negative betas
        beta_0_t = torch.clamp(beta_0_t, min=1e-8)
        beta_1_t = torch.clamp(beta_1_t, min=1e-8)
        beta_0_s = torch.clamp(beta_0_s, min=0.0)
        beta_1_s = torch.clamp(beta_1_s, min=0.0)
        
        # f_t coefficients for ground truth (f_t_actual)
        # Use max to avoid division issues when beta_0_t ≈ beta_0_s
        f_t_regular_actual = torch.clamp((beta_0_t - beta_0_s) / beta_0_t, min=0.0, max=1.0)  # [batch, 1]
        f_t_important_actual = torch.clamp((beta_1_t - beta_1_s) / beta_1_t, min=0.0, max=1.0)  # [batch, 1]
        
        # Get f_t_actual based on importance bit of ground truth
        importance_bit = y_0 // self.vocab_size  # [batch, seq_len]
        f_t_actual = torch.where(importance_bit == 0,
                                 f_t_regular_actual.expand(-1, y_0.shape[1]),
                                 f_t_important_actual.expand(-1, y_0.shape[1]))  # [batch, seq_len]
        
        # Compute r from model prediction (only over solution tokens 83-163)
        prob_important_per_position = p_y_theta[:, :, self.vocab_size:2*self.vocab_size].sum(dim=-1)  # [batch, seq_len]
        one_minus_r_model = prob_important_per_position[:, 83:164].mean(dim=1)  # [batch] - mean over solution only
        r_model = 1 - one_minus_r_model
        r_model_expanded = r_model.unsqueeze(-1)  # [batch, 1]
        
        # Beta means at time t' using model's r
        beta_0_t_model = t_prime_expanded + self.kappa * t_prime_expanded * (1 - t_prime_expanded) * (1 - r_model_expanded)
        beta_1_t_model = t_prime_expanded - self.kappa * t_prime_expanded * (1 - t_prime_expanded) * r_model_expanded
        
        # Beta means at time s' using model's r
        beta_0_s_model = s_prime_expanded + self.kappa * s_prime_expanded * (1 - s_prime_expanded) * (1 - r_model_expanded)
        beta_1_s_model = s_prime_expanded - self.kappa * s_prime_expanded * (1 - s_prime_expanded) * r_model_expanded
        
        # Clamp model betas
        beta_0_t_model = torch.clamp(beta_0_t_model, min=1e-8)
        beta_1_t_model = torch.clamp(beta_1_t_model, min=1e-8)
        beta_0_s_model = torch.clamp(beta_0_s_model, min=0.0)
        beta_1_s_model = torch.clamp(beta_1_s_model, min=0.0)
        
        # f_t coefficients using model's r
        f_t_regular_model = torch.clamp((beta_0_t_model - beta_0_s_model) / beta_0_t_model, min=0.0, max=1.0)
        f_t_important_model = torch.clamp((beta_1_t_model - beta_1_s_model) / beta_1_t_model, min=0.0, max=1.0)
        
        # Compute f_t_smooth (per-token coefficient for ground truth token)
        p_regular = p_y_theta[:, :, :self.vocab_size]  # [batch, seq_len, vocab_size]
        p_important = p_y_theta[:, :, self.vocab_size:2*self.vocab_size]  # [batch, seq_len, vocab_size]
        p_total = p_regular + p_important
        g = torch.where(p_total > 0, p_important / p_total, torch.full_like(p_important, 0.5))  # [batch, seq_len, vocab_size]
        
        # Per-token coefficient: broadcast f_t to [batch, seq_len, vocab_size]
        # f_t_regular_model and f_t_important_model are [batch, 1]
        f_t_reg_expanded = f_t_regular_model.unsqueeze(-1)  # [batch, 1, 1]
        f_t_imp_expanded = f_t_important_model.unsqueeze(-1)  # [batch, 1, 1]
        coefficients_smooth = f_t_reg_expanded * (1 - g) + f_t_imp_expanded * g  # [batch, seq_len, vocab_size]
        base_indices = y_0 % self.vocab_size  # [batch, seq_len]
        f_t_smooth = torch.gather(coefficients_smooth, dim=-1, index=base_indices.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
        
        # Compute f_t_position_avg (position-level average)
        S = p_y_theta[:, :, self.vocab_size:2*self.vocab_size].sum(dim=-1)  # [batch, seq_len]
        f_t_position_avg = f_t_regular_model * (1 - S) + f_t_important_model * S  # [batch, seq_len]
        
        # Compute 3-term loss (with proper numerical stability)
        loss_term1 = -f_t_actual * log_p_theta_y0  # [batch, seq_len]
        
        # Term 2: f_t * log(f_t / f_t_smooth)
        # Guard against f_t ≈ 0
        loss_term2 = torch.where(
            f_t_actual > 1e-6,
            f_t_actual * torch.log(torch.clamp(f_t_actual / (f_t_smooth + 1e-8), min=1e-10)),
            torch.zeros_like(f_t_actual)
        )
        
        # Term 3: (1 - f_t) * log((1 - f_t) / (1 - f_t_avg))
        # Guard against both (1 - f_t) ≈ 0 and (1 - f_t_avg) ≈ 0
        one_minus_f_actual = torch.clamp(1 - f_t_actual, min=1e-8, max=1.0)
        one_minus_f_avg = torch.clamp(1 - f_t_position_avg, min=1e-8, max=1.0)
        
        loss_term3 = torch.where(
            (1 - f_t_actual) > 1e-6,
            one_minus_f_actual * torch.log(one_minus_f_actual / one_minus_f_avg),
            torch.zeros_like(f_t_actual)
        )
        
        # loss = loss_term1 + loss_term2 + loss_term3  # [batch, seq_len]

        # testing out a simpler loss by just using the first term
        loss = loss_term1
        
        # Average over batch and sequence (only on masked positions during forward diffusion)
        loss = loss.masked_fill(~loss_mask, 0)  # Zero out non-masked positions
        
        # Check if any tokens were masked (avoid division by zero)
        num_masked = loss_mask.sum()
        if num_masked > 0:
            loss = loss.sum() / num_masked
        else:
            # If no tokens masked, return a small loss to avoid training issues
            loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        
        return loss


    def generate_samples(self, inputs):
        """
        A2DLM sampling: MDLM-style sampling with augmented vocabulary
        Following _a2dlm_ddpm_caching_update from a2dlm_diffusion_v6.py
        """
        self.model.cuda()
        self.model.eval()
        verbose = not self.is_in_train
        
        x = inputs['input_ids'].cuda()  # Original tokens [batch, seq_len]
        src_mask = inputs['src_mask'].bool().cuda()
        attention_mask = torch.ones_like(x) 
        batch_size = x.size(0)
        num_timesteps = self.diff_args.diffusion_steps

        # Convert to augmented space
        importance_mask = self.create_sequence_mask(x)  # [batch, seq_len]
        y_0_true = x + importance_mask.long().cuda() * self.vocab_size  # Ground truth in augmented space
        
        init_maskable_mask = maskable_mask = ~src_mask
        
        # Initialize: all solution tokens as [MASK]
        y_t = y_0_true.masked_fill(maskable_mask, self.mask_index)
        
        # Time step size dt' in [0, 1]
        dt_prime = 1.0 / num_timesteps
        
        for step_idx in range(num_timesteps-1, -1, -1):  # t from T-1 to 0
            with torch.no_grad():
                t = step_idx  # Discrete timestep
                t_prime = (t + 1.0) / num_timesteps  # Normalized time in (0, 1]
                
                if verbose:
                    # Decode back to original space for printing
                    x_t_display = y_t % self.vocab_size
                    x_t_display = torch.where(y_t == self.mask_index, 
                                             torch.full_like(x_t_display, self.tokenizer.mask_token_id),
                                             x_t_display)
                    print(f"t={t+1}(in):", self.tokenizer.decode(x_t_display.tolist()[0]))

                # Model forward pass
                t_tensor = torch.full((batch_size, ), t, device=x.device)
                logits = self.model(y_t, t_tensor, attention_mask=attention_mask)  # [batch, seq_len, augmented_vocab_size]

                # Subs parameterization: set mask_index logit to -inf and renormalize
                logits[:, :, self.mask_index] = float('-inf')  # Suppress mask token (use -1e4 for fp16 compatibility)
                logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # Renormalize

                # Get probabilities
                log_p_y0 = F.log_softmax(logits, dim=-1)  # [batch, seq_len, augmented_vocab_size]
                p_y0 = log_p_y0.exp()
                
                if verbose:
                    # Show predicted y0
                    y_0_pred = p_y0.argmax(dim=-1)
                    x_0_display = y_0_pred % self.vocab_size
                    print(f"t={t+1}(out):", self.tokenizer.decode(x_0_display.tolist()[0]))
                
                if t > 0:
                    if self.diff_args.topk_decoding:
                        # topk_decoding for A2DLM: confidence-based remasking
                        # Compute confidence score from q_ys distribution
                        
                        # First, compute q_ys (same as MDLM)
                        S = p_y0[:, :, self.vocab_size:2*self.vocab_size].sum(dim=-1, keepdim=True)  # [batch, seq_len, 1]
                        one_minus_r = S[:, 83:164, :].mean(dim=1)  # [batch, 1] - mean over solution only
                        r = 1 - one_minus_r  # [batch, 1]
                        
                        t_prime_tensor = torch.full((batch_size,), t_prime, device=x.device).unsqueeze(-1)  # [batch, 1]
                        s_prime = t_prime - dt_prime
                        s_prime_tensor = torch.full((batch_size,), s_prime, device=x.device).unsqueeze(-1)  # [batch, 1]
                        
                        beta_0_t = t_prime_tensor + self.kappa * t_prime_tensor * (1 - t_prime_tensor) * (1 - r)
                        beta_1_t = t_prime_tensor - self.kappa * t_prime_tensor * (1 - t_prime_tensor) * r
                        beta_0_s = s_prime_tensor + self.kappa * s_prime_tensor * (1 - s_prime_tensor) * (1 - r)
                        beta_1_s = s_prime_tensor - self.kappa * s_prime_tensor * (1 - s_prime_tensor) * r
                        
                        beta_0_t = torch.clamp(beta_0_t, min=1e-8).unsqueeze(1)
                        beta_1_t = torch.clamp(beta_1_t, min=1e-8).unsqueeze(1)
                        beta_0_s = beta_0_s.unsqueeze(1)
                        beta_1_s = beta_1_s.unsqueeze(1)
                        
                        f_t_regular = (beta_0_t - beta_0_s) / beta_0_t
                        f_t_important = (beta_1_t - beta_1_s) / beta_1_t
                        
                        p_regular = p_y0[:, :, :self.vocab_size]
                        p_important = p_y0[:, :, self.vocab_size:2*self.vocab_size]
                        p_total = p_regular + p_important
                        g = torch.where(p_total > 0, p_important / p_total, torch.full_like(p_important, 0.5))
                        
                        coefficients = f_t_regular * (1 - g) + f_t_important * g
                        
                        q_ys = torch.zeros_like(p_y0)
                        q_ys[:, :, :self.vocab_size] = p_regular * coefficients
                        q_ys[:, :, self.vocab_size:2*self.vocab_size] = p_important * coefficients
                        sum_regular = q_ys[:, :, :self.vocab_size].sum(dim=-1)
                        sum_important = q_ys[:, :, self.vocab_size:2*self.vocab_size].sum(dim=-1)
                        q_ys[:, :, self.mask_index] = 1 - (sum_regular + sum_important)
                        
                        # Sample y_0 from q_ys, EXCLUDING mask token (only sample actual tokens)
                        # Set mask probability to 0 before sampling
                        q_ys_for_sampling = q_ys.clone()
                        q_ys_for_sampling[:, :, self.mask_index] = 0
                        # Renormalize
                        q_ys_for_sampling = q_ys_for_sampling / (q_ys_for_sampling.sum(dim=-1, keepdim=True) + 1e-10)
                        y_0 = q_ys_for_sampling.argmax(dim=-1)
                        
                        # Keep non-masked predictions (preserve tokens from y_t for unmasked positions)
                        y_0 = torch.where(y_t == self.mask_index, y_0, y_t)
                        
                        # Compute confidence scores: EXCLUDE mask token (like trainer.py does)
                        q_ys_scores = torch.log(q_ys + 1e-10)  # [batch, seq_len, vocab]
                        q_ys_scores[:, :, self.mask_index] = -1000  # Exclude mask from scoring
                        y_0_scores = q_ys_scores.max(dim=-1)[0]  # [batch, seq_len]
                        
                        # Apply topk_decoding (returns y_t in augmented space)
                        y_t = topk_decoding(
                            y_0,
                            y_0_scores,
                            self.diff_args.decoding_strategy,
                            init_maskable_mask,
                            t,
                            num_timesteps,
                            self.mask_index  # Use augmented mask index
                        )
                        # y_t is already in augmented space - no conversion needed
                    else:
                        # MDLM sampling update following a2dlm_diffusion_v6
                        # Compute S = sum of probabilities over important tokens per position
                        S = p_y0[:, :, self.vocab_size:2*self.vocab_size].sum(dim=-1, keepdim=True)  # [batch, seq_len, 1]
                        
                        # Compute 1-r = average of S across solution positions only (83-163)
                        one_minus_r = S[:, 83:164, :].mean(dim=1)  # [batch, 1] - mean over solution only
                        r = 1 - one_minus_r  # [batch, 1]
                        
                        # Compute t' and s'
                        t_prime_tensor = torch.full((batch_size,), t_prime, device=x.device).unsqueeze(-1)  # [batch, 1]
                        s_prime = t_prime - dt_prime
                        s_prime_tensor = torch.full((batch_size,), s_prime, device=x.device).unsqueeze(-1)  # [batch, 1]
                        
                        # Compute beta values at t' and s'
                        beta_0_t = t_prime_tensor + self.kappa * t_prime_tensor * (1 - t_prime_tensor) * (1 - r)  # [batch, 1]
                        beta_1_t = t_prime_tensor - self.kappa * t_prime_tensor * (1 - t_prime_tensor) * r  # [batch, 1]
                        beta_0_s = s_prime_tensor + self.kappa * s_prime_tensor * (1 - s_prime_tensor) * (1 - r)  # [batch, 1]
                        beta_1_s = s_prime_tensor - self.kappa * s_prime_tensor * (1 - s_prime_tensor) * r  # [batch, 1]
                        
                        # Clamp to avoid numerical issues
                        beta_0_t = torch.clamp(beta_0_t, min=1e-8)
                        beta_1_t = torch.clamp(beta_1_t, min=1e-8)
                        
                        # Expand to [batch, 1, 1] for broadcasting
                        beta_0_t = beta_0_t.unsqueeze(1)  # [batch, 1, 1]
                        beta_1_t = beta_1_t.unsqueeze(1)  # [batch, 1, 1]
                        beta_0_s = beta_0_s.unsqueeze(1)  # [batch, 1, 1]
                        beta_1_s = beta_1_s.unsqueeze(1)  # [batch, 1, 1]
                        
                        # Compute f_t coefficients
                        f_t_regular = (beta_0_t - beta_0_s) / beta_0_t  # [batch, 1, 1]
                        f_t_important = (beta_1_t - beta_1_s) / beta_1_t  # [batch, 1, 1]
                        
                        # Extract regular and important token probabilities
                        p_regular = p_y0[:, :, :self.vocab_size]  # [batch, seq_len, vocab_size]
                        p_important = p_y0[:, :, self.vocab_size:2*self.vocab_size]  # [batch, seq_len, vocab_size]
                        
                        # Compute g_i = p(i,1) / (p(i,0) + p(i,1)) for each base token i
                        p_total = p_regular + p_important  # [batch, seq_len, vocab_size]
                        g = torch.where(p_total > 0, p_important / p_total, torch.full_like(p_important, 0.5))  # [batch, seq_len, vocab_size]
                        
                        # Compute per-token coefficient: f_t_regular * (1-g_i) + f_t_important * g_i
                        coefficients = f_t_regular * (1 - g) + f_t_important * g  # [batch, seq_len, vocab_size]
                        
                        # Apply coefficients to both regular and important versions
                        q_ys = torch.zeros_like(p_y0)  # [batch, seq_len, augmented_vocab_size]
                        q_ys[:, :, :self.vocab_size] = p_regular * coefficients
                        q_ys[:, :, self.vocab_size:2*self.vocab_size] = p_important * coefficients
                        
                        # Mask token probability = 1 - sum(all other probabilities)
                        sum_regular = q_ys[:, :, :self.vocab_size].sum(dim=-1)
                        sum_important = q_ys[:, :, self.vocab_size:2*self.vocab_size].sum(dim=-1)
                        q_ys[:, :, self.mask_index] = 1 - (sum_regular + sum_important)
                        
                        # Sample new tokens
                        _y = self._sample_categorical(q_ys)
                        
                        # Keep non-masked tokens, replace masked tokens
                        copy_flag = (y_t != self.mask_index).to(y_t.dtype)
                        y_t = copy_flag * y_t + (1 - copy_flag) * _y
                else:
                    # Final step: just take argmax
                    y_t = p_y0.argmax(dim=-1)
        
        # Convert back to original space (remove importance bit)
        x_final = y_t % self.vocab_size
        return x_final
    
    def _sample_categorical(self, categorical_probs):
        """Sample from categorical distribution using Gumbel-max trick."""
        categorical_probs = categorical_probs.to(torch.float64)
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    def create_sequence_mask(self, sequence, threshold=None):
        """
        Creates importance mask for Sudoku sequences.
        A solution token is marked as important if it belongs to:
        1) Row with maximum pre-filled elements in puzzle
        2) Column with maximum pre-filled elements in puzzle  
        3) 3x3 subbox with maximum pre-filled elements in puzzle
        
        Args:
            sequence: [batch, seq_len] tensor, format [BOS] + quiz[81] + [SEP] + solution[81] + [EOS]
            threshold: Not used for Sudoku (kept for compatibility)
        
        Returns:
            importance_mask: [batch, seq_len] boolean tensor, True for important tokens
        """
        batch_size, seq_len = sequence.shape
        device = sequence.device
        
        # Initialize mask (all False)
        importance_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Extract quiz portion (positions 1-81)
        # Sequence format: [BOS](0) + quiz(1-81) + [SEP](82) + solution(83-163) + [EOS](164)
        quiz = sequence[:, 1:82]  # [batch, 81]
        
        # Reshape to 9x9 grid for each batch
        quiz_grid = quiz.view(batch_size, 9, 9)  # [batch, 9, 9]
        
        # Count pre-filled elements (non-zero) per row, column, and box
        # In Sudoku quiz: 0 = empty cell, 1-9 = pre-filled clue
        is_prefilled = (quiz_grid != 0).float()  # [batch, 9, 9]
        
        # Count per row
        row_counts = is_prefilled.sum(dim=2)  # [batch, 9] - sum over columns
        
        # Count per column
        col_counts = is_prefilled.sum(dim=1)  # [batch, 9] - sum over rows
        
        # Count per 3x3 box
        box_counts = torch.zeros(batch_size, 9, device=device)  # [batch, 9]
        for box_idx in range(9):
            box_row = (box_idx // 3) * 3
            box_col = (box_idx % 3) * 3
            box_region = is_prefilled[:, box_row:box_row+3, box_col:box_col+3]
            box_counts[:, box_idx] = box_region.sum(dim=(1, 2))
        
        # Find maximum counts
        max_row_count = row_counts.max(dim=1, keepdim=True)[0]  # [batch, 1]
        max_col_count = col_counts.max(dim=1, keepdim=True)[0]  # [batch, 1]
        max_box_count = box_counts.max(dim=1, keepdim=True)[0]  # [batch, 1]
        
        # Find which rows/cols/boxes have max count (handle ties)
        max_rows = (row_counts == max_row_count)  # [batch, 9]
        max_cols = (col_counts == max_col_count)  # [batch, 9]
        max_boxes = (box_counts == max_box_count)  # [batch, 9]
        
        # Create importance mask for solution tokens (positions 83-163)
        # For each position in the 9x9 grid, check if it's in max row/col/box
        solution_importance = torch.zeros(batch_size, 9, 9, dtype=torch.bool, device=device)
        
        for i in range(9):  # row index
            for j in range(9):  # column index
                # Determine which 3x3 box this position belongs to
                box_idx = (i // 3) * 3 + (j // 3)
                # A position is important if it's in any of the max row/col/box
                is_important = max_rows[:, i] | max_cols[:, j] | max_boxes[:, box_idx]
                solution_importance[:, i, j] = is_important
        
        # Flatten to match solution token positions
        solution_importance_flat = solution_importance.view(batch_size, 81)  # [batch, 81]
        
        # Place in full sequence mask (positions 83-163)
        importance_mask[:, 83:164] = solution_importance_flat
        
        return importance_mask
    
    def _compute_regular_fraction(self, y):
        """Compute fraction of regular tokens in sequence y (augmented space).
        Only considers solution tokens (positions 83-163).
        
        Args:
            y: Token sequence in augmented space, shape [batch, seq_len]
            
        Returns:
            r: Fraction of regular tokens (tokens in [0, vocab_size)), shape [batch]
        """
        # Regular tokens are in range [0, vocab_size)
        # Important tokens are in range [vocab_size, 2*vocab_size)
        # Mask token is at 2*vocab_size
        
        # Extract solution tokens only (positions 83-163, which is 81 tokens)
        # Sequence format: [BOS](0) + quiz(1-81) + [SEP](82) + solution(83-163) + [EOS](164)
        y_solution = y[:, 83:164]  # [batch, 81]
        
        # Count regular tokens (excluding mask tokens) in solution
        is_regular = (y_solution < self.vocab_size) & (y_solution != self.mask_index)
        is_not_mask = (y_solution != self.mask_index)
        
        # Fraction of regular tokens among non-mask solution tokens
        regular_count = is_regular.float().sum(dim=1)  # [batch]
        non_mask_count = is_not_mask.float().sum(dim=1)  # [batch]
        
        # Avoid division by zero
        r = torch.where(non_mask_count > 0, regular_count / non_mask_count, torch.zeros_like(regular_count))
        
        return r  # [batch]

def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len) # + 1e-10
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking


def topk_decoding(
        x0, 
        x0_scores,
        decoding_strategy,
        init_maskable_mask, 
        t,
        max_step,
        noise
    ):
        # decoding_strategy needs to take the form of "<topk_mode>-<schedule>"
        topk_mode, schedule = decoding_strategy.split("-")

        # select rate% not confident tokens, ~1 -> 0
        if schedule == "linear":
            rate = t / max_step
        elif schedule == "cosine":
            rate = np.cos((max_step-t) / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError
        
        # compute the cutoff length for denoising top-k positions
        cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
        # set the scores of unmaskable symbols to a large value so that they will never be selected
        _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)

        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
        else:
            raise NotImplementedError

        ### recovered tokens can also be remasked based on current scores
        masked_to_noise = lowest_k_mask
        if isinstance(noise, torch.Tensor):
            xt = x0.masked_scatter(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            xt = x0.masked_fill(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")

        return xt