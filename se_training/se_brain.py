import speechbrain as sb
import torch


# Brain class for speech enhancement training
class SEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Apply masking to convert from noisy waveforms to enhanced signals.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : dict
            A dictionary with keys {"spec", "wav"} with predicted features.
        """

        # We first move the batch to the appropriate device, and
        # compute the features necesary for masking.
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.noisy_sig
        noisy_feats = self.compute_feats(noisy_wavs)

        # Masking is done here with the "signal approximation (SA)" algorithm.
        # The masked input is compared directly with clean speech targets.
        mask = self.modules.model(noisy_feats)
        predict_spec = torch.mul(mask, noisy_feats)

        # Also return predicted wav, for evaluation. Note that this could
        # also be used for a time-domain loss term.
        predict_wav = self.hparams.resynth(
            torch.expm1(predict_spec), noisy_wavs
        )

        # Return a dictionary so we don't have to remember the order
        return {"spec": predict_spec, "wav": predict_wav}

    def compute_feats(self, wavs):
        """Returns corresponding log-spectral features of the input waveforms.

        Arguments
        ---------
        wavs : torch.Tensor
            The batch of waveforms to convert to log-spectral features.
        """

        # Log-spectral features
        feats = self.hparams.compute_STFT(wavs)
        feats = sb.processing.features.spectral_magnitude(feats, power=0.5)

        # Log1p reduces the emphasis on small differences
        feats = torch.log1p(feats)

        return feats

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : dict
            The output dict from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        # Prepare clean targets for comparison
        clean_wavs, lens = batch.clean_sig
        clean_spec = self.compute_feats(clean_wavs)

        # Directly compare the masked spectrograms with the clean targets
        loss = sb.nnet.losses.mse_loss(predictions["spec"], clean_spec, lens)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions["spec"], clean_spec, lens, reduction="batch"
        )

        # Some evaluations are slower, and we only want to perform them
        # on the validation set.
        if stage != sb.Stage.TRAIN:

            # Evaluate speech intelligibility as an additional metric
            self.stoi_metric.append(
                batch.id,
                predictions["wav"],
                clean_wavs,
                lens,
                reduction="batch",
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.mse_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.stoi_metric = sb.utils.metric_stats.MetricStats(
                metric=sb.nnet.loss.stoi_loss.stoi_loss
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "stoi": -self.stoi_metric.summarize("average"),
            }

        # At the end of validation, we can write stats and checkpoints
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best STOI score.
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["stoi"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )