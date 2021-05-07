import speechbrain as sb

def dataio_prep(metadata_path):
    """Takes the metadata paths for train, dev, and test. Returns the a dict of these datasets with audio loaded"""
    @sb.utils.data_pipeline.takes("mixture_path", "source_1_path")
    @sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")
    def audio_pipeline(noisy_wav, clean_wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        noisy_sig = sb.dataio.dataio.read_audio(noisy_wav)
        clean_sig = sb.dataio.dataio.read_audio(clean_wav)
        return noisy_sig, clean_sig

    # Define datasets
    datasets = {}
    for dataset in ["train", "dev", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=metadata_path[dataset],
            dynamic_items=[audio_pipeline],
            output_keys=["id", "noisy_sig", "clean_sig"],
        )
    return datasets




