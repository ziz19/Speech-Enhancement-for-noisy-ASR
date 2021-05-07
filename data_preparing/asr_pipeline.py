import torch
import torchaudio
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR

run_opts = {"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech",
                                           savedir="./pretrained_ASR", run_opts=run_opts)

def fine_tune_prep(path):
    @sb.utils.data_pipeline.takes("transcript", "mixture_path")
    @sb.utils.data_pipeline.provides(
            "signal", "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def text_prep(words, enhanced_wav):
        enhanced_sig, sample_rate = torchaudio.load(enhanced_wav, channels_first=False)
        enhanced_sig = asr_model.audio_normalizer(enhanced_sig, sample_rate)
        yield enhanced_sig
        yield words
        tokens_list = asr_model.tokenizer.encode_as_ids(words)
        yield tokens_list
        tokens_bos = torch.LongTensor([asr_model.hparams.bos_index] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(
            tokens_list + [asr_model.hparams.eos_index])  # we use same eos and bos indexes as in pretrained model
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    fine_tune_dataset = {}
    for dataset in ["train", "dev", "test"]:
        fine_tune_dataset[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=path[dataset],
            dynamic_items=[text_prep],
            output_keys=["id", "signal", "words", "tokens_list", "tokens_bos", "tokens_eos", "tokens"]
        )

    return fine_tune_dataset
