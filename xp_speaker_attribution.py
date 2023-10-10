from transformers import BertTokenizerFast
import torch
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from rich import print
from renard.pipeline.core import Pipeline, Mention
from renard.pipeline.characters_extraction import Character
from renard.pipeline.quote_detection import Quote
from renard.pipeline.speaker_attribution import BertSpeakerDetector
from grimbert.datas import SpeakerAttributionDataset
from grimbert.model import SpeakerAttributionModel
from grimbert.train import train_speaker_attribution


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore


@ex.config
def config():
    batch_size: int = 8
    bert_checkpoint: str = "SpanBERT/spanbert-base-cased"
    lr: float = 1e-5
    epochs_nb: int = 2
    quote_ctx_len: int = 512
    speaker_repr_nb: int = 4
    PDNC_path: str


@ex.automain
def main(
    _run: Run,
    batch_size: int,
    bert_checkpoint: str,
    lr: float,
    epochs_nb: int,
    quote_ctx_len: int,
    speaker_repr_nb: int,
    PDNC_path: str,
):
    tokenizer = BertTokenizerFast.from_pretrained(bert_checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SpeakerAttributionDataset.from_PDNC(
        PDNC_path, quote_ctx_len, speaker_repr_nb, tokenizer
    )
    train_dataset, eval_dataset = dataset.splitted(0.8)

    weights = train_dataset.weights().to(device)
    model = SpeakerAttributionModel.from_pretrained(
        bert_checkpoint, weights=weights, segment_len=512
    ).to(device)

    model = train_speaker_attribution(
        model,
        train_dataset,
        eval_dataset,
        _run,
        # HG TrainingArgs
        output_dir="./sa-model",
        overwrite_output_dir=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs_nb,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_accumulation_steps=32,
        learning_rate=lr,
    )

    # Inference
    # ---------
    # Create eval characters in the format Renard expect
    characters = []
    for document in eval_dataset.documents:
        for speaker in document.speakers():
            speaker_mentions = [m for m in document.mentions if m.speaker == speaker]
            characters.append(
                Character(
                    frozenset(speaker),
                    [Mention(m.tokens, m.start, m.end) for m in speaker_mentions],
                )
            )

    # Create eval quotes in the format Renard expect
    quotes = []
    for document in eval_dataset.documents:
        for quote in document.quotes:
            quotes.append(Quote(quote.start, quote.end, quote.tokens))

    # Define and run pipeline
    pipeline = Pipeline([BertSpeakerDetector(model, tokenizer=tokenizer)])
    preds = []
    for document in eval_dataset.documents:
        out = pipeline(tokens=document.tokens, quotes=quotes, characters=characters)
        preds += out.speakers

    # Compute metrics
    refs = [
        quote.speaker
        for document in eval_dataset.documents
        for quote in document.quotes
    ]
    assert len(refs) > 0
    assert len(refs) == len(preds)

    accuracy = sum(
        [1 if speaker == pred else 0 for speaker, pred in zip(refs, preds)]
    ) / len(refs)

    TP = sum([1 if speaker == pred else 0 for speaker, pred in zip(refs, preds)])
    FP = sum(
        [
            1 if speaker != pred else 0
            for speaker, pred in zip(refs, preds)
            if not speaker is None
        ]
    )
    FN = sum([1 for pred in preds if pred is None])

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    _run.log_scalar("accuracy", accuracy)
    _run.log_scalar("precision", precision)
    _run.log_scalar("recall", recall)
    _run.log_scalar("f1", f1)

    print({"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy})
