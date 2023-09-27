from typing import Optional
from transformers import BertTokenizerFast
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from rich import print
from tibert import (
    BertForCoreferenceResolution,
    load_litbank_dataset,
    Mention,
    CoreferenceDocument,
)
from tibert.train import train_coref_model
from tibert.score import score_coref_predictions
from renard.pipeline import Pipeline
from renard.pipeline.corefs import BertCoreferenceResolver


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore


@ex.config
def config():
    litbank_path: str
    max_span_size: int = 10
    mentions_per_tokens: float = 0.4
    segment_size: int = 128
    sents_per_documents_train: int = 10
    sents_per_documents_test = None
    batch_size: int = 1
    epochs_nb: int = 30
    bert_lr: float = 1e-5
    task_lr: float = 2e-4


@ex.automain
def main(
    _run: Run,
    litbank_path: str,
    max_span_size: int,
    mentions_per_tokens: float,
    segment_size: int,
    sents_per_documents_train: int,
    sents_per_documents_test: Optional[int],
    bert_lr: float,
    task_lr: float,
    batch_size: int,
    epochs_nb: int,
):

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    dataset = load_litbank_dataset(litbank_path, tokenizer, max_span_size)
    train_dataset, test_dataset = dataset.splitted(0.8)
    train_dataset.limit_doc_size_(sents_per_documents_train)
    if not sents_per_documents_test is None:
        test_dataset.limit_doc_size_(sents_per_documents_test)

    model = BertForCoreferenceResolution.from_pretrained(
        "bert-base-cased",
        max_span_size=max_span_size,
        mentions_per_tokens=mentions_per_tokens,
        segment_size=segment_size,
    )

    # Training
    # --------
    model = train_coref_model(
        model,
        train_dataset,
        test_dataset,
        tokenizer,
        batch_size=batch_size,
        epochs_nb=epochs_nb,
        bert_lr=bert_lr,
        task_lr=task_lr,
        _run=_run,
    )

    # Inference
    # ---------
    pipeline = Pipeline(
        [BertCoreferenceResolver(model, tokenizer=tokenizer, batch_size=batch_size)]
    )
    preds = []
    for document in test_dataset.documents:
        out = pipeline(tokens=document.tokens)
        pred = CoreferenceDocument(
            document.tokens,
            [
                [
                    Mention(mention.tokens, mention.start_idx, mention.end_idx)
                    for mention in chain
                ]
                for chain in out.corefs
            ],
        )
        preds.append(pred)

    # Scoring
    # -------
    metrics = score_coref_predictions(preds, test_dataset.documents)

    for metric_name, metric_dict in metrics.items():
        for metric_key, value in metric_dict.items():
            _run.log_scalar(f"{metric_name}.{metric_key}", value)

    print(metrics)
