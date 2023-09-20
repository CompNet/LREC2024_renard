import functools
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import precision_score, recall_score, f1_score
from datasets import load_dataset
from sacred import Experiment
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
from more_itertools import flatten
from rich import print
from renard.pipeline.core import Pipeline
from renard.pipeline.ner import BertNamedEntityRecognizer
from ner import tokenize_and_align_labels


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore


@ex.config
def config():
    batch_size: int
    bert_checkpoint: str = "bert-base-cased"


@ex.automain
def main(_run: Run, batch_size: int, bert_checkpoint: str):
    tokenizer = BertTokenizerFast.from_pretrained(bert_checkpoint)

    # load dataset
    dataset = load_dataset(
        "csv",
        data_files={"train": "./data/ner/train.csv", "test": "./data/ner/test.csv"},
        # A surprising hack for sure, but a welcome one
        converters={"tokens": eval, "labels": eval},
    )
    label_list = ["O", "B-PER", "I-PER"]

    # save ref_tags and tokens now since BERT tokenization will
    # destroy that information
    ref_tags = list(flatten(list(dataset["test"]["labels"])))
    ref_tags = [label_list[tag_index] for tag_index in ref_tags]
    sentences = [example["tokens"] for example in dataset["test"]]
    tokens = list(flatten(sentences))

    # BERT tokenizer splits tokens into subtokens. The
    # tokenize_and_align_labels function correctly aligns labels and
    # subtokens.
    dataset = dataset.map(
        functools.partial(tokenize_and_align_labels, tokenizer=tokenizer), batched=True
    )

    model = BertForTokenClassification.from_pretrained(
        bert_checkpoint,
        num_labels=len(label_list),
        id2label={i: label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)},
    )

    train_args = TrainingArguments(
        f"model-ner",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        # num_train_epochs=3,
        num_train_epochs=1,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        train_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Inference
    # ---------
    pipeline = Pipeline([BertNamedEntityRecognizer(model, tokenizer=tokenizer)])
    out = pipeline(sentences=sentences, tokens=tokens)

    # Convert out.entities to bio tags since that's what seqeval takes
    pred_tags = ["O"] * len(tokens)
    for entity in out.entities:
        pred_tags[entity.start_idx] = "B-PER"
        for i in range(entity.start_idx + 1, entity.end_idx):
            pred_tags[i] = "I-PER"

    precision = precision_score([ref_tags], [pred_tags])
    recall = recall_score([ref_tags], [pred_tags])
    f1 = f1_score([ref_tags], [pred_tags])

    _run.log_scalar("precision", precision)
    _run.log_scalar("recall", recall)
    _run.log_scalar("f1", f1)

    print({"precision": precision, "recall": recall, "f1": f1})
