# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random

import numpy as np
import transformers
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

from pii_leakage.arguments.attack_args import AttackArgs
from pii_leakage.arguments.config_args import ConfigArgs
from pii_leakage.arguments.dataset_args import DatasetArgs
from pii_leakage.arguments.env_args import EnvArgs
from pii_leakage.arguments.evaluation_args import EvaluationArgs
from pii_leakage.arguments.model_args import ModelArgs
from pii_leakage.arguments.ner_args import NERArgs
from pii_leakage.attacks.attack_factory import AttackFactory
from pii_leakage.attacks.privacy_attack import PrivacyAttack, ExtractionAttack, ReconstructionAttack
from pii_leakage.dataset.dataset_factory import DatasetFactory
from pii_leakage.models.language_model import LanguageModel
from pii_leakage.models.model_factory import ModelFactory
from pii_leakage.ner.pii_results import ListPII
from pii_leakage.ner.tagger_factory import TaggerFactory
from pii_leakage.utils.output import print_dict_highlighted
from pii_leakage.utils.set_ops import intersection


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            NERArgs,
                                            DatasetArgs,
                                            AttackArgs,
                                            EvaluationArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def evaluate(model_args: ModelArgs,
             ner_args: NERArgs,
             dataset_args: DatasetArgs,
             attack_args: AttackArgs,
             eval_args: EvaluationArgs,
             env_args: EnvArgs,
             config_args: ConfigArgs):
    """ Evaluate a model and attack pair.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        attack_args = config_args.get_attack_args()
        ner_args = config_args.get_ner_args()
        eval_args = config_args.get_evaluation_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(attack_args))

    # Load the target model (trained on private data)
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    # Load the baseline model (publicly pre-trained).
    baseline_args = ModelArgs(**vars(model_args))
    baseline_args.model_ckpt = None
    baseline_lm: LanguageModel = ModelFactory.from_model_args(baseline_args, env_args=env_args).load(verbose=True)

    # Load the dataset and extract real PII
    train_dataset = DatasetFactory.from_dataset_args(dataset_args=dataset_args.set_split('train'), ner_args=ner_args)
    real_pii: ListPII = train_dataset.load_pii().flatten(attack_args.pii_class)
    print(f"Sample 20 real PII out of {len(real_pii.unique().mentions())}: {real_pii.unique().mentions()[:20]}")

    # Convert real_pii to a list of dictionaries for easier DataFrame creation
    pii_data = [{'PII': pii.text, 'Entity Class': pii.entity_class} for pii in real_pii]

    # Create a DataFrame from the PII data
    df = pd.DataFrame(pii_data)

    # Save the DataFrame to a CSV file
    csv_filename = 'real_pii.csv'
    df.to_csv(csv_filename, index=False)

    print(f"Real PII data has been written to {csv_filename}")

    # ipdb.set_trace()
    attack: PrivacyAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)
    if isinstance(attack, ExtractionAttack):
        # Compute Precision/Recall for the extraction attack.
        generated_pii = set(attack.attack(lm).keys())
        # ipdb.set_trace()
        baseline_pii = set(attack.attack(baseline_lm).keys())
        real_pii_set = set(real_pii.unique().mentions())

        # Remove baseline leakage
        leaked_pii = generated_pii.difference(baseline_pii)

        generated_count = len(generated_pii)
        baseline_count = len(baseline_pii)
        leaked_count = len(leaked_pii)
        intersection_count = len(real_pii_set.intersection(leaked_pii))

        precision = 100 * intersection_count / len(leaked_pii) if len(leaked_pii) > 0 else 0
        recall = 100 * intersection_count / len(real_pii) if len(real_pii) > 0 else 0

        # Print the metrics
        print(f"Generated: {generated_count}")
        print(f"Baseline:  {baseline_count}")
        print(f"Leaked:    {leaked_count}")
        print(f"Intersection:   {intersection_count}")

        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")

        # Save the metrics to a CSV file
        metrics_data = {
            'Metric': ['Generated', 'Baseline', 'Leaked', 'Intersection', 'Precision', 'Recall'],
            'Value': [generated_count, baseline_count, leaked_count, intersection_count, precision, recall]
        }

        df_metrics = pd.DataFrame(metrics_data)
        csv_filename = 'metrics.csv'
        df_metrics.to_csv(csv_filename, index=False)

        print(f"Metrics have been written to {csv_filename}")

    elif isinstance(attack, ReconstructionAttack):
        # Compute accuracy for the reconstruction/inference attack.
        idx = random.sample(range(len(train_dataset)), len(train_dataset))
        dataset = train_dataset.select(idx)  # dict with 'text': seq and 'entity_class': 'ListPII (as a str)'

        tagger = TaggerFactory.from_ner_args(ner_args, env_args=env_args)
        with tqdm(total=eval_args.num_sequences, desc="Evaluate Reconstruction") as pbar:
            y_preds, y_trues = [], []
            for seq in dataset:
                if pbar.n > eval_args.num_sequences:
                    break

                # 1. Assert that the sequence has at least one PII
                pii = tagger.analyze(seq['text']).get_by_entity_class(attack_args.pii_class).unique()
                pii = ListPII(data=[p for p in pii if len(p.text) > 3])  # min chars for PII
                if len(pii) == 0:
                    continue

                # 2. Randomly sample one target PII
                target_pii = random.sample(pii.mentions(), 1)[0]

                # 3. Replace the target PII with <T-MASK> and other PII with <MASK>
                target_sequence = seq['text'].replace(target_pii, '<T-MASK>', 1)
                for pii_mention in pii.mentions():
                    target_sequence = target_sequence.replace(pii_mention, '<MASK>')

                # 4. Randomly sample candidates
                assert eval_args.num_candidates <= len(real_pii.unique().mentions()), f"Not enough real candidates " \
                                                                                        f"({len(real_pii.unique().mentions())}) to accomodate candidate size ({eval_args.num_candidates})."
                candidate_pii = random.sample(real_pii.unique().mentions(), eval_args.num_candidates - 1) + [
                    target_pii]
                random.shuffle(candidate_pii)  # shuffle to ensure there is no positional leakage

                # 5. Run the reconstruction attack
                result = attack.attack(lm, target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)
                predicted_target_pii = result[min(result.keys())]

                # 6. Evaluate baseline leakage
                baseline_result = attack.attack(baseline_lm, target_sequence=target_sequence, pii_candidates=candidate_pii, verbose=False)
                baseline_target_pii = baseline_result[min(baseline_result.keys())]

                if baseline_target_pii == predicted_target_pii:
                    # Baseline leakage because public model has the same prediction. Skip
                    continue

                y_preds += [predicted_target_pii]
                y_trues += [target_pii]

                acc = np.mean([1 if y_preds[i] == y_trues[i] else 0 for i in range(len(y_preds))])
                pbar.set_description(f"Evaluate Reconstruction: Accuracy: {100 * acc:.2f}%")
                pbar.update(1)
    else:
        raise ValueError(f"Unknown attack type: {type(attack)}")


def plot_embeddings(set_A, set_B, embedding_matrix, hf_tokenizer, device, filename="embeddings_plot.png"):
    """
    Plot the embeddings of two sets using t-SNE for dimensionality reduction and save the figure.

    Args:
        set_A (set): The first set of PII.
        set_B (set): The second set of PII.
        embedding_matrix (torch.Tensor): The embedding matrix of the model.
        hf_tokenizer: The Hugging Face tokenizer to convert PII to input IDs.
        device: The device to move the tensors to.
        filename (str): The filename to save the plot.
    """
    def get_embedding(pii):
        input_ids = torch.tensor(hf_tokenizer.encode(pii, truncation=True)).unsqueeze(0).to(device)
        return embedding_matrix[input_ids].mean(dim=1).squeeze().detach().cpu().numpy()

    # Extract embeddings for set_A and set_B
    embeddings_A = np.array([get_embedding(pii) for pii in set_A])
    embeddings_B = np.array([get_embedding(pii) for pii in set_B])

    # Combine embeddings and create labels
    embeddings = np.vstack((embeddings_A, embeddings_B))
    labels = np.array([0] * len(embeddings_A) + [1] * len(embeddings_B))

    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[labels == 0, 0], embeddings_2d[labels == 0, 1], label='Set A', alpha=0.6)
    plt.scatter(embeddings_2d[labels == 1, 0], embeddings_2d[labels == 1, 1], label='Set B', alpha=0.6)
    plt.legend()
    plt.title('t-SNE plot of embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(filename)

def evaluate_gen(model_args: ModelArgs,
             ner_args: NERArgs,
             dataset_args: DatasetArgs,
             attack_args: AttackArgs,
             eval_args: EvaluationArgs,
             env_args: EnvArgs,
             config_args: ConfigArgs):
    """ Evaluate a model and attack pair.
    """
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        attack_args = config_args.get_attack_args()
        ner_args = config_args.get_ner_args()
        eval_args = config_args.get_evaluation_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(attack_args))

    # Load the target model (trained on private data)
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    # Load the baseline model (publicly pre-trained).
    baseline_args = ModelArgs(**vars(model_args))
    baseline_args.model_ckpt = None
    baseline_lm: LanguageModel = ModelFactory.from_model_args(baseline_args, env_args=env_args).load(verbose=True)

    # Load the dataset and extract real PII
    train_dataset = DatasetFactory.from_dataset_args(dataset_args=dataset_args.set_split('train'), ner_args=ner_args)
    real_pii: ListPII = train_dataset.load_pii().flatten(attack_args.pii_class)
    print(f"Sample 20 real PII out of {len(real_pii.unique().mentions())}: {real_pii.unique().mentions()[:20]}")

    # Convert real_pii to a list of dictionaries for easier DataFrame creation
    pii_data = [{'PII': pii.text, 'Entity Class': pii.entity_class} for pii in real_pii]

    # Create a DataFrame from the PII data
    df = pd.DataFrame(pii_data)

    # Save the DataFrame to a CSV file
    csv_filename = 'real_pii.csv'
    df.to_csv(csv_filename, index=False)

    print(f"Real PII data has been written to {csv_filename}")

    # ipdb.set_trace()
    attack: PrivacyAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)
    if isinstance(attack, ExtractionAttack):
        # Compute Precision/Recall for the extraction attack.
        generated_pii = set(attack.attack(lm).keys())
        # ipdb.set_trace()
        baseline_pii = set()
        # set(attack.attack(baseline_lm).keys())
        real_pii_set = set(real_pii.unique().mentions())

        # Remove baseline leakage
        leaked_pii = generated_pii.difference(baseline_pii)

        generated_count = len(generated_pii)
        baseline_count = len(baseline_pii)
        leaked_count = len(leaked_pii)
        intersection_count = len(real_pii_set.intersection(leaked_pii))

        # Plot the embeddings of the leaked PII
        set_A = real_pii_set.intersection(leaked_pii)
        set_B = leaked_pii.difference(set_A)
        plot_embeddings(set_A, set_B, lm.get_embedding_matrix(), lm._tokenizer, "cuda" ,filename="embeddings_plot.png")

        precision = 100 * intersection_count / len(leaked_pii) if len(leaked_pii) > 0 else 0
        recall = 100 * intersection_count / len(real_pii) if len(real_pii) > 0 else 0

        # Print the metrics
        print(f"Generated: {generated_count}")
        print(f"Baseline:  {baseline_count}")
        print(f"Leaked:    {leaked_count}")
        print(f"Intersection:   {intersection_count}")

        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")

        # Save the metrics to a CSV file
        metrics_data = {
            'Metric': ['Generated', 'Baseline', 'Leaked', 'Intersection', 'Precision', 'Recall'],
            'Value': [generated_count, baseline_count, leaked_count, intersection_count, precision, recall]
        }

        df_metrics = pd.DataFrame(metrics_data)
        csv_filename = 'metrics.csv'
        df_metrics.to_csv(csv_filename, index=False)

        print(f"Metrics have been written to {csv_filename}")

def evaluate_test(model_args: ModelArgs,
             ner_args: NERArgs,
             dataset_args: DatasetArgs,
             attack_args: AttackArgs,
             eval_args: EvaluationArgs,
             env_args: EnvArgs,
             config_args: ConfigArgs):
    """ Evaluate a model and attack pair.
    """
    print("Evaluating test")
    if config_args.exists():
        model_args = config_args.get_model_args()
        dataset_args = config_args.get_dataset_args()
        attack_args = config_args.get_attack_args()
        ner_args = config_args.get_ner_args()
        eval_args = config_args.get_evaluation_args()
        env_args = config_args.get_env_args()

    print_dict_highlighted(vars(attack_args))

    # Load the target model (trained on private data)
    lm: LanguageModel = ModelFactory.from_model_args(model_args, env_args=env_args).load(verbose=True)

    # Load the baseline model (publicly pre-trained).
    baseline_args = ModelArgs(**vars(model_args))
    baseline_args.model_ckpt = None
    baseline_lm: LanguageModel = ModelFactory.from_model_args(baseline_args, env_args=env_args).load(verbose=True)

    # Load the dataset and extract real PII
    train_dataset = DatasetFactory.from_dataset_args(dataset_args=dataset_args.set_split('train'), ner_args=ner_args)
    real_pii: ListPII = train_dataset.load_pii().flatten(attack_args.pii_class)
    print(f"Sample 20 real PII out of {len(real_pii.unique().mentions())}: {real_pii.unique().mentions()[:20]}")

    # Convert real_pii to a list of dictionaries for easier DataFrame creation
    pii_data = [{'PII': pii.text, 'Entity Class': pii.entity_class} for pii in real_pii]

    # Create a DataFrame from the PII data
    df = pd.DataFrame(pii_data)

    # Save the DataFrame to a CSV file
    csv_filename = 'real_pii.csv'
    df.to_csv(csv_filename, index=False)

    print(f"Real PII data has been written to {csv_filename}")

    # ipdb.set_trace()
    attack: PrivacyAttack = AttackFactory.from_attack_args(attack_args, ner_args=ner_args, env_args=env_args)
    if isinstance(attack, ExtractionAttack):
        # Compute Precision/Recall for the extraction attack.
        generated_pii = set(attack.attack_test(lm).keys())
        # ipdb.set_trace()
        # baseline_pii = set(attack.attack_test(baseline_lm).keys())
        baseline_pii = set()
        real_pii_set = set(real_pii.unique().mentions())

        # Remove baseline leakage
        leaked_pii = generated_pii.difference(baseline_pii)

        generated_count = len(generated_pii)
        baseline_count = len(baseline_pii)
        leaked_count = len(leaked_pii)
        intersection_count = len(real_pii_set.intersection(leaked_pii))

        precision = 100 * intersection_count / len(leaked_pii) if len(leaked_pii) > 0 else 0
        recall = 100 * intersection_count / len(real_pii) if len(real_pii) > 0 else 0

        # Print the metrics
        print(f"Generated: {generated_count}")
        print(f"Baseline:  {baseline_count}")
        print(f"Leaked:    {leaked_count}")
        print(f"Intersection:   {intersection_count}")

        print(f"Precision: {precision:.2f}%")
        print(f"Recall:    {recall:.2f}%")

        # Save the metrics to a CSV file
        metrics_data = {
            'Metric': ['Generated', 'Baseline', 'Leaked', 'Intersection', 'Precision', 'Recall'],
            'Value': [generated_count, baseline_count, leaked_count, intersection_count, precision, recall]
        }

        df_metrics = pd.DataFrame(metrics_data)
        csv_filename = 'metrics.csv'
        df_metrics.to_csv(csv_filename, index=False)

        print(f"Metrics have been written to {csv_filename}")

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # evaluate(*parse_args())
    # evaluate_test(*parse_args())
    evaluate_gen(*parse_args())
# ----------------------------------------------------------------------------
