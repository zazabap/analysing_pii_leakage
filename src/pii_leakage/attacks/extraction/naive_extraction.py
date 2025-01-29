# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ...arguments.ner_args import NERArgs
from ...arguments.sampling_args import SamplingArgs
from ..privacy_attack import ExtractionAttack
from ...models.language_model import LanguageModel, GeneratedTextList
from ...ner.tagger import Tagger
from ...ner.tagger_factory import TaggerFactory
from ...utils.output import print_highlighted
import random


# Try to understand NaiveExtractionAttack class and its attack method
import torch
import csv
import pandas as pd
import gc

class NaiveExtractionAttack(ExtractionAttack):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tagger = None

    def _get_tagger(self):
        if self._tagger is None:
            print_highlighted("Loading tagger...")
            # Using Named Entity Recognition (NER) arguments to initialize the tagger.
            ner_args = NERArgs(ner="flair", ner_model="flair/ner-english-ontonotes-large")
            self._tagger = TaggerFactory.from_ner_args(ner_args, env_args=self.env_args)
        return self._tagger

    def attack(self, lm: LanguageModel, *args, **kwargs) -> dict:
        # Setting up sampling arguments for the language model generation.
        sampling_args = SamplingArgs(N=self.attack_args.sampling_rate,
                                     seq_len=self.attack_args.seq_len,
                                     generate_verbose=True)

        # Generating text using the language model.
        generated_text: GeneratedTextList = lm.generate(sampling_args)

        # Analyzing the generated text with the tagger to extract entities.
        tagger: Tagger = self._get_tagger()
        entities = tagger.analyze([str(x) for x in generated_text])

        # Filter out the entities that are classified as the target entity class.
        pii = entities.get_by_entity_class(self.attack_args.pii_class)

        # Extracting the text of the entities.
        pii_mentions = [p.text for p in pii]

        # Counting the occurrence of each entity mention.
        result = {p: pii_mentions.count(p) for p in set(pii_mentions)}

        # Sorting the result dictionary based on the count of each entity mentions in descending order and returning it.
        return {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}

    def attack_adv(self, lm: LanguageModel, *args, **kwargs) -> dict:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_steps = 1000
        step_size = 0.1
        successful_attacks = 0
        
        embed_weights = lm.get_embedding_matrix()
        with open("CLERC_query_positive.csv", "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            lines = list(reader)
        
        total_steps = 0
        n = 0
        # ipdb.set_trace()
        prompt_list = []
        # for i in range(self.attack_args.sampling_rate):
        for i in range(len(lines)):
            if i == 639 or i==655 or i==732 or i==963: 
                continue # Skip this prompt
            target, fixed_prompt = lines[i]
            # control_prompt = self.attack_args.control_prompt_init
            input_tokens = torch.tensor(lm._tokenizer(fixed_prompt)["input_ids"], device=device)
            # attack_tokens = torch.tensor(lm.tokenizer(control_prompt)["input_ids"], device=lm.device)[1:]
            target_tokens = torch.tensor(lm._tokenizer(target)["input_ids"], device=device)[1:]
            one_hot_inputs, embeddings = self.create_one_hot_and_embeddings(input_tokens, embed_weights, device)
            one_hot_target, embeddings_target = self.create_one_hot_and_embeddings(target_tokens, embed_weights, device)

            adv_pert = torch.zeros_like(embeddings_target, requires_grad=True, device=device)
            output_str = ""

            for j in range(num_steps):
                total_steps += 1
                try:
                    loss, logits = self.calc_loss(lm._lm, embeddings, embeddings_target + adv_pert, embeddings_target, one_hot_target)
                    loss.backward()
                    grad = adv_pert.grad.data
                    adv_pert.data -= torch.sign(grad) * step_size
                    lm._lm.zero_grad()
                    adv_pert.grad.zero_()
                    tokens_pred = logits.argmax(2)
                    output_str = lm._tokenizer.decode(tokens_pred[0][3:].cpu().numpy())
                    from difflib import SequenceMatcher
                    similarity = SequenceMatcher(None, output_str, target).ratio()
                    success = similarity > 0.8  # Adjust the threshold as needed
                    if success:
                        successful_attacks += 1
                        prompt_list.append({"fixed_prompt": fixed_prompt, "target": target, "output": output_str, "similarity": similarity})
                        break
                except Exception as e:
                    print(f"Error during iteration {j}: {e}")
                    break
            
            print(f"Generated output string: {output_str} | Processing prompt index: {i}")
            # prompt_list.append(output_str)


        print(f"Number of successful prompts in the list: {len(prompt_list)}")
        df_prompt_list = pd.DataFrame(prompt_list)

        # Save the DataFrame to a CSV file
        csv_filename = 'prompt_list.csv'
        df_prompt_list.to_csv(csv_filename, index=False)
        print(f"Prompt list has been saved to {csv_filename}")
                
        torch.cuda.empty_cache()
        return

    def attack_test(self, lm: LanguageModel, *args, **kwargs) -> dict:

        df_prompt_list = pd.read_csv("prompt_list.csv")
        prompt_list = df_prompt_list['output'].tolist()
        del df_prompt_list
        gc.collect() # Garbage collection call to free up memory

        # Setting up sampling arguments for the language model generation.
        sampling_args = SamplingArgs(N=self.attack_args.sampling_rate,
                                     seq_len=self.attack_args.seq_len,
                                     generate_verbose=True)

        sampling_args.prompt = prompt_list
        torch.cuda.empty_cache()
        
        # Generating text using the language model.
        generated_text: GeneratedTextList = lm.generate_adv(sampling_args)

        # Analyzing the generated text with the tagger to extract entities.
        tagger: Tagger = self._get_tagger()
        entities = tagger.analyze([str(x) for x in generated_text])

        # Filter out the entities that are classified as the target entity class.
        pii = entities.get_by_entity_class(self.attack_args.pii_class)

        # Extracting the text of the entities.
        pii_mentions = [p.text for p in pii]

        # Counting the occurrence of each entity mention.
        result = {p: pii_mentions.count(p) for p in set(pii_mentions)}

        # Sorting the result dictionary based on the count of each entity mentions in descending order and returning it.
        return {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}

    def create_one_hot_and_embeddings(self, tokens, embed_weights, device):
        one_hot = torch.zeros(
            tokens.shape[0], embed_weights.shape[0], device=device, dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1,
            tokens.unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=device, dtype=embed_weights.dtype),
        )
        embeddings = (one_hot @ embed_weights).unsqueeze(0).data
        return one_hot, embeddings
    
    def calc_loss(self, model, embeddings, embeddings_attack, embeddings_target, targets):
        full_embeddings = torch.hstack([embeddings, embeddings_attack, embeddings_target])
        logits = model(inputs_embeds=full_embeddings).logits
        loss_slice_start = len(embeddings[0]) + len(embeddings_attack[0])
        loss = torch.nn.CrossEntropyLoss()(logits[0, loss_slice_start - 1 : -1, :], targets)
        return loss, logits[:, loss_slice_start - 4 : -1, :]
