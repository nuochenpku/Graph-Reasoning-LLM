from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class DropoutLayer(nn.Module):
    """
    Head for dropout getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
#         self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dropout(features)
#         x = self.activation(x)

        return x 

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp





class LlamaForCausalLM_Contrastive(LlamaForCausalLM):
    def __init__(self, config, layers=None):
        super().__init__(config)
        self.layers = layers
        print(self.layers)
        
      
    def construct_negtives(self, cls, batch_size, attention_mask, z1, cos_sim, loss_fct, hidden_states, num_sent, labels):  
        for i in range(cls.negative_layers):
            outputs = hidden_states[-2-i]
            if cls.pooler_type =='cls':
    #             outputs = hidden_states[-2-i]
                pooler_output = outputs[:, 0]
            elif cls.pooler_type =='avg':
                pooler_output = ((outputs * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
    #         print(outputs.shape)
            pooler_output = pooler_output.view((batch_size, num_sent, outputs.size(-1))) # (bs, num_sent, hidden)
    #         r1, r2 = pooler_output[:,0], pooler_output[:,1]
            if cls.pooler_type == "cls":
                pooler_output = cls.mlp(pooler_output)
            n1, n2 = pooler_output[:,0], pooler_output[:,1]
            cos_sim1 =  cls.sim(z1.unsqueeze(1), n1.unsqueeze(0))
            cos_sim2 =  cls.sim(z1.unsqueeze(1), n2.unsqueeze(0))
            cos_sim = torch.cat((cos_sim, cos_sim1), dim=-1)
            cos_sim = torch.cat((cos_sim, cos_sim2), dim=-1)
        loss = loss_fct(cos_sim, labels)
        return loss
            
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prompt_input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        # q = outputs["last_hidden_state"]
        # D = q.shape[-1]
        # kv = torch.cat([item.unsqueeze(2) for item in outputs["hidden_states"][:]], 2)
        if self.layers:
            hidden_states = outputs['hidden_states'][self.layers]
        else:
            hidden_states = outputs["last_hidden_state"]
   
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
            print(self.config.pretraining_tp)
        else:
            logits = self.lm_head(hidden_states)
            logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )