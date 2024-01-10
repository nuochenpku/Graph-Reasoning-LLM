import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F

# Load GPT-2 model and tokenizer
# model_name = 'gpt2'  # You can choose a different model if needed
# config = GPT2Config.from_pretrained(model_name)
# model = GPT2Model(config)
model_name = '/cpfs/user/chennuo/CN/llama-2-7b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model  = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
config = AutoConfig.from_pretrained(model_name)
class ContrastiveGPT(torch.nn.Module):
    def __init__(self, gpt_model):
        super().__init__()
        self.gpt = gpt_model
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gpt(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Get the last hidden state
        last_hidden = outputs.hidden_states[-1]
        # Apply dropout
        last_hidden_dropout = self.dropout(last_hidden)

        # Positive pairs
        positive_pairs = (last_hidden, last_hidden_dropout)

        # Negative samples (last 2-4 layers)
        negative_samples = outputs.hidden_states[-4:-1]

        return positive_pairs, negative_samples

# Initialize the contrastive model
contrastive_model = ContrastiveGPT(model)

# Example input
input_text = "Example text"
input_ids = torch.tensor(tokenizer(input_text, return_tensors='pt').input_ids)
# input_ids = torch.tensor([tokenizer.encode(input_text, return_tensors='pt')]).squeeze(0)
attention_mask = torch.tensor([[1] * len(input_ids)])

# Forward pass
positive_pairs, negative_samples = contrastive_model(input_ids, attention_mask)

# Calculate similarities
def calculate_similarity(anchor, positive, negatives):
    # Normalize the embeddings
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negatives = [F.normalize(neg, p=2, dim=1) for neg in negatives]

    # Calculate cosine similarities
    pos_similarity = (anchor * positive).sum(dim=1)
    
    neg_similarities = torch.cat([(anchor * neg).sum(dim=1, keepdim=True) for neg in negatives], dim=1)

    return pos_similarity, neg_similarities

# Get similarities
anchor, positive = positive_pairs
pos_similarity, neg_similarities = calculate_similarity(anchor, positive, negative_samples)

# Cross-entropy loss calculation
labels = torch.zeros(pos_similarity.size(0), dtype=torch.long)  # Labels for positive samples are 0
logits = torch.cat((pos_similarity.unsqueeze(1), neg_similarities), dim=1)
loss = F.cross_entropy(logits, labels)

# Loss value
print(loss.item())

# Now you can backpropagate this loss and update model weights as needed
