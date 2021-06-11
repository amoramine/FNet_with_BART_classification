from transformers import BartForSequenceClassification, BartConfig, AutoTokenizer
import torch
from transformers import BartConfig, PretrainedBartModel
from torch import nn 
from torch.nn import functional as F
from datasets import load_dataset
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from prettytable import PrettyTable
import plac



class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)


def fourier_transform(x):
    return torch.fft.fft2(x, dim=(-1, -2)).real


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(f"Total Trainable Params: {total_params}")
    return total_params



class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * d_model
        self.fc1 = nn.Linear(d_model, num_hidden)
        self.fc2 = nn.Linear(num_hidden, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        

    def forward(self, x):
        residual = x
        x = fourier_transform(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.final_layer_norm(x)
        return out



class FNet(nn.TransformerEncoder):
    def __init__(
        self, d_model=1024, expansion_factor=4, dropout=0.5, num_layers=6,
    ):

        encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)
        config = BartConfig()

        self.dropout = config.dropout

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.embed_scale = 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(d_model, 1)
        self.fc2 = nn.Linear(d_model, 1)


    def forward(self, x):
        # x represents input_ids
        input_shape = x.size()
        inputs_embeds = self.embed_tokens(x) * self.embed_scale
        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        x = F.dropout(hidden_states, p=self.dropout, training=self.training)
        for layer in self.layers:
            x = layer(x)
        x = self.fc1(x)
        x2 = x.reshape([x.shape[0], x.shape[2], x.shape[1]])
        x = self.fc2(x2)
        m = nn.Sigmoid()
        x = m(x)
        return x


def evaluate(model, tokenizer, dataset, device):
    total_matches = 0.0
    total_samples = 0.0
    for data in dataset:
        inputs = tokenizer(data['sentence'], return_tensors='pt', padding='max_length', max_length=1024)['input_ids'].to(device)
        labels = torch.FloatTensor([data['label']]).to(device)
        outputs = model(inputs)
        outputs = outputs.reshape([outputs.shape[0]])
        pred = torch.FloatTensor([1 if e>0.5 else 0 for e in outputs]).to(device)
        total_matches += sum(labels==pred)
        total_samples += 1
    acc = total_matches/total_samples
    return acc
    
    
def init_weights_with_BART(model, encoder_num_layers):

    bart_config = BartConfig()
    bart_classification_model = BartForSequenceClassification(bart_config)
    sd = model.state_dict().copy()
    
    for i in range(encoder_num_layers):
        sd['layers.'+str(i)+'.fc1.weight'] = bart_classification_model.state_dict()['model.encoder.layers.'+str(i)+'.fc1.weight']
        sd['layers.'+str(i)+'.fc1.bias'] = bart_classification_model.state_dict()['model.encoder.layers.'+str(i)+'.fc1.bias']
        sd['layers.'+str(i)+'.fc2.weight'] = bart_classification_model.state_dict()['model.encoder.layers.'+str(i)+'.fc2.weight']
        sd['layers.'+str(i)+'.fc2.bias'] = bart_classification_model.state_dict()['model.encoder.layers.'+str(i)+'.fc2.bias']
        sd['layers.'+str(i)+'.final_layer_norm.weight'] = bart_classification_model.state_dict()['model.encoder.layers.'+str(i)+'.final_layer_norm.weight']
        sd['layers.'+str(i)+'.final_layer_norm.bias'] = bart_classification_model.state_dict()['model.encoder.layers.'+str(i)+'.final_layer_norm.bias']

    model.load_state_dict(sd)
    return model


def main():

    # to set the device dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    
    encoder_num_layers = 1 # BART's original encoder has 6 layers
    model = FNet(num_layers = encoder_num_layers)
    model = init_weights_with_BART(model, encoder_num_layers).to(device)
    
#    uncomment this line if you can train the model on a cluster of GPUs and adapt the number of devices
#    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    dataset = load_dataset('glue', 'sst2')        

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    batch_size = 16 #should be >= num_gpus in parallel data processing

    trainloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=False)

    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        total_matches = 0.0
        total_samples = 0.0
        log_counter = 10
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['sentence']
            labels = data['label']

            labels = labels.float()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = tokenizer(inputs, return_tensors='pt', padding='max_length', max_length=1024)
            
            inputs = inputs['input_ids'].to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = outputs.reshape([outputs.shape[0]])

            pred = [1 if e>0.5 else 0 for e in outputs]
            pred = torch.FloatTensor(pred).to(device)

            matches = sum(labels==pred)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_matches += matches
            total_samples += batch_size
            
            if i % log_counter == 0:    # print every (log_counter * batch_size) mini-batches
                valid_acc = evaluate(model, tokenizer, dataset['validation'], device)
                num_mini_batches_passed = log_counter * batch_size
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / num_mini_batches_passed))
                running_loss = 0.0
                print("valid_acc: ", valid_acc)
                print("train_acc : ", total_matches/total_samples)
    print('Finished Training')


if __name__ == "__main__":
    plac.call(main)
