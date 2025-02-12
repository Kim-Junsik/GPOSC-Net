import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
import kornia


from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text).to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)
    
        
    
class LandmarkEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed=640, n_layer=32, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0, image_size=512, grid_size = 256):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,attn_layers=Encoder(dim=n_embed, depth=n_layer),emb_dropout=embedding_dropout).to(self.device)
        self.image_size = image_size
        self.grid_size = grid_size


    def encode(self, text):
        # output of length 77
        return self(text)
    
    def _tokenize(self, coordinate):
        coordinate = [c // (self.image_size // self.grid_size) for c in coordinate ]
        return ''.join([str(int((c / self.grid_size)*100)/100)+',' for c in coordinate]) 
    
    def tokenize_coordinates(self, cond):
        tokens = [self._tokenize(coord) for coord in cond]
        tokens = self.tknz_fn(tokens)

        return tokens
    
    def forward(self, cond):
        token = self.tokenize_coordinates(cond)
        z = self.transformer(token, return_embeddings=True)
        
        print(z.shape)
        
        return z
    
class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))

class LandmarkEmbedderv1(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, args, device = "cuda"):
        super().__init__()
        self.device = device
        self.conditioning = args.conditioning
        self.LandmarkEmbedder = Node_Prediction_GCN(args, args.movement_n_node)
        if 'landmark' in self.conditioning:
            self.LandmarkEmbedder2 = Node_Prediction_GCN(args, args.landmark_n_node)

        self.in_channels = 1
        self.n_stages = args.n_stages
        if 'line' in self.conditioning:
            self.in_channels += 1
        self.SpatialRescaler = SpatialRescaler(n_stages = args.n_stages, in_channels = self.in_channels)

        
    def encode(self, landmark):
        return self(landmark)
    
    def forward(self, cond):
        concat = 0
        crossattn = 0
        embedding = dict()
        pre = cond['pre'].permute(0, 3, 1, 2)
        
        if 'line' in self.conditioning:
            line = cond['line'].permute(0, 3, 1, 2)
        
        if self.in_channels != 0:
            if 'line' in self.conditioning:
                concat = torch.cat([line, pre], 1)
            else:
                concat = pre
            concat = self.SpatialRescaler(concat).float()
            embedding['c_concat'] = concat
        
        crossattn = []
        movement = cond['movement']
        movement = movement[:,:,:2] - movement[:,:,2:4]
        crossattn.append(self.LandmarkEmbedder(movement).float())
        
        if 'landmark' in self.conditioning:
            landmark = cond['landmark']
            crossattn.append(self.LandmarkEmbedder2(landmark).float())
            
        crossattn = torch.cat(crossattn, 1)
        embedding['c_crossattn'] = crossattn
        
        return embedding

    
class LandmarkEmbedderv2(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, args, device = "cuda"):
        super().__init__()
        self.device = device
        self.conditioning = args.conditioning
        self.MovementEmbedder = Node_Prediction_GCN(args, args.movement_n_node)
        if 'landmark' in self.conditioning:
            self.LandmarkEmbedder = Node_Prediction_GCN(args, args.landmark_n_node)

        self.in_channels = 1
        self.n_stages = args.n_stages
        if 'line' in self.conditioning:
            self.in_channels += 1
        self.SpatialRescaler = SpatialRescaler(n_stages = args.n_stages, in_channels = self.in_channels)

        
    def encode(self, landmark):
        return self(landmark)
    
    def forward(self, cond):
        concat = 0
        crossattn = 0
        embedding = dict()
        pre = cond['pre'].permute(0, 3, 1, 2)
        
        if 'line' in self.conditioning:
            line = cond['line'].permute(0, 3, 1, 2)
        
        if self.in_channels != 0:
            if 'line' in self.conditioning:
                concat = torch.cat([line, pre], 1)
            else:
                concat = pre
            concat = self.SpatialRescaler(concat).float()
            embedding['c_concat'] = concat
        
        crossattn = []
        movement = cond['movement']
        movement = movement[:,:,:2] - movement[:,:,2:4]
        crossattn.append(self.MovementEmbedder(movement).float())
        
        if 'landmark' in self.conditioning:
            landmark = cond['landmark']
            crossattn.append(self.LandmarkEmbedder(landmark).float())
            
        crossattn = torch.cat(crossattn, 1)
        embedding['c_crossattn'] = crossattn
        
        return embedding

class LandmarkEmbedderv3(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, args, device = "cuda"):
        super().__init__()
        self.device = device
        self.conditioning = args.conditioning
        self.MovementEmbedder = Node_Prediction_GCN(args, args.movement_n_node)
        if 'landmark' in self.conditioning:
            self.LandmarkEmbedder = Node_Prediction_GCN(args, args.landmark_n_node)

        self.in_channels = 1
        self.n_stages = args.n_stages
        if 'line' in self.conditioning:
            self.in_channels += 1
        self.SpatialRescaler = SpatialRescaler(n_stages = args.n_stages, in_channels = self.in_channels)

        
    def encode(self, landmark):
        return self(landmark)
    
    def forward(self, cond):
        concat = 0
        crossattn = 0
        embedding = dict()
        pre = cond['pre'].permute(0, 3, 1, 2)
        
        if 'line' in self.conditioning:
            line = cond['line'].permute(0, 3, 1, 2)
        
        if self.in_channels != 0:
            if 'line' in self.conditioning:
                concat = torch.cat([line, pre], 1)
            else:
                concat = pre
            concat = self.SpatialRescaler(concat).float()
            embedding['c_concat'] = concat
        
        crossattn = []
        movement = cond['movement']
        movement = movement[:,:,:2] - movement[:,:,2:4]
        crossattn.append(self.MovementEmbedder(movement).float())
        
        if 'landmark' in self.conditioning:
            landmark = cond['landmark']
            crossattn.append(self.LandmarkEmbedder(landmark).float())
            
        crossattn = torch.cat(crossattn, 1)
        embedding['c_crossattn'] = crossattn
        
        return embedding
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
https://github.com/andrejmiscic/gcn-pytorch/blob/main/gcn/model.py
"""


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, act, act_negative):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        act_name = act.__class__.__name__.lower()
        param = act_negative
        if act_name == 'leakyrelu':
            act_name = 'leaky_relu'
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain(act_name, param))
        self.act = act
        
    def forward(self, x: torch.Tensor, adj: torch.nn.Parameter):
#         assert isinstance(adj, nn.Parameter), "adj is not nn.Parameter"
        x = self.linear(x)
        if self.act != None:
            x = self.act(x)
#         x = torch.matmul(adj, x)
        out = []
        for i in range(x.shape[0]):
            a = torch.mm(adj, x[i]).unsqueeze(0)
            out.append(a)
        out = torch.cat(out, 0)
        return out

    
class GCNLayer(nn.Module):
    # 1D dataset
    def __init__(self, in_dim, out_dim, act=None, act_negative=None):
        super(GCNLayer, self).__init__()

        self.GCNConv = GCNConv(in_dim, out_dim, act, act_negative)

    def forward(self, x, adj, Identity):
        out1 = self.GCNConv(x, adj)
        out2 = self.GCNConv(x, Identity)
        out = out1+out2
        return out


class GCNBlock(nn.Module):
    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, is_residual=False, act=None, act_negative=None):
        super(GCNBlock, self).__init__()
        
        self.layers = nn.ModuleList([])
        self.is_residual = is_residual
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i==0 else hidden_dim,
                                        out_dim if i==n_layer-1 else hidden_dim,
                                        act,
                                        act_negative))#act if i!=n_layer-1 else None))

        self.shortcut = nn.ModuleList([])
        if is_residual and in_dim != out_dim:
            self.shortcut.append(GCNLayer(in_dim, out_dim, act, act_negative))
            self.shortcut.append(nn.BatchNorm1d(27))
        else:
            self.shortcut.append(nn.Identity(hidden_dim, hidden_dim))
            
    def forward(self, x, adj, Identity):
        residual = x
        for i, layer in enumerate(self.layers):
            out = layer((x if i==0 else out), adj, Identity)
        if self.is_residual:
            for j, layer in enumerate(self.shortcut):
                if j%2 == 0:
                    residual = layer(residual, adj, Identity)
                else:
                    residual = layer(residual)
            out += residual
        return out


"""
    MODELS
"""

class Node_Prediction_GCN(nn.Module):
    """
    input_dim             --> Graph input layer channel
    hidden_dim            --> Graph hidden layer channel
    pred_dim              --> FC layer channel
    num_hidden_blocks     --> Number of hidden block
    num_hidden_layers     --> Number of hidden layer
    activation            --> Activation function
    dropout               --> Probability of dropout layer(if 0 is no use dropout layer)
    is_residual           --> Use short-cut connection?
    """ 
    def __init__(self, args, n_node):
        super(Node_Prediction_GCN, self).__init__()
        self.args = args
        self.n_node = n_node
        assert len(self.args.hidden_dim) == self.args.num_hidden_blocks
        
        self.adj = nn.Parameter(torch.full((self.n_node, self.n_node), 1/self.n_node, dtype=torch.float32))
        
        self.register_buffer('adj_mask', (torch.full((self.n_node, self.n_node), 1) - torch.eye(self.n_node)), persistent=False)
        self.register_buffer('Identity', torch.eye(self.n_node), persistent=False)

        if self.args.activation == "ReLU":
            self.act = nn.ReLU()
        elif self.args.activation == "Tanh":
            self.act = nn.Tanh()
        elif self.args.activation == "LeakyReLU":
            self.act = nn.LeakyReLU()
        else:
            raise
            
        self.blocks = nn.ModuleList([]) 
        self.bns = nn.ModuleList([]) 
        for i in range(self.args.num_hidden_blocks):
            self.blocks.append(GCNBlock(args.num_hidden_layers,
                                        args.input_dim if i==0 else args.hidden_dim[i-1],
                                        args.hidden_dim[i],
                                        args.hidden_dim[i],
                                        args.is_residual,
                                        act=self.act, 
                                        act_negative = self.args.act_negative))

        self.blocks.append(GCNBlock(args.num_hidden_layers,
                                    args.input_dim if i==0 else args.hidden_dim[i-1],
                                    args.hidden_dim[-1],
                                    args.output_dim,
                                    args.is_residual,
                                    act=self.act, 
                                    act_negative = self.args.act_negative))

        if self.args.bn:
            for i in range(self.args.num_hidden_blocks-1):
                self.bns.append(nn.BatchNorm1d(self.n_node))
                
    def _L2_norm(self, A):
        # A -> V*V
        A_norm = torch.norm(A, 2, dim=0, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A
    
    def _softmax(self, A):
        # A -> V*V
        A_softmax = nn.functional.softmax(A, 1)# N,1,V
        return A_softmax
    
    def _sigmoid(self, A):
        # A -> V*V
        A_sigmoid = nn.functional.sigmoid(A)
        return A_sigmoid
    
    def _clip(slef, A):
        A_clamp = torch.clamp(A, 0)
        return A_clamp

    def _abs(self, A):
        A_abs = torch.abs(A)
        return A_abs
    
    def forward(self, x: torch.Tensor, absolute = False, L2=False, softmax=False, sigmoid=True, clip=False):
        A = self.adj*self.adj_mask
        if absolute:
            A = self._abs(A)
        elif L2:
            A = self._L2_norm(A)
        elif softmax:
            A = self._softmax(A)
        elif sigmoid:
            A = self._sigmoid(A)
        elif clip:
            A = self._clip(A)

        for idx in range(len(self.bns)):
            x = F.dropout(x, p=self.args.dropout, training=self.training)
            x = self.blocks[idx](x, A, self.Identity)
            if self.args.bn:
                x = self.bns[idx](x)
        x = self.blocks[-1](x, A, self.Identity)
        return x
    
    
