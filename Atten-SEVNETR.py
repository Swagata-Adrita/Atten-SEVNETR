

"""
Created on Tue Feb 21 12:02:01 2023

@author: Swagata
"""
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ ,_= x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1,1)
        return x * y.expand_as(x)


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.PReLU()
        # with learnable parameters
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
    

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class conv3d_x3(nn.Module):


    def __init__(self, in_channels, out_channels):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.conv_3 = conv3d(out_channels, out_channels)
        self.skip_connection=nn.Conv3d(in_channels,out_channels,1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        return z_3 + self.skip_connection(x)

class conv3d_x2(nn.Module):


    def __init__(self, in_channels, out_channels):
        super(conv3d_x2, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.skip_connection=nn.Conv3d(in_channels,out_channels,1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_2 = self.conv_2(z_1)
        return z_2 + self.skip_connection(x)


class conv3d_x1(nn.Module):


    def __init__(self, in_channels, out_channels):
        super(conv3d_x1, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.skip_connection=nn.Conv3d(in_channels,out_channels,1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        skip_connection = self.skip_connection(x)
        # print(skip_connection.shape)
        return z_1 + skip_connection

class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x3 = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels,3,1,1),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels,3,1,1),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels,3,1,1),
            nn.PReLU(),
        )
        self.SE4 =  SELayer(out_channels, reduction=16)
        self.Att4 = Attention_block(out_channels,out_channels,out_channels // 2)

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        lhs_conv = self.SE4(lhs_conv)
        attn = self.Att4(g=rhs_up, x=lhs_conv)
        rhs_add = torch.cat((attn, lhs_conv),dim=1) 
        return self.conv_x3(rhs_add)+ rhs_up

class deconv3d_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x2, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x2= nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels,3,1,1),    
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels,3,1,1),
            nn.PReLU(),
        )
        
        self.SE3 =  SELayer(out_channels, reduction=16)
        self.Att3 = Attention_block(out_channels,out_channels,out_channels // 2)

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        lhs_conv = self.SE3(lhs_conv)
        attn = self.Att3(g=rhs_up, x=lhs_conv)
        rhs_add = torch.cat((attn, lhs_conv),dim=1) 
        return self.conv_x2(rhs_add)+ rhs_up

class deconv3d_x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x1, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x1 = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels,3,1,1),
            nn.PReLU(),
        )
        self.SE2 =  SELayer(out_channels, reduction=16)
        self.Att2 = Attention_block(out_channels,out_channels,out_channels // 2)

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        lhs_conv = self.SE2(lhs_conv)
        attn = self.Att2(g=rhs_up, x=lhs_conv)
        rhs_add = torch.cat((attn, lhs_conv),dim=1) 
        return self.conv_x1(rhs_add)+ rhs_up
        

def conv3d_as_pool(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0),
        nn.PReLU())


def deconv3d_as_up(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        nn.PReLU()
    )


class softmax_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(softmax_out, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
    
        y_conv = self.conv_2(self.conv_1(x))
        return y_conv
  


    
    


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=1024, dropout=0.4):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, 1024)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights
    
class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers   
    
class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.Dropout(0.3),
            nn.InstanceNorm3d(out_planes),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    
class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.Dropout(0.3),
            nn.InstanceNorm3d(out_planes),
            nn.LeakyReLU(True)
        )
        
    def forward(self, x):
        return self.block(x)
    
  
    
#######################################################################################################33
   
class atten_sevnetr(nn.Module):
    def __init__(self, img_shape=(128, 128, 128), input_dim=1, output_dim=1, embed_dim=768, patch_size=16, num_heads=12, dropout=0):
        super().__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16,n1*32,n1*64]
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )
        self.decoder0 = \
                nn.Sequential(
                    Conv3DBlock(input_dim, filters[0], 3),
                    Conv3DBlock(filters[0],filters[0], 3)
                 )
        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, filters[3]),
                Deconv3DBlock(filters[3], filters[2]),
                Deconv3DBlock(filters[2], filters[1])
                )
                

        self.decoder6 = \
                nn.Sequential(
                    Deconv3DBlock(embed_dim, filters[3]),
                    Deconv3DBlock(filters[3], filters[2]),
                )
            
        self.decoder9 = \
            Deconv3DBlock(embed_dim, filters[3])
            
            
        self.decoder12 = \
                        (Conv3DBlock(embed_dim, filters[4]))
                                     
  

         
        
        self.conv_1 = conv3d_x1(self.input_dim, filters[0])
        self.pool_1 = conv3d_as_pool(filters[0], filters[1])
        self.conv_2 = conv3d_x2(filters[1], filters[1])
        self.pool_2 = conv3d_as_pool(filters[1], filters[2])
        self.conv_3 = conv3d_x3(filters[2], filters[2])
        self.pool_3 = conv3d_as_pool(filters[2], filters[3])
        self.conv_4 = conv3d_x3(filters[3], filters[3])
        self.pool_4 = conv3d_as_pool(filters[3], filters[4])

        self.bottom = conv3d_x3(filters[4], filters[4])

        self.deconv_4 = deconv3d_x3(filters[4], filters[4])

        self.deconv_3 = deconv3d_x3(filters[4], filters[3])

        self.deconv_2 = deconv3d_x2(filters[3], filters[2])

        self.deconv_1 = deconv3d_x1(filters[2], filters[1])

        self.out = softmax_out(filters[1], self.output_dim)

    def forward(self, x):
        
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)      
        z0 = self.decoder0(z0)     
        z3 = self.decoder3(z3)
        z6 = self.decoder6(z6)       
        z9 = self.decoder9(z9)
        z12 = self.decoder12(z12)

        conv_1 = self.conv_1(x)
        conv_1 = conv_1+z0
        pool = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool)
        conv_2 = conv_2+z3
        pool = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool)
        conv_3 = conv_3+z6
        pool = self.pool_3(conv_3)
        conv_4 = self.conv_4(pool)
        conv_4 = z9+conv_4
        pool = self.pool_4(conv_4)
        bottom = self.bottom(pool)
        bottom = bottom+z12
        deconv = self.deconv_4(conv_4, bottom)
        deconv = self.deconv_3(conv_3, deconv)
        deconv = self.deconv_2(conv_2, deconv)
        deconv = self.deconv_1(conv_1, deconv)
        return self.out(deconv)       
    
  
