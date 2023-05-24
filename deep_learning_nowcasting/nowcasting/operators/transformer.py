import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.nn.modules.sparse import Embedding

########## QKVNet ##########
class QKVNet(nn.Module):
    def __init__(self, depth=32):
        super(QKVNet, self).__init__()
        self.pad = (3 - 1) *1
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=depth, out_channels=depth*3, kernel_size=(3,3,3), stride=1, padding=(self.pad,1,1), bias=True),
            #nn.GroupNorm(num_groups=1, num_channels=depth*3)
        )

    def forward(self, input_tensor):
        qkvconcat = self.conv0(input_tensor)
        qkvconcat = qkvconcat[:, :, :-self.pad]
        return qkvconcat

class out(nn.Module):
    def __init__(self, depth):
        super(out, self).__init__()
        self.pad = (3 - 1) *1
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=depth, out_channels=depth, kernel_size=(3,3,3), stride=1, padding=(self.pad,1,1), bias=True),
            #nn.GroupNorm(num_groups=1, num_channels=depth)
        )

    def forward(self, input_tensor):
        out = self.conv0(input_tensor.permute(0, 2, 1, 3, 4))
        out = out[:, :, :-self.pad].permute(0, 2, 1, 3, 4)
        return out

class FeedForwardNet(nn.Module):
    def __init__(self, depth=128):
        super(FeedForwardNet, self).__init__()
        self.pad = (3 - 1) *1
        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels=depth, out_channels=depth*3, kernel_size=(3,3,3), stride=1, padding=(self.pad,1,1), bias=True),
            #nn.GroupNorm(num_groups=1, num_channels=depth*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=depth*3, out_channels=depth, kernel_size=(3,3,3), stride=1, padding=(self.pad,1,1), bias=True),
            #nn.GroupNorm(num_groups=1, num_channels=depth)
        )
        
        self.dropout1 = nn.Dropout3d(0.1)

    def forward(self, input_tensor):
        #[batch, seq, channel, height, width]
        out = self.conv0(input_tensor.permute(0, 2, 1, 3, 4))
        out = out[:, :, :-self.pad]
        out = self.dropout1(out)
        out = self.conv1(out)
        out = out[:, :, :-self.pad].permute(0, 2, 1, 3, 4)
        return out


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal(m.weight.data)

    if classname.find('ConvTranspose2d') != -1:
        init.xavier_normal(m.weight.data)


def get_angles(pos, i, d_model):
    # 这里的i等价与上面公式中的2i和2i+1
    angle_rates = 1 / np.power(10000, (2*(i // 2))/ np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model, h=128, w=226):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # 第2i项使用sin
    sines = np.sin(angle_rads[:, 0::2])
    # 第2i+1项使用cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1).astype(np.float32)
    pos_embedding = torch.from_numpy(0.5*pos_encoding)
    pos = pos_embedding.unsqueeze(2).repeat(1, 1, h * w).reshape(position, d_model, h, w).cuda()
    return pos


class PositionalEmbeddingLearned(nn.Module):
    def __init__(self, embedding_depth=128):
        super(PositionalEmbeddingLearned, self).__init__()
        self.depth = embedding_depth
        self.positional_embedding = nn.Embedding(10, self.depth).to('cuda:0')

    def forward(self, shape):
        b, c, h, w = shape
        index = torch.arange(b).to('cuda:0')
        position = self.positional_embedding(index)  # 5 * 64
        position = position.unsqueeze(2).repeat(1, 1, h * w).reshape(b, self.depth, h, w)
        return position

def get_model_name(cfg):
    if cfg.w_res:
        s_res = 'w_res-'
    else:
        s_res = 'wo_res-'
    if cfg.w_pos:
        s_pos = 'w_pos-'
        s_pos_kind = cfg.pos_kind
    else:
        s_pos = 'wo_pos-'
        s_pos_kind = 'none'
    s_num_heads = f'{cfg.n_heads}heads-'
    s_num_layers = f'{cfg.n_layers}layers-'
    s_num_dec_frames = f'dec_{cfg.dec_frames}-'
    s_model_type = '-inter' if cfg.model_type == 0 else '-extra'
    model_kind = s_num_heads + s_num_layers + s_num_dec_frames + s_res + s_pos + s_pos_kind + s_model_type
    return model_kind

if __name__ == '__main__':
    x = positional_encoding(3, 64)
    print('debug')


########## Feature Embedding ##########
class DecoderEmbedding(nn.module):
    def __init__(self, depth):
        super(DecoderEmbedding, self).__init__()
        self.conv0 = nn.sequential(
            nn.Conv3d(in_channels=16, out_channels=depth, kernel_size=(1,7,7), stride=1, padding=(0,3,3), bias=True),
            #nn.GroupNorm(num_groups=1, num_channels=depth),
            nn.LeakyReLU(0.2, inplace=True)
            )
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=depth, out_channels=depth, kernel_size=(1,5,5), stride=1, padding=(0,2,2), bias=True),
            #nn.GroupNorm(num_groups=1, num_channels=depth),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=depth, out_channels=depth, kernel_size=(1,5,5), stride=1, padding=(0,2,2), bias=True),
            #nn.GroupNorm(num_groups=1, num_channels=depth),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.depth = depth
        self.dropout = nn.Dropout3d(0.1)


    def forward(self, input_img):
        #[batch, seq, channel, height, width]
        img_ = input_img.permute(0, 2, 1, 3, 4).clone()
        
        feature_0 = self.conv0(img_)
        #feature_0 = feature_0[:, :, :-self.pad]
        feature_1 = self.conv1(feature_0)
        #feature_1 = feature_1[:, :, :-self.pad]
        feature_1 = feature_0 + feature_1
        #feature_2 = self.conv2(feature_1)
        #feature_2 = feature_1+feature_2

        
        # b, c, s, h, w = feature_2.shape
        # pos = positional_encoding(s, self.depth, h, w)
        # pos = pos.unsqueeze(0).expand(b, -1, -1, -1, -1)
        # feature_2 = feature_2 + pos.permute(0, 2, 1, 3, 4)
        
        out = self.dropout(feature_1).permute(0, 2, 1, 3, 4)
        return out
    


########## Decoder ##########
class Decoder(nn.Module):
    def __init__(self, num_layers=5, num_frames=1, model_depth=128, num_heads=4,
                 with_residual=True, with_pos=True, pos_kind='sine'):
        super(Decoder, self).__init__()
        self.depth = model_depth
        self.decoderlayer = DecoderLayer(model_depth, num_heads, with_pos=with_pos)
        self.num_layers = num_layers
        self.decoder = self.__get_clones(self.decoderlayer, self.num_layers)
        self.positionnet = PositionalEmbeddingLearned(int(model_depth/num_heads))
        self.num_frames = num_frames
        self.pos_kind = pos_kind
        self.GN = nn.GroupNorm(num_groups=1, num_channels=model_depth)

    def __get_clones(self, module, n):
        return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

    def forward(self, dec_init):    
        b, s, c, h, w = dec_init.shape
        out = dec_init
        if self.pos_kind == 'sine':
            pos_dec = positional_encoding(s, self.depth, h, w)
            pos_dec = pos_dec.unsqueeze(0).expand(b, -1, -1, -1, -1)
        elif self.pos_kind == 'learned':
            pos_dec = self.positionnet(out.shape)
            #pos_enc = self.positionnet(encoderin.shape)
        else:
            print('Positional Encoding is wrong')
            return
        for layer in self.decoder:
            out = layer(out, pos_dec)
        return self.GN(out.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)


class DecoderLayer(nn.Module):
    def __init__(self, model_depth=128, num_heads=4, with_pos=True):
        super(DecoderLayer, self).__init__()
        self.depth = model_depth
        self.depth_perhead = int(model_depth / num_heads)
        self.attention = self.__get_clones(MultiHeadAttention(self.depth_perhead, num_heads, with_pos=with_pos), 1)
        self.out = out(self.depth)
        self.feedforward = FeedForwardNet(self.depth)
        self.GN1 = nn.GroupNorm(num_groups=1, num_channels=model_depth)
        self.GN2 = nn.GroupNorm(num_groups=1, num_channels=model_depth)
        self.dropout1 = nn.Dropout3d(0.1)
        self.dropout2 = nn.Dropout3d(0.1)

    def __get_clones(self, module, n):
        return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

    def forward(self, input_tensor, pos_decoding):
        # sequence mask query self-attention
        att_layer_in = self.GN1(input_tensor.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        i = 0
        for layer in self.attention:
            att_out = layer(att_layer_in, pos_decoding, type=0)
        att_layer_out = self.dropout1(self.out(att_out).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4) + input_tensor
        
        # feedforward
        ff_in = self.GN2(att_layer_out.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        out = self.dropout2(self.feedforward(ff_in).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)+att_layer_out

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_depth=32, num_heads=4,with_pos=True):
        super(MultiHeadAttention, self).__init__()
        self.depth_perhead = head_depth
        self.num_heads = num_heads
        self.qkv =QKVNet(self.depth_perhead*self.num_heads)
        self.with_pos = with_pos
        #self.time_weighting = nn.Parameter(torch.ones(batch, height, width, self.num_heads, seq, seq))
        self.dropout1 = nn.Dropout3d(0.1)

    def forward(self, input_tensor, pos_decoding, type=0):  # encoderin换用model方式描述， 比如model = 0 or 1
        if type == 0:  # deocder--query self attention
            batch, seq, channel, height, width = input_tensor.shape
            input_tensor = input_tensor.permute(0, 2, 1, 3, 4)
            qkvconcat = self.qkv(input_tensor)
            q_feature, k_feature, v_feature =torch.split(qkvconcat, self.depth_perhead*self.num_heads, dim=1)
            if self.with_pos:
                q_feature = (q_feature + pos_decoding.permute(0, 2, 1, 3, 4))
                k_feature = (k_feature + pos_decoding.permute(0, 2, 1, 3, 4))
            q_feature = q_feature.view(batch, self.num_heads, self.depth_perhead, seq, height, width)
            k_feature = k_feature.view(batch, self.num_heads, self.depth_perhead, seq, height, width)
            v_feature = v_feature.view(batch, self.num_heads, self.depth_perhead, seq, height, width)
            
            # scaled dot product attention
            q = q_feature.permute(0, 4, 5, 1, 3, 2)#[batch, height, width, heads, seq, channel/head]
            k = k_feature.permute(0, 4, 5, 1, 2, 3)
            v = v_feature.permute(0, 4, 5, 1, 3, 2)
            attention_map = torch.matmul(q, k)/math.sqrt(self.depth_perhead)#[batch, height, width, heads seq, seq]
            #print(attention_map[0][0][0][0])


            #distribution
            # s_q = np.arange(seq)[:, np.newaxis]
            # s_k = np.arange(seq)[np.newaxis, :]
            # GD = np.exp(-(s_k-s_q)*(s_k-s_q)/2)/seq
            #GD = -(s_k-s_q)*(s_k-s_q)/(seq*seq*seq)
            # GD = torch.from_numpy(GD).unsqueeze(0).expand(batch*height*width*self.num_heads, -1, -1).view(batch, height, width, self.num_heads, seq, seq)
            # GD = torch.tensor(GD, dtype=torch.float32).cuda()
            # attention_map = attention_map + GD

            #sequence mask & sparse attention
            mask = 1- torch.triu(torch.ones((seq, seq)),diagonal=1)
            #mask = torch.triu(mask,diagonal=-1)#-1
            mask = mask.unsqueeze(0).expand(batch*height*width*self.num_heads, -1, -1).view(batch, height, width, self.num_heads, seq, seq).cuda()
            attention_map = attention_map * mask
            attention_map = attention_map.masked_fill(attention_map==0, -1e9)
            attention_map = nn.Softmax(dim=-1)(attention_map)

            #[batch, heads, seq_k, seq_q, height, width]
            attention_map_ = attention_map.permute(0, 3, 5, 4, 1, 2).contiguous().view(batch, -1, seq, height, width)
            attention_map_ = self.dropout1(attention_map_) 
            attention_map = attention_map_.view(batch, self.num_heads, seq, seq, height, width).permute(0, 4, 5, 1, 3, 2)
            attentioned_v_Feature = torch.matmul(attention_map,v).permute(0, 4, 3, 5, 1, 2).reshape(batch, seq, self.num_heads*self.depth_perhead, height, width)
        return attentioned_v_Feature
    

class TCTN(nn.Module):
    def __init__(self, num_layers, num_dec_frames, model_depth, num_heads, with_residual,
                 with_pos, pos_kind, mode, config):
        super(TCTN, self).__init__()
        self.configs = config
        
        self.decoder_embedding = DecoderEmbedding(model_depth)
        self.decoder = Decoder(num_layers=config.de_layers, model_depth=model_depth, num_heads=num_heads,
                                              num_frames=num_dec_frames, with_residual=with_residual,
                                              with_pos=with_pos, pos_kind=pos_kind)

        self.conv_last = nn.Conv3d(model_depth, config.img_channel*config.patch_size*config.patch_size,
                                   kernel_size=1, stride=1, padding=0, bias=False)

        self.task = mode
        self.num_dec_frames = num_dec_frames

    def forward(self, input_img, val_signal=1):#[batch, seq-1, channel, height, width]
        # decoder
        if val_signal == 0:
            dec_init = self.decoder_embedding(input_img)
            decoderout = self.decoder(dec_init)
            if self.configs.w_pffn == 1:
                out = self.prediction(decoderout)
            else:
                out = self.conv_last(decoderout.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
            
        else:
            for i in range(self.configs.total_length - self.configs.input_length):
                if i  == 0 :
                    dec_init = self.decoder_embedding(input_img[:, 0:self.configs.input_length])
                else:
                    #embedding_out[3+i] = new_embedding
                    #dec_init = self.encoder_embedding(embedding_out[0:4+i])
                    dec_init = torch.cat((dec_init,new_embedding),1)
                    #print(dec_init.shape)
                    #dec_init = self.decoder_embedding(input_img[0:self.configs.input_length+i])
                decoderout = self.decoder(dec_init)
                #print(decoderout.shape)
                #out = self.prediction(decoderout)

                if i < self.configs.total_length - self.configs.input_length - 1:
                    nex_img = decoderout[:,-1].unsqueeze(1)
                    if self.configs.w_pffn == 1:
                        img = self.prediction(nex_img)
                    else:
                        img = self.conv_last(nex_img.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                        #print(img.shape)
                    new_embedding = self.decoder_embedding(img)
                    #print(new_embedding.shape)
                else:
                    if self.configs.w_pffn == 1:
                        out = self.prediction(decoderout)
                    else:
                        out = self.conv_last(decoderout.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                    
        
        return out
