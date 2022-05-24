import argparse
from collections import OrderedDict

import mmcv
import numpy as np
import torch


def convert(src, dst):
    src_model = mmcv.load(src)

    dst_state_dict = OrderedDict()
    for k, v in src_model['model'].items():
        key_name_split = k.split('.')
        name = None
        if 'backbone' in key_name_split:
            if 'backbone.stem.conv1.norm.' in k:
                name = f'backbone.bn1.{key_name_split[-1]}'
            elif 'backbone.stem.conv1.' in k:
                name = f'backbone.conv1.{key_name_split[-1]}'
            elif 'backbone.res' in k:
                weight_type = key_name_split[-1]
                res_id = int(key_name_split[1][-1]) - 1
                # deal with short cut
                if 'shortcut' in key_name_split[3]:
                    if 'shortcut' == key_name_split[-2]:
                        name = f'backbone.layer{res_id}.{key_name_split[2]}.downsample.0.{key_name_split[-1]}'
                    elif 'shortcut' == key_name_split[-3]:
                        name = f'backbone.layer{res_id}.{key_name_split[2]}.downsample.1.{key_name_split[-1]}'
                    else:
                        print(f'Unvalid key {k}')
                # deal with conv
                elif 'conv' in key_name_split[-2]:
                    conv_id = int(key_name_split[-2][-1])
                    name = f'backbone.layer{res_id}.{key_name_split[2]}.conv{conv_id}.{key_name_split[-1]}'
                # deal with BN
                elif key_name_split[-2] == 'norm':
                    conv_id = int(key_name_split[-3][-1])
                    name = f'backbone.layer{res_id}.{key_name_split[2]}.bn{conv_id}.{key_name_split[-1]}'
                else:
                    print(f'{k} is invalid')
            else:
                print(f'{k} is not converted!!')
        elif 'sem_seg_head' in key_name_split[0]:
            if 'pixel_decoder.input_proj' in k:
                level_id = int(key_name_split[3][0])
                layer_id = int(key_name_split[4][0])
                if layer_id == 0:  # conv
                    # name = f"panoptic_head.pixel_decoder.input_projs.{level_id}.conv.{key_name_split[-1]}"
                    name = f'panoptic_head.pixel_decoder.input_convs.{level_id}.conv.{key_name_split[-1]}'
                elif layer_id == 1:  # gn
                    # name = f"panoptic_head.pixel_decoder.input_projs.{level_id}.gn.{key_name_split[-1]}"
                    name = f'panoptic_head.pixel_decoder.input_convs.{level_id}.gn.{key_name_split[-1]}'
                else:
                    print(f'{k} is not converted')

            elif 'pixel_decoder.transformer.level_embed' in k:
                name = 'panoptic_head.pixel_decoder.level_encoding.weight'

            elif 'pixel_decoder.mask_features' in k:
                name = f'panoptic_head.pixel_decoder.mask_feature.{key_name_split[-1]}'

            elif 'pixel_decoder.adapter_' in k:
                lateral_id = int(key_name_split[2][-1]) - 1
                if 'norm' in key_name_split[-2]:
                    weight_type = key_name_split[-1]
                    name = f'panoptic_head.pixel_decoder.lateral_convs.{lateral_id}.gn.{weight_type}'
                elif 'adapter_' in key_name_split[-2]:
                    name = f'panoptic_head.pixel_decoder.lateral_convs.{lateral_id}.conv.weight'
                else:
                    print(f'{k} is not converted')
            elif 'pixel_decoder.layer_' in k:
                layer_id = int(key_name_split[2][-1]) - 1
                if 'norm' == key_name_split[-2]:
                    weight_type = key_name_split[-1]
                    name = f'panoptic_head.pixel_decoder.output_convs.{layer_id}.gn.{weight_type}'
                elif 'layer_' in key_name_split[-2]:
                    name = f'panoptic_head.pixel_decoder.output_convs.{layer_id}.conv.weight'
                else:
                    print(f'{k} is not converted')

            elif 'pixel_decoder.transformer.encoder.' in k:
                encoder_layer_id = int(key_name_split[5])
                if 'self_attn' in key_name_split[6]:
                    name = f'panoptic_head.pixel_decoder.encoder.layers.{encoder_layer_id}.attentions.0.{key_name_split[-2]}.{key_name_split[-1]}'

                elif 'linear' in key_name_split[-2]:
                    linear_id = int(key_name_split[-2][-1]) - 1
                    if linear_id == 0:
                        name = f'panoptic_head.pixel_decoder.encoder.layers.{encoder_layer_id}.ffns.0.layers.{linear_id}.0.{key_name_split[-1]}'
                    elif linear_id == 1:
                        name = f'panoptic_head.pixel_decoder.encoder.layers.{encoder_layer_id}.ffns.0.layers.{linear_id}.{key_name_split[-1]}'
                    else:
                        print(f'{k} is not convert')

                elif 'norm' in key_name_split[-2]:
                    norm_id = int(key_name_split[-2][-1]) - 1
                    name = f'panoptic_head.pixel_decoder.encoder.layers.{encoder_layer_id}.norms.{norm_id}.{key_name_split[-1]}'
                else:
                    print(f'{k} is not converted')

            elif 'predictor.transformer_self_attention_layers' in k:
                layer_id = int(key_name_split[3])
                if 'self_attn' in key_name_split[4]:
                    if 'in_proj' in key_name_split[-1]:
                        name = f'panoptic_head.transformer_decoder.layers.{layer_id}.attentions.1.attn.{key_name_split[-1]}'
                    elif 'out_proj' in key_name_split[-2]:
                        name = f'panoptic_head.transformer_decoder.layers.{layer_id}.attentions.1.attn.{key_name_split[-2]}.{key_name_split[-1]}'
                    else:
                        print(f'{k} is not converted')
                elif 'norm' in key_name_split[-2]:
                    name = f'panoptic_head.transformer_decoder.layers.{layer_id}.norms.1.{key_name_split[-1]}'
                else:
                    print(f'{k} is not converted')

            elif 'predictor.transformer_cross_attention_layers' in k:
                layer_id = int(key_name_split[3])
                if 'multihead_attn' in key_name_split[4]:
                    if 'in_proj' in key_name_split[-1]:
                        name = f'panoptic_head.transformer_decoder.layers.{layer_id}.attentions.0.attn.{key_name_split[-1]}'
                    elif 'out_proj' in key_name_split[-2]:
                        name = f'panoptic_head.transformer_decoder.layers.{layer_id}.attentions.0.attn.{key_name_split[-2]}.{key_name_split[-1]}'
                    else:
                        print(f'{k} is not converted')
                elif 'norm' in key_name_split[-2]:
                    name = f'panoptic_head.transformer_decoder.layers.{layer_id}.norms.0.{key_name_split[-1]}'
                else:
                    print(f'{k} is not converted')

            elif 'predictor.transformer_ffn_layers' in k:
                layer_id = int(key_name_split[3])
                if 'linear' in key_name_split[-2]:
                    linear_id = int(key_name_split[-2][-1]) - 1
                    if linear_id == 0:
                        name = f'panoptic_head.transformer_decoder.layers.{layer_id}.ffns.0.layers.0.0.{key_name_split[-1]}'
                    else:
                        name = f'panoptic_head.transformer_decoder.layers.{layer_id}.ffns.0.layers.{linear_id}.{key_name_split[-1]}'

                elif 'norm' in key_name_split[-2]:
                    name = f'panoptic_head.transformer_decoder.layers.{layer_id}.norms.2.{key_name_split[-1]}'
                else:
                    print(f'{k} is not converted')

            elif 'predictor.decoder_norm' in k:
                name = f'panoptic_head.transformer_decoder.post_norm.{key_name_split[-1]}'

            elif 'predictor.query_embed' in k:
                name = f'panoptic_head.query_embed.weight'
            elif 'predictor.static_query' in k or 'query_feat' in k:
                name = f'panoptic_head.query_feat.weight'
            elif 'predictor.level_embed' in k:
                name = f'panoptic_head.level_embed.weight'

            elif 'sem_seg_head.predictor.class_embed' in k:
                name = f'panoptic_head.cls_embed.{key_name_split[-1]}'

            elif 'predictor.mask_embed' in k:
                layer_id = int(key_name_split[-2]) * 2
                weight_type = key_name_split[-1]
                name = f'panoptic_head.mask_embed.{layer_id}.{weight_type}'
            else:
                print(f'{k} is not converted')
        else:
            print(f'{k} is not converted!!')

        if name is None:
            continue

        if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
            raise ValueError(
                'Unsupported type found in checkpoint! {}: {}'.format(
                    k, type(v)))
        if not isinstance(v, torch.Tensor):
            v = torch.from_numpy(v)
        dst_state_dict[name] = v

    mmdet_model = dict(state_dict=dst_state_dict, meta=dict())
    torch.save(mmdet_model, dst)


def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
