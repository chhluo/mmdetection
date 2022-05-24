import argparse
from collections import OrderedDict

import mmcv
import numpy as np
import torch


def correct_unfold_reduction_order(x):
    x = torch.from_numpy(x)
    out_channel, in_channel = x.shape
    x = x.reshape(out_channel, 4, in_channel // 4)
    x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
    return x.cpu().numpy()


def correct_unfold_norm_order(x):
    x = torch.from_numpy(x)
    in_channel = x.shape[0]
    x = x.reshape(4, in_channel // 4)
    x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
    return x.cpu().numpy()


def convert(src, dst, ori_patch_merge=False):

    src_model = mmcv.load(src)

    dst_state_dict = OrderedDict()
    for k, v in src_model['model'].items():
        key_name_split = k.split('.')
        name = None
        if 'backbone' in key_name_split:
            if 'patch_embed.proj' in k:
                name = f'backbone.patch_embed.projection.{key_name_split[-1]}'
            elif 'patch_embed.norm' in k:
                name = k
            elif 'layers' in key_name_split[1]:
                stage_id = int(key_name_split[2])
                if 'blocks' in key_name_split[3]:
                    block_id = int(key_name_split[4])
                    if 'norm' in key_name_split[5]:
                        name = f'backbone.stages.{stage_id}.blocks.{block_id}.{key_name_split[-2]}.{key_name_split[-1]}'
                    elif 'attn' in key_name_split[5]:
                        if 'relative_position' in key_name_split[-1]:
                            name = f'backbone.stages.{stage_id}.blocks.{block_id}.attn.w_msa.{key_name_split[-1]}'
                        else:
                            name = f'backbone.stages.{stage_id}.blocks.{block_id}.attn.w_msa.{key_name_split[-2]}.{key_name_split[-1]}'
                    elif 'mlp' in key_name_split[5]:
                        fc_id = int(key_name_split[6][-1]) - 1
                        if fc_id == 0:
                            name = f'backbone.stages.{stage_id}.blocks.{block_id}.ffn.layers.0.0.{key_name_split[-1]}'
                        else:
                            name = f'backbone.stages.{stage_id}.blocks.{block_id}.ffn.layers.{fc_id}.{key_name_split[-1]}'
                    pass
                elif 'downsample' in key_name_split[3]:
                    if 'reduction' in key_name_split[4]:
                        name = f'backbone.stages.{stage_id}.downsample.reduction.weight'
                        if not ori_patch_merge:
                            v = correct_unfold_reduction_order(v)
                    elif 'norm' == key_name_split[4]:
                        name = f'backbone.stages.{stage_id}.downsample.norm.{key_name_split[-1]}'
                        if not ori_patch_merge:
                            v = correct_unfold_norm_order(v)
                    else:
                        print(f'{k} is not converted!!')
                else:
                    print(f'{k} is not converted!!')
            elif 'norm' in key_name_split[1]:
                name = k
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
    parser.add_argument('--ori_patch_merge', action='store_true')
    args = parser.parse_args()
    convert(args.src, args.dst, args.ori_patch_merge)


if __name__ == '__main__':
    main()
